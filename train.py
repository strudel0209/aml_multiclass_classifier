"""
train.py — Standalone AML Command Job training script for the IMO classifier.

Submitted by the AML Command Job cell in imo_extractor_pipeline_CSV.ipynb.
Contains identical logic to the notebook cells, packaged for headless GPU execution.

Usage (local test):
    python train.py --data 01-dataset_onlyWithImoUntil2023.csv

Usage (AML job — set by the Command Job cell):
    python train.py \
        --data ${{inputs.training_data}} \
        --model-name answerdotai/ModernBERT-base \
        --output-dir ${{outputs.model_dir}} \
        --epochs 15 --batch-size 16 --lr 3e-5 --max-seq-len 2048 --min-examples 20
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split as sk_split
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import MLflowCallback


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMO multi-class classifier training")
    p.add_argument("--data",         required=True,                         help="Path to training CSV (£-separated)")
    p.add_argument("--model-name",   default="answerdotai/ModernBERT-base", help="HuggingFace model ID")
    p.add_argument("--arch-alias",   default="",                             help="Short alias for the architecture (e.g. modernbert-base, deberta-v3-base) — logged as MLflow param/tag for bake-off comparisons. Defaults to the basename of --model-name.")
    p.add_argument("--registered-model-name", default="imo-extractor-model",  help="Name under which to register the trained model in the AML model registry.")
    p.add_argument("--output-dir",   default="./outputs-ModernBERT",        help="Where to save checkpoints and final model")
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--eval-batch",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-5)
    p.add_argument("--max-seq-len",  type=int,   default=2048)
    p.add_argument("--min-examples", type=int,   default=20,                help="Min emails per IMO class")
    # Canonical training/test files (dataset_onlyWithImoUntil2023.csv, dataset_onlyWithImoFrom2023.csv)
    # use ';' as the separator with '"'-quoted fields. The legacy 01-*.csv variants used '£'.
    p.add_argument("--csv-sep",      default=";")
    p.add_argument("--test-size",    type=float, default=0.2)
    # Cap on inverse-frequency class weights. Pure 1/freq weighting on a heavy
    # long-tailed label set produces gigantic weights for 20-example classes
    # (e.g. > 100×), which destabilises early training. Clipping at 10× keeps
    # the rebalancing pressure without exploding gradients.
    p.add_argument("--class-weight-cap", type=float, default=10.0)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ── Constants ──────────────────────────────────────────────────────────────────

UNKNOWN_LABEL = "UNKNOWN"
USECOLS = ["emailSubject", "emailBody", "Attachments", "emailAddresses", "IMO"]


# ── Data utilities ─────────────────────────────────────────────────────────────

def normalize_imo(value: str) -> str:
    if value is None:
        return UNKNOWN_LABEL
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return UNKNOWN_LABEL
    s = s.replace("IMO", "").replace("imo", "")
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 6:
        return UNKNOWN_LABEL
    return "IMO" + digits


def merge_fields(row) -> str:
    """
    Structured field merge with explicit separator tokens.
    Distinguishing [ATTACH] from [BODY] lets the model learn that
    'RFQ234-22b.pdf' in attachment context signals a specific vessel —
    a latent pattern invisible to rule-based systems.
    """
    subject     = str(row.get("emailSubject",   "") or "").strip()
    addresses   = str(row.get("emailAddresses", "") or "").strip()
    attachments = str(row.get("Attachments",    "") or "").strip()
    body        = str(row.get("emailBody",      "") or "").strip()
    parts = []
    if subject:     parts.append(f"[SUBJECT] {subject}")
    if addresses:   parts.append(f"[FROM] {addresses}")
    if attachments: parts.append(f"[ATTACH] {attachments}")
    if body:        parts.append(f"[BODY] {body}")
    return " ".join(parts)


def load_and_preprocess(data_path: str, csv_sep: str, min_examples: int):
    df = pd.read_csv(data_path, sep=csv_sep, usecols=USECOLS, dtype=str, low_memory=False)
    df["label_imo"] = df["IMO"].map(normalize_imo)
    df["text"]      = df.apply(merge_fields, axis=1)
    df = df[["text", "label_imo"]].dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    dataset   = Dataset.from_pandas(df, preserve_index=False)
    imo_counts = Counter(dataset["label_imo"])

    frequent_imos = sorted([
        imo for imo, c in imo_counts.items()
        if imo != UNKNOWN_LABEL and c >= min_examples
    ])

    label_list  = frequent_imos + [UNKNOWN_LABEL]
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    num_labels  = len(label_list)

    def encode_label(example):
        imo = example["label_imo"]
        if imo == UNKNOWN_LABEL or imo_counts.get(imo, 0) < min_examples:
            imo = UNKNOWN_LABEL
        return {"label": label_to_id[imo]}

    dataset = dataset.map(encode_label)

    unknown_mapped = sum(
        1 for imo in dataset["label_imo"]
        if imo == UNKNOWN_LABEL or imo_counts.get(imo, 0) < min_examples
    )

    print(f"Rows loaded:          {len(dataset):,}")
    print(f"Named IMO classes:    {len(frequent_imos)} (≥{min_examples} examples each)")
    print(f"UNKNOWN-mapped rows:  {unknown_mapped:,}")
    print(f"num_labels:           {num_labels}")

    return dataset, label_to_id, id_to_label, num_labels, imo_counts, frequent_imos


def split_dataset(dataset, test_size: float, seed: int):
    df_labels = dataset.to_pandas()
    train_idx, val_idx = sk_split(
        range(len(df_labels)),
        test_size=test_size,
        stratify=df_labels["label"],
        random_state=seed,
    )
    train_ds = dataset.select(train_idx)
    val_ds   = dataset.select(val_idx)
    print(f"Train: {len(train_ds):,}   Val: {len(val_ds):,}")
    return train_ds, val_ds


# ── Regex pre-filter (Stage 1) ─────────────────────────────────────────────────
# Must stay byte-for-byte identical to the logic in score.py — the training-time
# `prefilter_coverage_pct` metric is meaningless if it disagrees with the
# deployed endpoint's behaviour.
_IMO_RE = re.compile(r'\bIMO[\s#:\-]?(\d{7})\b', re.IGNORECASE)


def _imo_checksum_valid(digits: str) -> bool:
    """IMO Resolution A.600(15) check-digit: weights 7,6,5,4,3,2 applied to
    digits 1-6; units digit of the sum must equal digit 7."""
    if len(digits) != 7 or not digits.isdigit():
        return False
    weights = [7, 6, 5, 4, 3, 2]
    total = sum(int(digits[i]) * weights[i] for i in range(6))
    return (total % 10) == int(digits[6])


def extract_explicit_imo(text: str, known_imo_set: set) -> str | None:
    """Returns a single unambiguous known IMO found in text, else None.
    Guards: explicit 'IMO' prefix, valid check-digit, single known hit."""
    raw_matches = _IMO_RE.findall(text)
    known_hits  = {
        f"IMO{m}" for m in raw_matches
        if _imo_checksum_valid(m) and f"IMO{m}" in known_imo_set
    }
    return known_hits.pop() if len(known_hits) == 1 else None


def log_prefilter_stats(dataset, known_imo_set: set) -> None:
    texts      = dataset.to_pandas()["text"]
    hits       = sum(1 for t in texts if extract_explicit_imo(t, known_imo_set) is not None)
    pct        = hits / len(texts) * 100
    print(f"Regex pre-filter coverage: {hits:,} / {len(texts):,} ({pct:.1f}%)")
    mlflow.log_metric("prefilter_coverage_pct", pct)


# ── Weighted trainer ───────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer that applies per-class inverse-frequency CrossEntropyLoss."""
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(
            weight=self._class_weights.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


class AzureMLSafeCallback(MLflowCallback):
    """MLflowCallback that prevents Azure ML's 500-char param limit from
    rejecting the large id2label / label2id dicts logged from model.config."""
    def setup(self, args, state, model):
        _id2label = model.config.id2label
        _label2id = model.config.label2id
        model.config.id2label = {0: f"{len(_id2label)} classes - see checkpoint config.json"}
        model.config.label2id = {"see_config_json": 0}
        try:
            super().setup(args, state, model)
        finally:
            model.config.id2label = _id2label
            model.config.label2id = _label2id


# ── Metrics ────────────────────────────────────────────────────────────────────

def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1":       f1_score(labels, preds, average="weighted"),
        }
    return compute_metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Reproducibility: seed every RNG before any model/data work ──────────
    # HF `set_seed` covers random / numpy / torch (CPU + CUDA, all devices).
    # Setting PYTHONHASHSEED also makes Python's dict/set ordering stable across
    # processes — important for the stratified split and Counter-based class
    # weights to match across DDP ranks.
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # In DDP (torchrun --nproc_per_node=N), RANK is set per process.
    # MLflow calls, file writes, and model registration must only run on rank 0;
    # the other ranks participate in training/eval but must not log or save.
    is_main_process = int(os.environ.get("RANK", "0")) == 0

    # Default the alias to the HF model basename so single-model runs still log a clean value.
    arch_alias = args.arch_alias or args.model_name.rsplit("/", 1)[-1].lower()

    if is_main_process:
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "imo-extractor-experiment"))
        mlflow.start_run()
        mlflow.log_params({
            "model_name":    args.model_name,
            "arch_alias":    arch_alias,
            "registered_model_name": args.registered_model_name,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "lr":            args.lr,
            "max_seq_len":   args.max_seq_len,
            "min_examples":  args.min_examples,
            "test_size":     args.test_size,
        })
        mlflow.set_tag("arch_alias", arch_alias)

    # ── Load & preprocess ──────────────────────────────────────────────────────
    dataset, label_to_id, id_to_label, num_labels, imo_counts, frequent_imos = \
        load_and_preprocess(args.data, args.csv_sep, args.min_examples)

    if is_main_process:
        mlflow.log_params({
            "num_labels":    num_labels,
            "named_classes": len(frequent_imos),
        })

    train_ds, val_ds = split_dataset(dataset, args.test_size, args.seed)

    known_imo_set = set(frequent_imos)
    if is_main_process:
        log_prefilter_stats(dataset, known_imo_set)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_seq_len

    def tokenize_batch(batch):
        # No padding here — DataCollatorWithPadding pads each batch to its own
        # longest sample instead of the global max (`args.max_seq_len`). For
        # ABB Marine emails averaging ~400-600 raw tokens, effective attention
        # sequence length per batch drops to ~600-800 tokens, cutting attention
        # compute by 3-5× vs. `padding="max_length"`. Matches the notebook.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
        )

    train_remove = [c for c in train_ds.column_names if c != "label"]
    val_remove   = [c for c in val_ds.column_names   if c != "label"]
    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=train_remove)
    val_ds   = val_ds.map(tokenize_batch,   batched=True, remove_columns=val_remove)

    # ── Class weights (capped inverse-frequency) ──────────────────────────────
    # Pure 1/freq weights are unbounded on long tails (the rarest class with 20
    # samples out of 100 K rows gets ~ num_labels× 100 weight while UNKNOWN gets
    # ~1). Capping at `--class-weight-cap` keeps rebalancing pressure but avoids
    # exploding gradients during the first few warm-up steps.
    label_counts_train = Counter(train_ds["label"])
    total_train        = sum(label_counts_train.values())
    raw_weights        = [
        total_train / (num_labels * max(label_counts_train.get(i, 1), 1))
        for i in range(num_labels)
    ]
    capped_weights = [min(w, args.class_weight_cap) for w in raw_weights]
    class_weights  = torch.tensor(capped_weights, dtype=torch.float)
    if is_main_process:
        n_capped = sum(1 for r, c in zip(raw_weights, capped_weights) if r > c)
        print(
            f"Class weights: raw_max={max(raw_weights):.2f} "
            f"capped_max={max(capped_weights):.2f}  "
            f"({n_capped}/{num_labels} classes clipped at {args.class_weight_cap})"
        )
        mlflow.log_metric("class_weight_raw_max",    max(raw_weights))
        mlflow.log_metric("class_weight_capped_max", max(capped_weights))
        mlflow.log_metric("class_weight_clipped_n",  n_capped)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # ── Training ───────────────────────────────────────────────────────────────
    # warmup_ratio was removed in transformers v5.2 — compute warmup_steps explicitly.
    # total_steps = epochs × ceil(train_size / per_device_batch); 6% warmup.
    _total_steps  = args.epochs * (len(train_ds) // args.batch_size)
    _warmup_steps = max(1, int(0.06 * _total_steps))

    # ── Mixed-precision autodetect ─────────────────────────────────────────────
    # Picks the best precision the host GPU supports, falling through to fp32
    # on CPU. Trained weights are identical across paths (within numerical
    # noise), so a model trained with bf16 on A100 deploys unchanged on T4 fp16
    # or on CPU fp32.
    #   • bf16 — Ampere+ (sm_80+):  A100, H100, L4, L40S  → wider range, no loss scaling
    #   • fp16 — Volta/Turing (sm_70/sm_75): V100, T4     → Tensor Cores, requires loss scaling
    #   • fp32 — everything else (incl. CPU and pre-Volta GPUs)
    _use_bf16 = (
        torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    _use_fp16 = torch.cuda.is_available() and not _use_bf16
    if torch.cuda.is_available():
        _gpu_name = torch.cuda.get_device_name(0)
        _precision = "bf16" if _use_bf16 else "fp16"
        print(f"[precision] {_gpu_name} → {_precision}")
    else:
        print("[precision] CPU → fp32")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch,
        learning_rate=args.lr,
        warmup_steps=_warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=_use_bf16,
        fp16=_use_fp16,
        dataloader_num_workers=4,
        report_to="mlflow",
        save_total_limit=3,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,         # 'tokenizer=' renamed to 'processing_class=' in transformers v5
        data_collator=DataCollatorWithPadding(tokenizer),  # dynamic per-batch padding
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
    )

    # Swap default MLflowCallback for Azure-safe version that avoids 500-char limit
    trainer.remove_callback(MLflowCallback)
    trainer.add_callback(AzureMLSafeCallback())

    trainer.train()
    metrics = trainer.evaluate()
    # ── Save final model + label map ───────────────────────────────────────────
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))          # HF Trainer already guards rank 0 internally

    if is_main_process:
        print(metrics)
        print(f"\n✓ Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"✓ Best eval_f1:    {trainer.state.best_metric:.4f}")

        tokenizer.save_pretrained(str(final_dir))

        # Persist label maps so score.py can load them at inference time
        with open(final_dir / "label_map.json", "w") as f:
            json.dump({
                "label_to_id":          label_to_id,
                "id_to_label":          {str(k): v for k, v in id_to_label.items()},
                "frequent_imos":        frequent_imos,
                "unknown_label":        UNKNOWN_LABEL,
                "min_examples_per_imo": args.min_examples,
            }, f, indent=2)

        mlflow.log_metrics({
            "final_eval_accuracy": metrics.get("eval_accuracy", 0),
            "final_eval_f1":       metrics.get("eval_f1", 0),
        })
        mlflow.log_artifact(str(final_dir / "label_map.json"))

        # Log model to MLflow model registry (auto-registered in AML)
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="imo-classifier",
            registered_model_name=args.registered_model_name,
            task="text-classification",
        )

        mlflow.end_run()
        print(f"\n✓ Training complete. Final model saved to: {final_dir}")
        print(f"  eval_accuracy={metrics.get('eval_accuracy', 0):.4f}  eval_f1={metrics.get('eval_f1', 0):.4f}")


if __name__ == "__main__":
    main()
