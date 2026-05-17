"""
evaluate_on_holdout.py — Tier-5 acceptance evaluation on a temporal holdout.

Replays the exact two-stage pipeline used by score.py (regex+checksum prefilter →
neural classifier) over a holdout CSV and emits the acceptance metrics required
by the Tier-5 gate:

    * Overall accuracy on rows with a named ground-truth IMO.
    * Stage 1 (regex) precision and coverage — the ≥99 % gate.
    * Stage 2 (neural) accuracy, plus auto-assign vs. human-review cohorts.
    * Per-class P/R/F1 with P10/P50/P90 summary over named classes.
    * Expected Calibration Error (10-bin reliability diagram).
    * Top-K confusion pairs.
    * Per-row predictions parquet for ad-hoc drill-downs.

The script reuses `merge_fields`, `normalize_imo`, the regex, and the checksum
helper from `train.py` so the holdout view is byte-for-byte identical to both
training-time preprocessing and the deployed score.py.

Usage (local):
    python evaluate_on_holdout.py \\
        --checkpoint ./outputs-deBERTa \\
        --data dataset_onlyWithImoFrom2023.csv

Usage (AML component): submit with `--output-dir ${{outputs.eval_dir}}` and let
the active MLflow run (started by the parent job) absorb the metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reuse the *exact* preprocessing helpers used by train.py / score.py.
# Importing train.py is side-effect-free (no top-level main() call).
from train import (
    UNKNOWN_LABEL,
    USECOLS,
    _IMO_RE,
    _imo_checksum_valid,
    merge_fields,
    normalize_imo,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal-holdout evaluation for the IMO classifier")
    p.add_argument("--checkpoint",      required=True,                                help="Path to a trained model dir, or a parent dir containing checkpoint-*/")
    p.add_argument("--data",            default="dataset_onlyWithImoFrom2023.csv",   help="Holdout CSV path")
    p.add_argument("--csv-sep",         default=";")
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--max-seq-len",     type=int,   default=2048)
    p.add_argument("--min-confidence",  type=float, default=0.70,                     help="Auto-assign threshold (matches MIN_CONFIDENCE env var in score.py)")
    p.add_argument("--top-k-confusion", type=int,   default=50)
    p.add_argument("--output-dir",      default="./holdout-eval-outputs")
    p.add_argument("--mlflow-run-name", default=None,                                 help="Override active-run reuse; if set, force a new run with this name")
    return p.parse_args()


# ── Checkpoint resolution (mirrors score.py::_pick_load_dir) ───────────────────

def _pick_load_dir(model_root: Path) -> Path:
    """Pick the highest-step checkpoint-* subdir, or model_root itself if it
    already contains a config.json."""
    if (model_root / "config.json").exists():
        return model_root
    checkpoints: list[tuple[int, Path]] = []
    for p in model_root.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-", 1)[1])
        except Exception:
            step = -1
        checkpoints.append((step, p))
    if not checkpoints:
        return model_root
    checkpoints.sort(key=lambda t: t[0])
    return checkpoints[-1][1]


# ── Regex prefilter (byte-identical to score.py::_regex_prefilter) ─────────────

def regex_prefilter(text: str, known_imos: set[str]) -> Optional[str]:
    if not known_imos:
        return None
    raw_matches = _IMO_RE.findall(text)
    known_hits = {
        f"IMO{m}" for m in raw_matches
        if _imo_checksum_valid(m) and f"IMO{m}" in known_imos
    }
    return known_hits.pop() if len(known_hits) == 1 else None


# ── Metrics ────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Standard ECE over [0, 1] with equal-width bins.

    Returns (ECE, per-bin DataFrame for the reliability diagram artefact).
    """
    if len(confidences) == 0:
        return float("nan"), pd.DataFrame(
            columns=["bin_lower", "bin_upper", "n", "mean_confidence", "accuracy", "gap"]
        )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confidences)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Last bin is closed on the right to capture conf == 1.0.
        mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 \
            else (confidences >= lo) & (confidences <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            rows.append({"bin_lower": lo, "bin_upper": hi, "n": 0,
                         "mean_confidence": float("nan"),
                         "accuracy": float("nan"), "gap": float("nan")})
            continue
        mean_conf = float(confidences[mask].mean())
        acc       = float(correct[mask].mean())
        gap       = abs(mean_conf - acc)
        ece      += (n_bin / n_total) * gap
        rows.append({"bin_lower": lo, "bin_upper": hi, "n": n_bin,
                     "mean_confidence": mean_conf, "accuracy": acc, "gap": gap})
    return float(ece), pd.DataFrame(rows)


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return pd.DataFrame({
        "label":     labels,
        "support":   support.astype(int),
        "precision": p,
        "recall":    r,
        "f1":        f1,
    })


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + tokenizer + label map ────────────────────────────────────
    model_root = Path(args.checkpoint).resolve()
    load_dir   = _pick_load_dir(model_root)
    print(f"Loading model from: {load_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(load_dir))
    tokenizer.model_max_length = args.max_seq_len
    model = AutoModelForSequenceClassification.from_pretrained(str(load_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    label_map_path = load_dir / "label_map.json"
    if label_map_path.exists():
        lm = json.loads(label_map_path.read_text())
        known_imos: set[str] = set(lm.get("frequent_imos", []))
    else:
        id2label = getattr(model.config, "id2label", {}) or {}
        known_imos = {v for v in id2label.values() if v != UNKNOWN_LABEL}

    # id2label may be keyed by int *or* str depending on serialization round-trip.
    raw_id2label = getattr(model.config, "id2label", {}) or {}
    id2label: dict[int, str] = {int(k): v for k, v in raw_id2label.items()}
    label_to_id: dict[str, int] = {v: k for k, v in id2label.items()}

    if UNKNOWN_LABEL not in label_to_id:
        raise RuntimeError(
            f"Model config is missing the {UNKNOWN_LABEL!r} class — cannot map "
            f"out-of-distribution / low-confidence rows."
        )
    unknown_id = label_to_id[UNKNOWN_LABEL]

    print(f"Classes in model: {len(id2label)} (named: {len(known_imos)})")

    # ── Load holdout ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.data, sep=args.csv_sep, usecols=USECOLS, dtype=str, low_memory=False)
    df["label_imo"] = df["IMO"].map(normalize_imo)
    df["text"]      = df.apply(merge_fields, axis=1)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    n_total = len(df)
    print(f"Holdout rows: {n_total:,}")

    # Ground-truth bucketing:
    #   - 'named_in_dist'  : true label is a named class the model knows about.
    #   - 'named_oov'      : true label is a named IMO the model has NEVER seen
    #                        (new 2023+ vessel) — model can't get this right by
    #                        construction; tracked separately.
    #   - 'truly_unknown'  : ground truth was UNKNOWN already (no IMO recorded).
    def _bucket(lbl: str) -> str:
        if lbl == UNKNOWN_LABEL:
            return "truly_unknown"
        return "named_in_dist" if lbl in known_imos else "named_oov"
    df["gt_bucket"] = df["label_imo"].map(_bucket)
    bucket_counts = Counter(df["gt_bucket"])
    print(f"  named_in_dist: {bucket_counts['named_in_dist']:,}")
    print(f"  named_oov:     {bucket_counts['named_oov']:,}")
    print(f"  truly_unknown: {bucket_counts['truly_unknown']:,}")

    # ── Stage 1: regex prefilter ──────────────────────────────────────────────
    df["regex_hit"]    = df["text"].map(lambda t: regex_prefilter(t, known_imos))
    df["source"]       = df["regex_hit"].where(df["regex_hit"].isna(), "regex_prefilter")
    df["source"]       = df["source"].fillna("neural_classifier")
    n_regex            = int(df["regex_hit"].notna().sum())
    regex_coverage_pct = (n_regex / n_total * 100.0) if n_total else 0.0
    # Regex precision is defined over rows where regex fired AND ground truth is
    # a *named* class (truly_unknown rows have no positive label to match).
    regex_mask = df["regex_hit"].notna()
    regex_eval = df.loc[regex_mask & (df["gt_bucket"] != "truly_unknown")]
    if len(regex_eval) > 0:
        regex_correct       = int((regex_eval["regex_hit"] == regex_eval["label_imo"]).sum())
        regex_precision_pct = regex_correct / len(regex_eval) * 100.0
        regex_fp_count      = len(regex_eval) - regex_correct
    else:
        regex_precision_pct = float("nan")
        regex_fp_count      = 0
    print(f"\nStage 1 (regex):  coverage={regex_coverage_pct:.2f}%  "
          f"precision={regex_precision_pct:.2f}%  FP={regex_fp_count}")

    # ── Stage 2: neural classifier on remaining rows ──────────────────────────
    neural_df  = df.loc[~regex_mask].reset_index(drop=False)  # keep original index
    pred_label = [None] * n_total
    pred_conf  = [None] * n_total

    if len(neural_df) > 0:
        print(f"\nStage 2 (neural): scoring {len(neural_df):,} rows on {device}…")
        texts = neural_df["text"].tolist()
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start:start + args.batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
                probs  = torch.softmax(logits, dim=-1)
                confs, pred_ids = torch.max(probs, dim=-1)
            for offset, (pid, c) in enumerate(zip(pred_ids.tolist(), confs.tolist())):
                orig_idx = int(neural_df.iloc[start + offset]["index"])
                raw_lbl  = id2label.get(int(pid), UNKNOWN_LABEL)
                # Apply min-confidence gate exactly like score.py.
                final = UNKNOWN_LABEL if (c < args.min_confidence or raw_lbl == UNKNOWN_LABEL) else raw_lbl
                pred_label[orig_idx] = final
                pred_conf[orig_idx]  = float(c)

    # Fill regex slots with conf=1.0.
    for idx in df.index[regex_mask]:
        pred_label[idx] = df.at[idx, "regex_hit"]
        pred_conf[idx]  = 1.0

    df["pred_label"]  = pred_label
    df["pred_conf"]   = pred_conf
    df["correct"]     = (df["pred_label"] == df["label_imo"]).astype(int)
    df["auto_assign"] = (df["pred_conf"] >= args.min_confidence) & (df["pred_label"] != UNKNOWN_LABEL)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    overall_acc       = float(df["correct"].mean()) if n_total else float("nan")
    in_dist           = df[df["gt_bucket"] == "named_in_dist"]
    in_dist_acc       = float(in_dist["correct"].mean()) if len(in_dist) else float("nan")
    oov_named_acc     = float(df.loc[df["gt_bucket"] == "named_oov", "correct"].mean()) if bucket_counts["named_oov"] else float("nan")
    truly_unknown_acc = float(df.loc[df["gt_bucket"] == "truly_unknown", "correct"].mean()) if bucket_counts["truly_unknown"] else float("nan")

    # Neural-only cohort metrics
    neural_mask     = df["source"] == "neural_classifier"
    neural_eval     = df.loc[neural_mask & (df["gt_bucket"] != "named_oov")]  # OOV is structurally unreachable
    neural_acc      = float(neural_eval["correct"].mean()) if len(neural_eval) else float("nan")
    auto_eval       = neural_eval[neural_eval["auto_assign"]]
    review_eval     = neural_eval[~neural_eval["auto_assign"]]
    neural_auto_acc = float(auto_eval["correct"].mean())   if len(auto_eval)   else float("nan")
    neural_rev_acc  = float(review_eval["correct"].mean()) if len(review_eval) else float("nan")
    human_review_rate_pct = (len(review_eval) / len(neural_eval) * 100.0) if len(neural_eval) else float("nan")

    # ── Per-class F1 (named classes only, in-distribution rows only) ──────────
    named_labels = sorted(known_imos)
    if len(in_dist) > 0:
        pc = per_class_metrics(
            in_dist["label_imo"].to_numpy(),
            in_dist["pred_label"].to_numpy(),
            named_labels,
        )
    else:
        pc = pd.DataFrame(columns=["label", "support", "precision", "recall", "f1"])
    pc.to_csv(output_dir / "per_class_metrics.csv", index=False)

    if len(pc) and pc["support"].sum() > 0:
        present = pc[pc["support"] > 0]["f1"].to_numpy()
        f1_p10, f1_p50, f1_p90 = (
            float(np.percentile(present, 10)),
            float(np.percentile(present, 50)),
            float(np.percentile(present, 90)),
        )
        f1_macro = float(present.mean())
    else:
        f1_p10 = f1_p50 = f1_p90 = f1_macro = float("nan")

    # ── ECE on neural predictions only (regex hits are conf=1.0 by fiat) ──────
    ece_df_source = neural_eval.dropna(subset=["pred_conf"])
    ece, reliability_df = expected_calibration_error(
        ece_df_source["pred_conf"].to_numpy(dtype=float),
        ece_df_source["correct"].to_numpy(dtype=int),
        n_bins=10,
    )
    reliability_df.to_csv(output_dir / "reliability_diagram.csv", index=False)

    # ── Top-K confusion pairs ─────────────────────────────────────────────────
    confusion = (
        in_dist[in_dist["pred_label"] != in_dist["label_imo"]]
        .groupby(["label_imo", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(args.top_k_confusion)
    )
    confusion.to_csv(output_dir / "confusion_top.csv", index=False)

    # ── Per-row predictions ───────────────────────────────────────────────────
    pred_cols = ["label_imo", "gt_bucket", "source", "pred_label",
                 "pred_conf", "auto_assign", "correct"]
    df[pred_cols].to_parquet(output_dir / "predictions.parquet", index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "n_total":                       n_total,
        "n_named_in_dist":               int(bucket_counts["named_in_dist"]),
        "n_named_oov":                   int(bucket_counts["named_oov"]),
        "n_truly_unknown":               int(bucket_counts["truly_unknown"]),
        "overall_accuracy":              overall_acc,
        "in_dist_accuracy":              in_dist_acc,
        "oov_named_accuracy":            oov_named_acc,
        "truly_unknown_accuracy":        truly_unknown_acc,
        "regex_coverage_pct":            regex_coverage_pct,
        "regex_precision_pct":           regex_precision_pct,
        "regex_false_positive_count":    regex_fp_count,
        "neural_accuracy":               neural_acc,
        "neural_auto_assign_accuracy":   neural_auto_acc,
        "neural_human_review_accuracy":  neural_rev_acc,
        "human_review_rate_pct":         human_review_rate_pct,
        "f1_macro_named":                f1_macro,
        "f1_p10_named":                  f1_p10,
        "f1_p50_named":                  f1_p50,
        "f1_p90_named":                  f1_p90,
        "ece_neural_10bin":              ece,
        "min_confidence":                args.min_confidence,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n── Holdout summary ───────────────────────────────────────────")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<32s} {v:.4f}")
        else:
            print(f"  {k:<32s} {v}")
    print("──────────────────────────────────────────────────────────────")

    # ── MLflow logging ────────────────────────────────────────────────────────
    # Reuse the active run if a parent (e.g. the AML pipeline) already started
    # one; otherwise open our own. `mlflow.active_run()` returns None when no
    # run is in progress.
    active = mlflow.active_run()
    run_ctx = mlflow.start_run(
        run_name=args.mlflow_run_name or f"holdout-eval-{Path(args.data).stem}",
        nested=False,
    ) if (active is None or args.mlflow_run_name) else None

    try:
        mlflow.log_params({
            "eval_checkpoint":  str(load_dir),
            "eval_data":        os.path.abspath(args.data),
            "eval_batch_size":  args.batch_size,
            "eval_max_seq_len": args.max_seq_len,
            "min_confidence":   args.min_confidence,
        })
        for k, v in summary.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                mlflow.log_metric(f"holdout/{k}", float(v))
        for fname in ("summary.json", "per_class_metrics.csv",
                      "reliability_diagram.csv", "confusion_top.csv",
                      "predictions.parquet"):
            mlflow.log_artifact(str(output_dir / fname), artifact_path="holdout-eval")
    finally:
        if run_ctx is not None:
            mlflow.end_run()

    # ── Tier-5 acceptance gate echo (non-fatal — operator decides) ────────────
    print("\nTier-5 acceptance reference targets:")
    print(f"  in-dist accuracy           ≥ 90%   →  observed {in_dist_acc*100:.2f}%")
    print(f"  regex precision            ≥ 99%   →  observed {regex_precision_pct:.2f}%")
    print(f"  ECE (neural)               ≤ 0.05  →  observed {ece:.4f}")


if __name__ == "__main__":
    main()
