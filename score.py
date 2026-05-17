import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_TOKENIZER = None
_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_KNOWN_IMOS: set = set()          # Populated in init() from label_map.json

# Require explicit "IMO" prefix — bare 7-digit numbers (order refs, part codes, etc.)
# are no longer matched and are sent directly to the neural classifier.
_IMO_RE = re.compile(r'\bIMO[\s#:\-]?(\d{7})\b', re.IGNORECASE)


def _pick_load_dir(model_root: Path) -> Path:
    if (model_root / "config.json").exists():
        return model_root

    checkpoints = []
    for p in model_root.glob("checkpoint-*"):
        if p.is_dir():
            try:
                step = int(p.name.split("-", 1)[1])
            except Exception:
                step = -1
            checkpoints.append((step, p))

    if not checkpoints:
        return model_root

    checkpoints.sort(key=lambda t: t[0])
    return checkpoints[-1][1]


def init():
    global _TOKENIZER, _MODEL, _KNOWN_IMOS
    model_root = Path(os.getenv("AZUREML_MODEL_DIR", ".")).resolve()
    load_dir = _pick_load_dir(model_root)

    _TOKENIZER = AutoTokenizer.from_pretrained(str(load_dir))
    _MODEL = AutoModelForSequenceClassification.from_pretrained(str(load_dir))
    _MODEL.to(_DEVICE)
    _MODEL.eval()

    # Load the known IMO set from label_map.json (written by train.py)
    # This powers the high-confidence regex pre-filter (Stage 1).
    label_map_path = load_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            lm = json.load(f)
        _KNOWN_IMOS = set(lm.get("frequent_imos", []))
    else:
        # Fallback: derive known IMOs from model config id2label
        id2label = getattr(_MODEL.config, "id2label", {}) or {}
        unknown = os.getenv("UNKNOWN_LABEL", "UNKNOWN")
        _KNOWN_IMOS = {v for v in id2label.values() if v != unknown}


def _parse_request(data: Union[str, bytes, Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")

    if isinstance(data, str):
        data = json.loads(data)

    if isinstance(data, dict):
        if "inputs" in data and isinstance(data["inputs"], list):
            return [x if isinstance(x, dict) else {"text": str(x)} for x in data["inputs"]]
        return [data]

    if isinstance(data, list):
        return [x if isinstance(x, dict) else {"text": str(x)} for x in data]

    return [{"text": str(data)}]


def _imo_checksum_valid(digits: str) -> bool:
    """IMO Resolution A.600(15) check-digit: weights 7,6,5,4,3,2 applied to
    digits 1-6; units digit of the sum must equal digit 7."""
    if len(digits) != 7 or not digits.isdigit():
        return False
    weights = [7, 6, 5, 4, 3, 2]
    total = sum(int(digits[i]) * weights[i] for i in range(6))
    return (total % 10) == int(digits[6])


def _regex_prefilter(text: str) -> Optional[str]:
    """
    Stage 1: Regex pre-filter.
    Returns a known IMO string (e.g. 'IMO9319466') if exactly one unambiguous
    known IMO is found in the email text, else None (routes to neural classifier).
    Guards:
      1. Explicit 'IMO' prefix required (tightened regex)
      2. IMO check-digit must be valid (Resolution A.600(15))
      3. Ambiguous matches (two different known IMOs) return None
    """
    if not _KNOWN_IMOS:
        return None
    raw_matches = _IMO_RE.findall(text)
    known_hits  = {
        f"IMO{m}" for m in raw_matches
        if _imo_checksum_valid(m) and f"IMO{m}" in _KNOWN_IMOS
    }
    return known_hits.pop() if len(known_hits) == 1 else None


def _merge_text(item: Dict[str, Any]) -> str:
    """Assemble email fields into a structured token sequence matching train.py."""
    if "text" in item and item["text"]:
        return str(item["text"])

    # Structured separator tokens mirror the training-time merge_fields() function.
    # Keeping them identical is critical — the model learned on this exact format.
    subject     = str(item.get("emailSubject",   "") or "").strip()
    addresses   = str(item.get("emailAddresses", "") or "").strip()
    attachments = str(item.get("Attachments",    "") or "").strip()
    body        = str(item.get("emailBody",      "") or "").strip()
    parts = []
    if subject:     parts.append(f"[SUBJECT] {subject}")
    if addresses:   parts.append(f"[FROM] {addresses}")
    if attachments: parts.append(f"[ATTACH] {attachments}")
    if body:        parts.append(f"[BODY] {body}")
    return " ".join(parts)


def run(data):
    if _MODEL is None or _TOKENIZER is None:
        init()

    items = _parse_request(data)
    texts = [_merge_text(x) for x in items]

    min_conf      = float(os.getenv("MIN_CONFIDENCE", "0.70"))  # 70% threshold for auto-assign
    unknown_label = os.getenv("UNKNOWN_LABEL", "UNKNOWN")
    id2label      = getattr(_MODEL.config, "id2label", {}) or {}

    results        = []
    neural_indices = []   # indices in `texts` that need the neural model
    neural_texts   = []

    # ── Stage 1: Regex pre-filter ──────────────────────────────────────────────
    # Handle emails that contain an explicit, unambiguous known IMO number.
    # These are returned immediately at near-100% confidence without a model call,
    # reducing latency and GPU load for ~40-60% of traffic.
    prefilter_slots: List[Optional[str]] = [_regex_prefilter(t) for t in texts]

    for i, explicit_imo in enumerate(prefilter_slots):
        if explicit_imo is not None:
            results.append({
                "predicted_imo":      explicit_imo,
                "confidence":         1.0,
                "raw_prediction":     explicit_imo,
                "min_confidence":     min_conf,
                "requires_human_review": False,
                "source":             "regex_prefilter",
            })
        else:
            neural_indices.append(i)
            neural_texts.append(texts[i])
            results.append(None)  # Placeholder — filled in below

    # ── Stage 2: Neural classifier for remaining emails ────────────────────────
    if neural_texts:
        encoded = _TOKENIZER(
            neural_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(_DEVICE)

        with torch.no_grad():
            outputs  = _MODEL(**encoded)
            probs    = torch.softmax(outputs.logits, dim=-1)
            confs, pred_ids = torch.max(probs, dim=-1)

        for slot_pos, (pred_id, conf) in enumerate(
            zip(pred_ids.tolist(), confs.tolist())
        ):
            orig_idx    = neural_indices[slot_pos]
            raw_label   = id2label.get(pred_id) or id2label.get(str(pred_id)) or str(pred_id)
            if raw_label in ("None", None):
                raw_label = unknown_label
            final_label = unknown_label if (conf < min_conf or raw_label == unknown_label) else raw_label
            needs_review = final_label == unknown_label or conf < min_conf

            results[orig_idx] = {
                "predicted_imo":      final_label,
                "confidence":         float(conf),
                "raw_prediction":     raw_label,
                "min_confidence":     min_conf,
                "requires_human_review": bool(needs_review),
                "source":             "neural_classifier",
            }

    return results[0] if len(results) == 1 else results