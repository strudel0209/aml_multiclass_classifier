"""End-to-end test of score.init() with a stubbed model/tokenizer: verifies the
known-IMO set is populated from `label_map.json` so the Stage-1 regex prefilter
gates the right vessels at serving time."""
import json
import os
from unittest.mock import patch

import score


def _write_minimal_checkpoint(root, frequent_imos, id2label=None):
    cp = root / "checkpoint-50"
    cp.mkdir()
    (cp / "config.json").write_text(json.dumps({
        "id2label": id2label or {},
        "label2id": {v: int(k) for k, v in (id2label or {}).items()},
    }))
    (cp / "label_map.json").write_text(json.dumps({
        "label_to_id":   {imo: i for i, imo in enumerate(frequent_imos + ["UNKNOWN"])},
        "id_to_label":   {str(i): imo for i, imo in enumerate(frequent_imos + ["UNKNOWN"])},
        "frequent_imos": frequent_imos,
        "unknown_label": "UNKNOWN",
        "min_examples_per_imo": 20,
    }))
    return cp


def _reset_score_globals():
    score._TOKENIZER = None
    score._MODEL = None
    score._KNOWN_IMOS = set()


def test_init_populates_known_imos_from_label_map(tmp_path):
    _write_minimal_checkpoint(tmp_path, ["IMO9319466", "IMO9703318"])
    _reset_score_globals()
    with patch.dict(os.environ, {"AZUREML_MODEL_DIR": str(tmp_path)}, clear=False):
        score.init()
    assert score._KNOWN_IMOS == {"IMO9319466", "IMO9703318"}


def test_init_falls_back_to_id2label_when_label_map_missing(tmp_path):
    cp = tmp_path / "checkpoint-50"
    cp.mkdir()
    (cp / "config.json").write_text(json.dumps({
        "id2label": {"0": "IMO9319466", "1": "IMO9703318", "2": "UNKNOWN"},
    }))
    # No label_map.json — score.init() must fall back to model.config.id2label.
    _reset_score_globals()
    # Stub the model's config.id2label attribute so the fallback branch sees it.
    with patch.dict(os.environ, {"AZUREML_MODEL_DIR": str(tmp_path)}, clear=False), \
         patch.object(score, "AutoModelForSequenceClassification") as mock_model_cls:
        mock_model = mock_model_cls.from_pretrained.return_value
        mock_model.config.id2label = {0: "IMO9319466", 1: "IMO9703318", 2: "UNKNOWN"}
        score.init()
    # UNKNOWN must be filtered out of the known set.
    assert score._KNOWN_IMOS == {"IMO9319466", "IMO9703318"}
