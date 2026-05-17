"""Tests for normalize_imo and merge_fields — the two text preprocessing
functions that must produce identical output in training and in score.py."""
import pytest

from train import UNKNOWN_LABEL, merge_fields, normalize_imo


# ── normalize_imo ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("9319466",         "IMO9319466"),
    ("IMO9319466",      "IMO9319466"),
    ("imo 9319466",     "IMO9319466"),
    ("IMO  9319466",    "IMO9319466"),
    ("9319466.0",       "IMO9319466"),   # pandas-style float→str leakage
    (" 9319466 ",       "IMO9319466"),
])
def test_normalize_imo_canonicalisation(raw, expected):
    assert normalize_imo(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "nan", "NaN", "   "])
def test_normalize_imo_blank_or_nan_maps_to_unknown(raw):
    assert normalize_imo(raw) == UNKNOWN_LABEL


@pytest.mark.parametrize("raw", ["123", "12345", "abc"])
def test_normalize_imo_too_short_maps_to_unknown(raw):
    assert normalize_imo(raw) == UNKNOWN_LABEL


# ── merge_fields ──────────────────────────────────────────────────────────────

def test_merge_fields_omits_empty_sections():
    text = merge_fields({"emailSubject": "X", "emailBody": "Y"})
    assert "[SUBJECT] X" in text
    assert "[BODY] Y"    in text
    assert "[FROM]"   not in text
    assert "[ATTACH]" not in text


def test_merge_fields_field_order_is_subject_from_attach_body():
    text = merge_fields({
        "emailSubject":   "s",
        "emailAddresses": "f",
        "Attachments":    "a",
        "emailBody":      "b",
    })
    # Order matters: the model was trained on this exact ordering. score.py
    # must replicate it byte-for-byte at serving time.
    assert (
        text.index("[SUBJECT]") <
        text.index("[FROM]")    <
        text.index("[ATTACH]")  <
        text.index("[BODY]")
    )


def test_merge_fields_all_empty_returns_empty_string():
    assert merge_fields({}) == ""
    assert merge_fields({"emailSubject": "", "emailBody": None}) == ""


def test_merge_fields_strips_whitespace():
    text = merge_fields({"emailSubject": "  hello  "})
    assert text == "[SUBJECT] hello"
