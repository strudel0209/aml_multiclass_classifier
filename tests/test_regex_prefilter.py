"""Tests for the Stage-1 IMO regex and the known-set + ambiguity guards."""
import pytest

from train import _IMO_RE, extract_explicit_imo


# All three pass A.600(15); see tests/test_imo_checksum.py for the math.
KNOWN = {"IMO9319466", "IMO9703318", "IMO1234567"}


# ── Raw regex behaviour ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Please quote for vessel IMO 9319466.",
    "Subject: refit IMO9319466 next week",
    "Reference IMO#9319466 in attached PDF",
    "See IMO-9319466 invoice",
    "see imo:9319466 for details",       # case-insensitive
    "IMO\t9319466 found in body",        # tab acceptable as separator
])
def test_regex_matches_imo_prefix_variants(text):
    matches = _IMO_RE.findall(text)
    assert "9319466" in matches


@pytest.mark.parametrize("text", [
    "Bare digits 9319466 should not match",         # no IMO prefix
    "RFQ-9319466 is an order code, not an IMO",     # different prefix
    "9319466",                                       # bare
])
def test_regex_rejects_bare_digits(text):
    assert _IMO_RE.findall(text) == []


# ── extract_explicit_imo (regex + checksum + known-set + ambiguity) ───────────

def test_single_known_hit_returns_imo():
    text = "Vessel IMO 9319466 requires parts."
    assert extract_explicit_imo(text, KNOWN) == "IMO9319466"


def test_no_imo_in_text_returns_none():
    text = "Please reply with a quote ASAP."
    assert extract_explicit_imo(text, KNOWN) is None


def test_bare_digits_returns_none():
    text = "Order 9319466 has been processed."
    assert extract_explicit_imo(text, KNOWN) is None


def test_invalid_checksum_returns_none():
    # 9319467 has 'IMO' prefix and 7 digits but fails A.600(15).
    text = "Reference IMO 9319467 in the PO."
    assert extract_explicit_imo(text, KNOWN) is None


def test_valid_checksum_but_unknown_returns_none():
    # 1234567 is checksum-valid but we remove it from the known set.
    known_subset = {"IMO9319466", "IMO9703318"}
    text = "Vessel IMO1234567."
    assert extract_explicit_imo(text, known_subset) is None


def test_two_different_known_imos_is_ambiguous():
    text = "Forwarded thread: IMO 9319466 and also IMO9703318."
    assert extract_explicit_imo(text, KNOWN) is None


def test_same_known_imo_mentioned_twice_returns_it():
    text = "IMO 9319466 — confirmed. See IMO9319466 in subject."
    assert extract_explicit_imo(text, KNOWN) == "IMO9319466"


def test_empty_known_set_returns_none_even_for_valid_match():
    # Mirrors score.py::_regex_prefilter, which short-circuits to None when
    # _KNOWN_IMOS is empty.  extract_explicit_imo in train.py doesn't have that
    # early-return — it just produces an empty intersection — so we verify the
    # observable behaviour is identical.
    text = "Vessel IMO9319466."
    assert extract_explicit_imo(text, set()) is None
