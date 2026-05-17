"""Tests for IMO Resolution A.600(15) check-digit validation."""
import pytest

from train import _imo_checksum_valid


# ── Verified valid IMO numbers ────────────────────────────────────────────────
# 9319466: Ever Given.  9*7 + 3*6 + 1*5 + 9*4 + 4*3 + 6*2 = 146 → digit 7 = 6  ✓
# 9703318:              9*7 + 7*6 + 0*5 + 3*4 + 3*3 + 1*2 = 128 → digit 7 = 8  ✓
# 1234567:              1*7 + 2*6 + 3*5 + 4*4 + 5*3 + 6*2 =  77 → digit 7 = 7  ✓
VALID_IMOS = ["9319466", "9703318", "1234567"]


@pytest.mark.parametrize("imo", VALID_IMOS)
def test_valid_checksums(imo):
    assert _imo_checksum_valid(imo) is True


@pytest.mark.parametrize("imo", [
    "9319467",   # last digit off by one
    "9319460",   # last digit zeroed
    "0000000",   # all-zero — sum 0, last digit 0, technically valid arithmetic
                 # — keep this in invalid block? Actually 0 == 0, it *is* valid.
                 # Remove it; see test_all_zero below.
])
def test_invalid_checksums(imo):
    if imo == "0000000":
        return  # handled in test_all_zero
    assert _imo_checksum_valid(imo) is False


def test_all_zero_is_arithmetically_valid():
    # Documents the (harmless) edge case: '0000000' passes A.600(15) by
    # construction. The known-set guard in extract_explicit_imo / score.py's
    # _regex_prefilter rejects it because no real vessel has IMO0000000.
    assert _imo_checksum_valid("0000000") is True


@pytest.mark.parametrize("bad", ["", "123", "12345678", "123456"])
def test_wrong_length(bad):
    assert _imo_checksum_valid(bad) is False


@pytest.mark.parametrize("bad", ["123A567", "IMO9319466", "9319466 ", " 9319466"])
def test_non_digit_payload(bad):
    assert _imo_checksum_valid(bad) is False
