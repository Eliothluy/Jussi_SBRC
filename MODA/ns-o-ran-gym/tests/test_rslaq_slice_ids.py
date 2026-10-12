"""Tests for rslaq_slice_ids module."""

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.rslaq_slice_ids import (
    normalize_slice_id,
    slice_name,
    slice_id_from_name,
    get_num_slices,
)


def test_normalize_zero_based():
    assert normalize_slice_id(0) == 0
    assert normalize_slice_id(1) == 1
    assert normalize_slice_id(2) == 2


def test_normalize_one_based_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # 0-based values pass through directly
        assert normalize_slice_id(0) == 0
        assert normalize_slice_id(1) == 1
        assert normalize_slice_id(2) == 2
        # 3 is unambiguously legacy 1-based (MTC), convert to 2 with warning
        assert normalize_slice_id(3) == 2
        assert len(w) == 1
        assert "1-based" in str(w[0].message)


def test_slice_name():
    assert slice_name(0) == "eMBB"
    assert slice_name(1) == "URLLC"
    assert slice_name(2) == "MTC"


def test_slice_id_from_name():
    assert slice_id_from_name("eMBB") == 0
    assert slice_id_from_name("URLLC") == 1
    assert slice_id_from_name("MTC") == 2


def test_get_num_slices():
    assert get_num_slices() == 3


def test_invalid_slice_id():
    try:
        normalize_slice_id(99)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_normalize_zero_based()
    test_normalize_one_based_warning()
    test_slice_name()
    test_slice_id_from_name()
    test_get_num_slices()
    test_invalid_slice_id()
    print("All slice_ids tests passed.")
