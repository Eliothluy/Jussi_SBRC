"""
RSLAQ Slice ID utilities.

Standardizes slice identifiers across the Python pipeline to 0-based indexing,
matching the ns-3 simulation output (0=eMBB, 1=URLLC, 2=MTC).
"""

import warnings
from typing import Dict

SLICE_ID_TO_NAME: Dict[int, str] = {
    0: "eMBB",
    1: "URLLC",
    2: "MTC",
}

SLICE_NAME_TO_ID: Dict[str, int] = {
    "eMBB": 0,
    "URLLC": 1,
    "MTC": 2,
}

_NUM_SLICES = len(SLICE_ID_TO_NAME)


def normalize_slice_id(raw_slice_id: int) -> int:
    """
    Convert a slice id to the canonical 0-based index.

    If a 1-based id is detected (1, 2, 3), it is converted to (0, 1, 2)
    and a warning is issued once per process.

    Args:
        raw_slice_id: The slice id as received from an external source.

    Returns:
        Canonical 0-based slice id.

    Raises:
        ValueError: If the id is outside the expected range.
    """
    if raw_slice_id in SLICE_ID_TO_NAME:
        return raw_slice_id

    if raw_slice_id in {1, 2, 3}:
        warnings.warn(
            f"Detected 1-based slice id {raw_slice_id}. "
            f"Converting to 0-based ({raw_slice_id - 1}). "
            f"Please update the producer to use 0-based ids.",
            stacklevel=2,
        )
        return raw_slice_id - 1

    raise ValueError(
        f"Invalid slice id {raw_slice_id}. Expected 0, 1, 2 (or legacy 1, 2, 3)."
    )


def slice_name(slice_id: int) -> str:
    """Return the human-readable name for a canonical slice id."""
    canonical = normalize_slice_id(slice_id)
    return SLICE_ID_TO_NAME[canonical]


def slice_id_from_name(name: str) -> int:
    """Return the canonical slice id from a human-readable name."""
    name_clean = name.strip()
    if name_clean not in SLICE_NAME_TO_ID:
        raise ValueError(f"Unknown slice name '{name}'. Expected: {list(SLICE_NAME_TO_ID.keys())}")
    return SLICE_NAME_TO_ID[name_clean]


def get_num_slices() -> int:
    """Return the number of slices (3)."""
    return _NUM_SLICES
