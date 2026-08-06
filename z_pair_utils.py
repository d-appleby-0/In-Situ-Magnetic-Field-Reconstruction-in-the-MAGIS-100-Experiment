"""
z_pair_utils.py
===============
Tags each slice_id with its z-pair grouping and provides small helpers used
throughout the rest of the per-pair pipeline (z_pair_pipeline.py,
sensor_layout_generalization.py, etc).
"""

import pandas as pd

# slice_id -> pair_id
PAIR_MAP = {
    1: "P1", 2: "P1",
    3: "P2", 4: "P2",
    5: "P3", 6: "P3",
    7: "P4", 8: "P4",
}

PAIR_IDS = ["P1", "P2", "P3", "P4"]


def add_pair_id(df: pd.DataFrame, pair_map: dict = PAIR_MAP) -> pd.DataFrame:
    """
    Return a copy of df with a new 'pair_id' column derived from slice_id.

    Raises if any slice_id has no mapping — this is deliberately loud rather
    than silently dropping rows, per the "no silent except" lesson learned
    earlier in this project.
    """
    out = df.copy()
    out["pair_id"] = out["slice_id"].map(pair_map)
    missing = out["pair_id"].isna()
    if missing.any():
        bad = sorted(out.loc[missing, "slice_id"].unique())
        raise ValueError(
            f"slice_id value(s) {bad} have no entry in pair_map. "
            f"Update PAIR_MAP in z_pair_utils.py if the slice layout changed."
        )
    return out


def pair_slices(pair_id: str, pair_map: dict = PAIR_MAP) -> list:
    """Return the slice_id values belonging to a given pair_id, sorted."""
    return sorted(k for k, v in pair_map.items() if v == pair_id)


def validate_pair_coverage(df: pd.DataFrame, pair_map: dict = PAIR_MAP) -> None:
    """
    Sanity check: every slice_id referenced in pair_map that also appears in
    df should have data, and every pair should have exactly 2 slices present.
    Warns (does not raise) since partial pair coverage may be intentional
    during incremental data collection.
    """
    import warnings
    present_slices = set(df["slice_id"].unique())
    for pid in sorted(set(pair_map.values())):
        expected = set(pair_slices(pid, pair_map))
        have = expected & present_slices
        if len(have) == 0:
            warnings.warn(f"Pair {pid}: no slices present in df (expected {sorted(expected)}).")
        elif len(have) < len(expected):
            warnings.warn(
                f"Pair {pid}: only {sorted(have)} present, expected {sorted(expected)}. "
                f"Symmetry/consistency checks for this pair will be skipped."
            )
