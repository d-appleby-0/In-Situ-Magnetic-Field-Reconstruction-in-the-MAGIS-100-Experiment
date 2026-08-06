"""
verify_raw_field_vs_J.py
==========================
Standalone, dependency-light diagnostic. Reads parsed_fields.csv directly
from disk with plain pandas, bypassing load_and_reclassify, add_pair_id,
fit_all_scenarios, compute_predicted_vs_actual, and every other pipeline
transformation, reporting whether tube_interior field_value actually
varies with coil_density in the raw data itself.

Usage
-----
    python verify_raw_field_vs_J.py /path/to/parsed_fields.csv
    python verify_raw_field_vs_J.py /path/to/parsed_fields.csv --slice-id 1
"""

import sys
import argparse
import numpy as np
import pandas as pd


def verify_raw_field_vs_J(csv_path: str, region: str = "tube_interior",
                          slice_id: int | None = None, n_bins: int = 10):
    """
    Read the raw CSV with a plain pd.read_csv (no reclassification, no
    dedup, no pipeline logic at all) and report field_value statistics
    grouped by coil_density, for the given region.
    """
    print(f"Reading RAW csv directly (no pipeline processing): {csv_path}")
    raw = pd.read_csv(csv_path)
    print(f"  Total rows in raw CSV: {len(raw):,}")
    print(f"  Columns: {list(raw.columns)}")

    if "region" not in raw.columns:
        raise ValueError(
            "No 'region' column in this CSV -- if this is truly the raw "
            "parser output it should already have one (parse_table_files.py "
            "assigns it during parsing). If it's missing, you may be "
            "pointing at the wrong file."
        )

    sub = raw[raw["region"] == region].copy()
    print(f"  Rows with region=={region!r}: {len(sub):,}")
    if slice_id is not None:
        sub = sub[sub["slice_id"] == slice_id]
        print(f"  Rows after slice_id=={slice_id}: {len(sub):,}")

    if sub.empty:
        raise ValueError(f"No rows found for region={region!r}"
                         + (f", slice_id={slice_id}" if slice_id is not None else ""))

    n_unique_J = sub["coil_density"].nunique()
    print(f"\n  Unique coil_density values in RAW data for this region"
          + (f"/slice" if slice_id is not None else "") + f": {n_unique_J}")

    # Per-J summary stats on the RAW field_value column, completely
    # independent of any fitting, joining, or dataframe transformation.
    per_J = sub.groupby("coil_density")["field_value"].agg(["count", "min", "mean", "max"])
    per_J = per_J.sort_index()

    print(f"\n  Raw field_value statistics per coil_density (first 5, middle 5, last 5):")
    print(f"  {'coil_density':>14} {'count':>7} {'min':>12} {'mean':>12} {'max':>12}")
    print("  " + "-"*62)
    idx = per_J.index.tolist()
    show_idx = set(idx[:5]) | set(idx[len(idx)//2-2:len(idx)//2+3]) | set(idx[-5:])
    for J in sorted(show_idx):
        row = per_J.loc[J]
        print(f"  {J:>14.4f} {int(row['count']):>7} {row['min']:>12.6e} "
              f"{row['mean']:>12.6e} {row['max']:>12.6e}")

    overall_min = per_J["mean"].min()
    overall_max = per_J["mean"].max()
    overall_range_ratio = overall_max / overall_min if overall_min > 0 else float("inf")
    overall_cv = per_J["mean"].std() / per_J["mean"].mean() if per_J["mean"].mean() != 0 else float("nan")

    print(f"\n  Across all {n_unique_J} coil_density values (raw, unprocessed):")
    print(f"    mean field_value ranges from {overall_min:.6e} to {overall_max:.6e}")
    print(f"    ratio (max/min of per-J means): {overall_range_ratio:.3f}x")
    print(f"    coefficient of variation across J:  {overall_cv*100:.2f}%")

    if overall_range_ratio < 1.5:
        print(f"\n  -> The raw, unprocessed data for this region/slice already shows a "
              f"narrow ({overall_range_ratio:.2f}x) range across the full J sweep.")
        print(f"     This is not a pipeline artifact but instead a property of the simulated")
        print(f"     field itself (e.g. strong shielding attenuation), present before any")
        print(f"     of the Python pipeline (dedup, joins, fitting) ever touches the data.")
    else:
        print(f"\n  -> The raw data shows a {overall_range_ratio:.2f}x range across J -- if the")
        print(f"     predicted-vs-actual plot's True |H| axis looks much narrower than this,")
        print(f"     something is collapsing/misaligning the data somewhere downstream of")
        print(f"     this raw CSV (check load_and_reclassify's drop_duplicates step first,")
        print(f"     then the per-pair filtering, then compute_predicted_vs_actual's join).")

    return per_J


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--region", default="tube_interior")
    p.add_argument("--slice-id", type=int, default=None)
    args = p.parse_args()

    verify_raw_field_vs_J(args.csv_path, region=args.region, slice_id=args.slice_id)
