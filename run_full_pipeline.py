"""
run_full_pipeline.py
======================
raw Opera .table files -> parsed/reclassified data
z-pair tagging -> sensor layout search with cross-pair generalization(and fallback)
per-pair multipole report -> per-pair model selection, each with a cross-pair summary printed first.

Usage
-----
    python run_full_pipeline.py

or from Python:

    from run_full_pipeline import run_full_pipeline
    results = run_full_pipeline("/path/to/raw_table_files")

Pipeline stages
---------------
  1. parse_table_files.parse_all_table_files()  — raw .table -> combined df,
     saved to <out_dir>/parsed_fields.csv
  2. magis_pca_pipeline.load_and_reclassify()   — re-applies geometry
     classification + de-dup on the saved CSV (kept as a separate step,
     matching the existing pipeline's expectations; parse step already
     assigns 'region' but load_and_reclassify is the source of truth for
     dedup + the tube_wall/shield_material drop)
  3. z_pair_utils.add_pair_id() + validate_pair_coverage()
  4. z_pair_pipeline.run_all_pairs()            — per-pair PCA + multipole
     fit/CV, plus print_symmetry_report() for the within-pair consistency
     diagnostic
  5. sensor_layout_generalization.search_and_validate_layout() — search on
     one reference pair, validate + fallback on the others
  6. multipole_report_multi.generate_multi_pair_report()       — cross-pair
     summary + full per-pair report, using final_layouts from step 5
  7. multipole_model_selection_multi.run_multi_pair_model_selection() —
     cross-pair n_max recommendation summary + full per-pair Phase 4/5 detail
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from parse_table_files import parse_all_table_files, has_vector_data
from magis_pca_pipeline import load_and_reclassify
from z_pair_utils import add_pair_id, validate_pair_coverage, PAIR_IDS
from z_pair_pipeline import run_all_pairs, print_symmetry_report
from sensor_layout_generalization import (
    search_and_validate_layout, generalization_report_to_df,
)
from multipole_report_multi import generate_multi_pair_report
from multipole_model_selection_multi import run_multi_pair_model_selection
from multipole_fitter import DEFAULT_N_MAX
from plot_rmse_convergence import plot_rmse_vs_nmax_convergence
from recursive_order_refinement import compare_greedy_vs_joint


def run_full_pipeline(raw_table_dir: str | Path,
                      out_dir: str | Path | None = None,
                      n_sensors: int = 2,
                      reference_pair: str = "P1",
                      max_candidates: int | None = None,
                      degradation_tolerance: float = 0.5,
                      n_max: int = DEFAULT_N_MAX,
                      n_max_range: range = range(1, 6),
                      noise_sigma_T: float = 1e-9,
                      ridge_alpha: float = 0.1,
                      run_orientation_search: bool = True,
                      target_J=None, target_slice=None,
                      verbose: bool = True) -> dict:
    """
    Run the complete MAGIS-100 analysis pipeline end to end, starting from
    raw Opera .table files and ending with per-pair multipole reports and
    model-selection recommendations.

    Parameters
    ----------
    raw_table_dir          : directory containing Results_DefField_*Ain-2.*
                              .table files
    out_dir                : where to write parsed_fields.csv and all
                              downstream tables/reports. Defaults to
                              raw_table_dir itself.
    n_sensors              : number of magnetometer positions to search for
    reference_pair         : which z-pair to run the initial sensor layout
                              search on (default 'P1', i.e. slices {1,2})
    max_candidates         : cap on layout combinations tried per search
                              (None = exhaustive; set this for n_sensors>=3
                              or large candidate grids to keep runtime sane)
    degradation_tolerance  : max allowed fractional RMSE increase before a
                              pair falls back to its own pair-specific layout
                              (default 0.5 = 50% worse than that pair's own
                              best achievable layout)
    n_max                  : multipole order used for the main report step
    n_max_range            : range of n_max values swept in model selection
    noise_sigma_T          : magnetometer noise floor in Tesla (~1e-9 per
                              the project's stated noise floor)
    ridge_alpha            : ridge regularisation for PCA regression steps
    run_orientation_search : whether to run the full grid search over sensor
                              orientations in the report (slower, more
                              thorough) vs just comparing named strategies
    verbose                : print progress messages for each stage

    Returns
    -------
    dict with keys:
        df                     : final combined, reclassified, pair-tagged dataframe
        pair_results           : output of z_pair_pipeline.run_all_pairs()
        symmetry_report_df     : within-pair coefficient consistency table
        layout_search          : output of search_and_validate_layout()
        report_results         : output of generate_multi_pair_report()
        model_selection_results: output of run_multi_pair_model_selection()
    """
    raw_table_dir = Path(raw_table_dir)
    out_dir = Path(out_dir) if out_dir is not None else raw_table_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: parse raw Opera .table files
    if verbose:
        print(f"\n{'='*100}\n  STAGE 1 — Parsing raw .table files from {raw_table_dir}\n{'='*100}")
    raw_df = parse_all_table_files(raw_table_dir, verbose=verbose)
    csv_path = out_dir / "parsed_fields.csv"
    raw_df.to_csv(csv_path, index=False)
    if verbose:
        mode_note = "vector (Hx,Hy,Hz)" if has_vector_data(raw_df) else "scalar (|H| only)"
        print(f"    Saved parsed data -> {csv_path}  [{mode_note}]")

    # Stage 2: reclassify + dedup
    if verbose:
        print(f"\n{'='*100}\n  STAGE 2 — Reclassifying geometry + de-duplicating\n{'='*100}")
    df = load_and_reclassify(csv_path)

    # Stage 3: tag z-pairs
    if verbose:
        print(f"\n{'='*100}\n  STAGE 3 — Tagging z-pairs {{1,2}}, {{3,4}}, {{5,6}}, {{7,8}}\n{'='*100}")
    df = add_pair_id(df)
    validate_pair_coverage(df)
    if verbose:
        counts = df.groupby("pair_id")["scenario_id"].nunique()
        print(counts.to_string())

    # Stage 4: per-pair PCA + multipole fit/CV, symmetry diagnostic
    if verbose:
        print(f"\n{'='*100}\n  STAGE 4 — Per-pair PCA reference + multipole fit/CV\n{'='*100}")
    pair_results = run_all_pairs(df, sensor_positions=None, n_max=n_max, verbose=verbose)
    symmetry_report_df = print_symmetry_report(pair_results)

    # Stage 5: sensor layout search + cross-pair generalization/fallback
    if verbose:
        print(f"\n{'='*100}\n  STAGE 5 — Sensor layout search on {reference_pair}, "
              f"validating generalization to other pairs\n{'='*100}")
    layout_search = search_and_validate_layout(
        df, n_sensors=n_sensors, reference_pair=reference_pair,
        max_candidates=max_candidates, ridge_alpha=ridge_alpha,
        noise_sigma_T=noise_sigma_T,
        degradation_tolerance=degradation_tolerance, verbose=verbose,
    )
    generalization_report_to_df(layout_search["generalization_report"]).to_csv(
        out_dir / "layout_generalization_report.csv", index=False)
    """
    if has_vector_data(df):
        if verbose:
            print(f"\n{'='*100}\n  STAGE 5b — Greedy-vs-joint bias check (vector-mode data available)\n{'='*100}")

        for pid, layout in layout_search["final_layouts"].items():
            pair_df = df[df["pair_id"] == pid]
            rep_scenario = pair_df[pair_df["region"] == "sensor_zone"].iloc[0]
            # pick one representative scenario's true Hx/Hy/Hz at the chosen sensor positions
            sub = pair_df[(pair_df["region"] == "sensor_zone")]
            # build Hx_true, Hy_true, Hz_true at layout positions for one scenario
            _, _, rel_diff, _ = compare_greedy_vs_joint(
                layout, Hx_true, Hy_true, Hz_true, n_max, r0=2.75, verbose=False)
            trust_greedy = rel_diff < 0.1 
            print(f"  Pair {pid}: greedy/joint relative bias = {rel_diff:.4f}  "
                  f"-> {'safe to use adaptive order selection' if trust_greedy else 'DO NOT use greedy on this layout'}")
    """
    # Stage 6: per-pair multipole report, cross-pair summary on top
    if verbose:
        print(f"\n{'='*100}\n  STAGE 6 — Multipole field reconstruction reports (per pair)\n{'='*100}")
    report_results = generate_multi_pair_report(
        df, pair_layouts=layout_search["final_layouts"], n_max=n_max,
        noise_sigma_T=noise_sigma_T,
        run_orientation_search=run_orientation_search,
        save_dir=out_dir / "reports",
    )
    

    # Stage 7: per-pair model selection, cross-pair summary on top
    if verbose:
        print(f"\n{'='*100}\n  STAGE 7 — Model selection (Phase 4/5, per pair)\n{'='*100}")
    model_selection_results = run_multi_pair_model_selection(
        df, pair_layouts=layout_search["final_layouts"], n_max_range=n_max_range,
        noise_sigma_T=noise_sigma_T, alpha=ridge_alpha,
        save_dir=out_dir / "model_selection",
    )

    per_pair_sweeps = {
        pid: result["sweep"]
        for pid, result in model_selection_results["per_pair"].items()
    }

    plot_rmse_vs_nmax_convergence(
        per_pair_sweeps,
        out_path="C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/rmse_convergence_all_pairs.pdf"
    )    

    if verbose:
        print(f"\n{'='*100}")
        print(f"  PIPELINE COMPLETE")
        print(f"  Outputs saved under: {out_dir}")
        print(f"{'='*100}\n")

    return {
        "df": df,
        "pair_results": pair_results,
        "symmetry_report_df": symmetry_report_df,
        "layout_search": layout_search,
        "report_results": report_results,
        "model_selection_results": model_selection_results,
    }

if __name__ == "__main__":

    run_full_pipeline(
        raw_table_dir = "C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method",
        out_dir = None,
        n_sensors = 6,
        # reference_pair =   ,
        max_candidates = 5000,
        # degradation_tolerance =   ,
        n_max = 3,                          # multipole mode
        # n_max_range =   ,
        # noise_sigma_T =   ,               # magnetometer noise floor
        # ridge_alpha=   ,
        # run_orientation_search=   ,       # defaults true but cartesian is selected
        # verbose=   ,                      # default true
        target_J = 66.039,
        target_slice = 1,                   # plot functionality
    )
