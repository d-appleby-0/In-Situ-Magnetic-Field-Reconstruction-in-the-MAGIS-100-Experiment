"""
multipole_model_selection_multi.py
====================================
Runs multipole_model_selection.run_model_selection() once per z-pair,
unmodified, then reorders output so a cross-pair comparison of recommended
n_max / oversampling / RMSE appears first, followed by each pair's full detail. Same capture-and-reorder approach as
"""

import io
import contextlib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from z_pair_utils import add_pair_id, validate_pair_coverage, PAIR_IDS, pair_slices
from multipole_model_selection import run_model_selection
from multipole_fitter import DEFAULT_ALPHA
from multipole_basis import R0_DEFAULT


def _extract_summary_row(pid: str, pair_df: pd.DataFrame, result: dict,
                         layout: list) -> dict:
    sweep = result["sweep"]
    sweep_df = sweep["sweep_df"]
    rec_n_max = result["recommended_n_max"]
    rec_row = sweep_df.loc[sweep_df["n_max"] == rec_n_max]
    rec_row = rec_row.iloc[0] if not rec_row.empty else None

    return {
        "pair_id"           : pid,
        "slices"            : "+".join(str(s) for s in pair_slices(pid)),
        "n_scenarios"       : pair_df["scenario_id"].nunique(),
        "sensor_layout"     : str(layout),
        "recommended_n_max" : rec_n_max,
        "n_coeffs_at_rec"   : (2*rec_n_max + 2),
        "oversampling_at_rec": float(rec_row["oversampling"]) if rec_row is not None else np.nan,
        "cv_rmse_at_rec"    : float(rec_row["cv_rmse"]) if rec_row is not None else np.nan,
        "ratio_to_pca_at_rec": float(rec_row["ratio_to_pca"]) if rec_row is not None else np.nan,
        "pca_rmse"          : sweep.get("pca_rmse", np.nan),
        "plateau_detected"  : sweep.get("plateau_detected", False),
        "mode_a_bias"       : sweep.get("mode_a_bias", np.nan),
    }


def print_cross_pair_model_selection_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "═"*100)
    print("  CROSS-PAIR MODEL SELECTION SUMMARY")
    print("  (recommended n_max and convergence status for each z-pair region)")
    print("═"*100)

    if summary_df.empty:
        print("  No pairs were successfully processed.")
        return

    print(f"\n  {'pair':<5} {'slices':<7} {'nscen':>6} {'rec n_max':>10} "
          f"{'n_coeffs':>9} {'oversamp':>9} {'CV RMSE':>12} {'÷PCA':>8} {'plateau?':>9}")
    print("  " + "-"*100)
    for _, r in summary_df.iterrows():
        ratio_str = f"{r['ratio_to_pca_at_rec']:.3f}x" if np.isfinite(r["ratio_to_pca_at_rec"]) else "—"
        plateau_str = "yes X" if r["plateau_detected"] else "no"
        print(f"  {r['pair_id']:<5} {r['slices']:<7} {int(r['n_scenarios']):>6} "
              f"{int(r['recommended_n_max']):>10} {int(r['n_coeffs_at_rec']):>9} "
              f"{r['oversampling_at_rec']:>9.2f} {r['cv_rmse_at_rec']:>12.6f} "
              f"{ratio_str:>8} {plateau_str:>9}")

    n_max_values = summary_df["recommended_n_max"].dropna().unique()
    if len(n_max_values) > 1:
        print(f"\n  X Recommended n_max is NOT consistent across pairs: {sorted(n_max_values)}.")
        print(f"    Consider using the max recommended value across all pairs for a single")
        print(f"    shared n_max, or keep per-pair n_max if pair-specific fits are acceptable.")
    else:
        print(f"\n    All pairs agree on recommended n_max = {int(n_max_values[0])}.")

    n_plateau = int(summary_df["plateau_detected"].sum())
    if n_plateau > 0:
        print(f"  X {n_plateau}/{len(summary_df)} pair(s) show a plateau above PCA "
              f"(Mode A linearisation bias) — see per-pair detail below.")
    print("═"*100 + "\n")


def run_multi_pair_model_selection(df: pd.DataFrame,
                                   pair_layouts: dict,
                                   n_max_range: range | list = range(1, 6),
                                   noise_sigma_T: float = 1e-9,
                                   r0: float = R0_DEFAULT,
                                   alpha: float = DEFAULT_ALPHA,
                                   n_folds: int = 10,
                                   save_dir=None) -> dict:
    """
    Run run_model_selection() once per z-pair, print a cross-pair summary
    of recommended n_max first, then dump each pair's full detail.

    Parameters
    ----------
    df           : full dataset (all pairs); pair_id added if missing
    pair_layouts : dict {pair_id: [(x,y), ...]} — sensor layout per pair
                   (typically the 'final_layouts' from
                   sensor_layout_generalization.search_and_validate_layout)

    Returns
    -------
    dict with:
        summary_df : cross-pair comparison DataFrame
        per_pair   : {pair_id: full result dict from run_model_selection}
    """
    if "pair_id" not in df.columns:
        df = add_pair_id(df)
    validate_pair_coverage(df)

    captured_text = {}
    results = {}
    summary_rows = []

    for pid in PAIR_IDS:
        pair_df = df[df["pair_id"] == pid]
        if pair_df.empty:
            continue
        if pid not in pair_layouts:
            warnings.warn(f"No sensor layout supplied for pair {pid} in pair_layouts — skipping.")
            continue

        layout = pair_layouts[pid]
        pair_save_dir = None
        if save_dir is not None:
            pair_save_dir = Path(save_dir) / pid
            pair_save_dir.mkdir(parents=True, exist_ok=True)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_model_selection(
                df=pair_df, sensor_positions=layout, n_max_range=n_max_range,
                pca_rmse=None, noise_sigma_T=noise_sigma_T, r0=r0,
                alpha=alpha, n_folds=n_folds, save_dir=pair_save_dir,
            )

        captured_text[pid] = buf.getvalue()
        results[pid] = result
        summary_rows.append(_extract_summary_row(pid, pair_df, result, layout))

    summary_df = pd.DataFrame(summary_rows)

    print_cross_pair_model_selection_summary(summary_df)

    for pid in PAIR_IDS:
        if pid not in captured_text:
            continue
        print(f"\n{'#'*100}")
        print(f"  FULL DETAIL — PAIR {pid}  (slices {pair_slices(pid)})")
        print(f"{'#'*100}")
        print(captured_text[pid])

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(Path(save_dir) / "cross_pair_model_selection_summary.csv", index=False)
        print(f"   Cross-pair model selection summary saved to "
              f"{Path(save_dir)/'cross_pair_model_selection_summary.csv'}")

    return {"summary_df": summary_df, "per_pair": results}
