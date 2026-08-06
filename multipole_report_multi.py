"""
multipole_report_multi.py
==========================
Runs multipole_report.generate_full_report() once per z-pair, unmodified,
then reorders the output so a condensed cross-pair comparison table appears
first, followed by each pair's full detailed report.

Design note: generate_full_report() prints as it computes and has no verbose
flag, so rather than editing it, each pair's run is executed with stdout
captured (contextlib.redirect_stdout) into a string buffer.
"""

import io
import contextlib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from z_pair_utils import add_pair_id, validate_pair_coverage, PAIR_IDS, pair_slices
from multipole_report import generate_full_report
from multipole_fitter import DEFAULT_N_MAX
from multipole_basis import R0_DEFAULT


def _extract_summary_row(pid: str, pair_df: pd.DataFrame, payload: dict,
                         layout: list, n_max: int) -> dict:
    """
    Pull the handful of headline numbers out of a generate_full_report
    payload dict into one flat row for the cross-pair table.
    """
    pca_ref = payload.get("pca_reference") or {}
    fit_df  = payload["fit_df"]
    cv      = payload["cv_multipole"]
    uncert  = payload.get("uncertainty") or {}

    pca_rmse  = pca_ref.get("field_rmse_cv", np.nan)
    cv_rmse   = cv.get("mean_rmse", np.nan)
    ratio_pca = (cv_rmse / pca_rmse) if (pca_rmse and pca_rmse > 0
                                        and np.isfinite(pca_rmse)) else np.nan

    finite_cond = fit_df["condition_number"].replace([np.inf, -np.inf], np.nan).dropna()
    cond_median = float(finite_cond.median()) if len(finite_cond) else np.nan

    n_unresolved = len(uncert.get("unresolved", [])) if uncert else np.nan

    return {
        "pair_id"            : pid,
        "slices"             : "+".join(str(s) for s in pair_slices(pid)),
        "n_scenarios"        : pair_df["scenario_id"].nunique(),
        "sensor_layout"      : str(layout),
        "n_max"              : n_max,
        "pca_cv_rmse"        : pca_rmse,
        "multipole_cv_rmse"  : cv_rmse,
        "ratio_to_pca"       : ratio_pca,
        "tube_rmse_fit"      : float(fit_df["rmse_tube"].mean()),
        "cond_number_median" : cond_median,
        "n_coeffs_unresolved": n_unresolved,
    }


def print_cross_pair_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "═"*100)
    print("  CROSS-PAIR COMPARISON SUMMARY")
    print("  (quick-glance performance for each z-pair region — full detail follows below)")
    print("═"*100)

    if summary_df.empty:
        print("  No pairs were successfully processed.")
        return

    cols = ["pair_id", "slices", "n_scenarios", "n_max", "pca_cv_rmse",
            "multipole_cv_rmse", "ratio_to_pca", "cond_number_median",
            "n_coeffs_unresolved"]
    display_df = summary_df[cols].copy()

    def fmt_row(r):
        ratio_str = f"{r['ratio_to_pca']:.3f}x" if np.isfinite(r["ratio_to_pca"]) else "—"
        cond_str  = f"{r['cond_number_median']:.1f}" if np.isfinite(r["cond_number_median"]) else "inf"
        verdict   = ("  comparable" if np.isfinite(r["ratio_to_pca"]) and r["ratio_to_pca"] < 2
                     else ("X worse than PCA" if np.isfinite(r["ratio_to_pca"]) else ""))
        return (f"  {r['pair_id']:<5} {r['slices']:<7} {int(r['n_scenarios']):>6} "
                f"{int(r['n_max']):>6} {r['pca_cv_rmse']:>12.3e} {r['multipole_cv_rmse']:>16.3e} "
                f"{ratio_str:>10} {cond_str:>10} {int(r['n_coeffs_unresolved']) if np.isfinite(r['n_coeffs_unresolved']) else '—':>10}  {verdict}")

    print(f"\n  {'pair':<5} {'slices':<7} {'nscen':>6} {'n_max':>6} "
          f"{'PCA CV RMSE':>12} {'multipole RMSE':>16} {'÷PCA':>10} {'cond#':>10} {'unresolved':>10}")
    print("  " + "-"*100)
    for _, r in display_df.iterrows():
        print(fmt_row(r))

    print("\n  Sensor layouts used per pair:")
    for _, r in summary_df.iterrows():
        print(f"    {r['pair_id']}: {r['sensor_layout']}")

    unique_layouts = summary_df["sensor_layout"].nunique()
    if unique_layouts == 1 and len(summary_df) > 1:
        print(f"\n  * All {len(summary_df)} pairs share the same sensor layout (no generalization "
              f"fallback was triggered), so the orientation comparison/grid search (Steps 4-5 in "
              f"each pair's detail below) was computed ONCE and reused -- this is expected, not a bug.")

    n_worse = int((display_df["ratio_to_pca"] > 2).sum())
    if n_worse > 0:
        print(f"\n  X {n_worse}/{len(display_df)} pair(s) show multipole CV RMSE more than "
              f"2x the PCA benchmark — inspect that pair's full report below.")
    print("═"*100 + "\n")


def generate_multi_pair_report(df: pd.DataFrame,
                               pair_layouts: dict,
                               n_max: int = DEFAULT_N_MAX,
                               noise_sigma_T: float = 1e-9,
                               run_orientation_search: bool = True,
                               r0: float = R0_DEFAULT,
                               save_dir=None, target_J=None,
                               target_slice=None) -> dict:
    """
    Run generate_full_report() once per z-pair using that pair's sensor
    layout (from sensor_layout_generalization.search_and_validate_layout's
    'final_layouts'), print a cross-pair summary table first, then dump
    each pair's full detailed report below it.

    Parameters
    ----------
    df           : full dataset (all pairs); pair_id will be added if missing
    pair_layouts : dict {pair_id: [(x,y), ...]} — e.g. the 'final_layouts'
                   output of search_and_validate_layout(). Pairs not present
                   as keys are skipped with a warning.
    n_max, noise_sigma_T, run_orientation_search, r0, save_dir : passed
                   through to generate_full_report for every pair.

    Returns
    -------
    dict with:
        summary_df   : cross-pair comparison DataFrame
        per_pair     : {pair_id: full payload dict from generate_full_report}
    """
    if "pair_id" not in df.columns:
        df = add_pair_id(df)
    validate_pair_coverage(df)

    captured_text = {}
    payloads = {}
    summary_rows = []
    orientation_cache_by_signature = {}   # {(sensor_positions, n_max, r0, noise_sigma_T): cache_dict}

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

        signature = (tuple(layout), n_max, r0, noise_sigma_T)
        cached = orientation_cache_by_signature.get(signature)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = generate_full_report(
                pair_df, sensor_positions=layout, n_max=n_max,
                noise_sigma_T=noise_sigma_T,
                run_orientation_search=run_orientation_search,
                r0=r0, save_dir=pair_save_dir,
                orientation_cache=cached, pair_label=pid,
                target_J=target_J, target_slice=target_slice,
            )
        

        if cached is None:
            orientation_cache_by_signature[signature] = payload["orientation_cache"]

        captured_text[pid] = buf.getvalue()
        payloads[pid] = payload
        summary_rows.append(_extract_summary_row(pid, pair_df, payload, layout, n_max))

    summary_df = pd.DataFrame(summary_rows)

    # Print summary FIRST, then full per-pair detail
    print_cross_pair_summary(summary_df)

    for pid in PAIR_IDS:
        if pid not in captured_text:
            continue
        print(f"\n{'#'*100}")
        print(f"  FULL DETAIL — PAIR {pid}  (slices {pair_slices(pid)})")
        print(f"{'#'*100}")
        print(captured_text[pid])

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(Path(save_dir) / "cross_pair_summary.csv", index=False)
        print(f"   Cross-pair summary saved to {Path(save_dir)/'cross_pair_summary.csv'}")

    return {"summary_df": summary_df, "per_pair": payloads}
