"""
z_pair_pipeline.py
===================
Runs the existing PCA + multipole pipeline independently for each z-pair
({1,2}, {3,4}, {5,6}, {7,8}) instead of pooling all 8 slices into one global
model. None of the underlying functions (build_tube_matrix, fit_pca,
fit_all_scenarios, cross_validate_multipole, ...) change as they
already key off scenario_id and operate correctly on whatever subset of df
they're handed. This module supplies the right subset, per pair, and
collects results.

Also includes a within-pair consistency check: since the two slices in a
pair are not exact mirror images but comparable regions, this is framed
as a validation/augmentation diagnostic rather than an assumption enforced
during fitting.
"""

import warnings
import numpy as np
import pandas as pd

from z_pair_utils import add_pair_id, validate_pair_coverage, PAIR_IDS, pair_slices

from magis_pca_pipeline import build_tube_matrix, fit_pca, build_sensor_matrix, fit_regression, evaluate
from multipole_fitter import fit_all_scenarios, cross_validate_multipole, DEFAULT_N_MAX
from multipole_basis import R0_DEFAULT, coefficient_names
from coefficient_smoothing import smooth_coefficients_vs_J


def run_pca_reference_for_pair(pair_df: pd.DataFrame) -> dict:
    """Same as multipole_report.run_pca_reference, but explicit about scope
    (kept local here to avoid a circular import back into multipole_report)."""
    T, scenarios, _ = build_tube_matrix(pair_df)
    pca, K, A, _ = fit_pca(T, variance_threshold=0.9999)
    S, _ = build_sensor_matrix(pair_df, scenarios)
    model, scaler, cv_preds = fit_regression(S, A)
    results = evaluate(pca, T, A, cv_preds, S, model, scaler,
                       noise_levels=(0.0,), n_noise_trials=1)
    return {"K": K, "field_rmse_cv": results["field_rmse_cv"], "n_scenarios": len(scenarios)}


def run_all_pairs(df: pd.DataFrame,
                  sensor_positions: list | None = None,
                  n_max: int = DEFAULT_N_MAX,
                  r0: float = R0_DEFAULT,
                  verbose: bool = True) -> dict:
    """
    Run PCA reference + multipole fit + multipole CV independently for each
    z-pair.

    Parameters
    ----------
    df               : full dataset from load_and_reclassify (all slices,
                        already J-value-finalized upstream by the Opera
                        automation — no J resampling happens here)
    sensor_positions  : sensor (x,y) layout to use for the multipole fit.
                        If None, fit_all_scenarios falls back to using every
                        sensor_zone grid point (its existing default).
    n_max            : multipole truncation order
    r0               : normalisation radius (tube inner radius by default)

    Returns
    -------
    dict keyed by pair_id, each value a dict with:
        pca_ref   : PCA benchmark for that pair
        fit_df    : per-scenario multipole coefficients (that pair only)
        cv        : multipole CV RMSE dict for that pair
        n_slices_present : how many of the pair's 2 slices had data
    """
    df = add_pair_id(df)
    validate_pair_coverage(df)

    results = {}
    for pid in PAIR_IDS:
        sub = df[df["pair_id"] == pid]
        if sub.empty:
            if verbose:
                print(f"\nPair {pid}: no data present, skipping.")
            continue

        slices_present = sorted(sub["slice_id"].unique())
        if verbose:
            print(f"\n{'='*80}")
            print(f"  PAIR {pid}  (slices {pair_slices(pid)}, present: {slices_present})")
            print(f"{'='*80}")

        try:
            pca_ref = run_pca_reference_for_pair(sub)
            if verbose:
                print(f"  PCA reference: K={pca_ref['K']}  CV RMSE={pca_ref['field_rmse_cv']:.3e} A/m")
        except Exception as e:
            warnings.warn(f"Pair {pid}: PCA reference failed: {e}")
            pca_ref = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_df = fit_all_scenarios(sub, sensor_xy=sensor_positions, n_max=n_max, r0=r0, mode="auto")
            cv = cross_validate_multipole(sub, sensor_xy=sensor_positions, n_max=n_max, r0=r0, mode="auto")
            # smooth coefficients across J sweep per slice
            coeff_nms = coefficient_names(n_max)
            smoothed_by_slice = {}
            for sid_num in sorted(fit_df["slice_id"].unique()):
                smoothed_df, diagnostics = smooth_coefficients_vs_J(
                    fit_df, n_max, slice_id=sid_num, method="linear_through_origin",
                    coeff_names_override=coeff_nms)
                mean_r2 = np.mean(list(diagnostics.values()))
                print(f"    Pair {pid}, slice {sid_num} - coefficient smoothing self-consistency "
                      f"R²={mean_r2:.4f}"
                      + ("  X low - linear_through_origin may not fit, try method='polynomial'"
                         if mean_r2 < 0.9 else ""))
                smoothed_by_slice[sid_num] = smoothed_df

            # merge the per-slice smoothed frames back into one fit_df 
            fit_df = pd.concat(smoothed_by_slice.values(), ignore_index=True).sort_values("scenario_id")

        if verbose:
            print(f"  Multipole fit: {len(fit_df)} scenarios, "
                  f"tube RMSE={fit_df['rmse_tube'].mean():.6f} A/m")
            print(f"  Multipole CV : {cv['mean_rmse']:.6f} ± {cv['std_rmse']:.6f} A/m")

        results[pid] = {
            "pca_ref": pca_ref,
            "fit_df": fit_df,
            "cv": cv,
            "n_slices_present": len(slices_present),
            "slices_present": slices_present,
        }

    return results


def pair_symmetry_check(results: dict, pid: str) -> dict | None:
    """
    Compare the fitted multipole coefficients between the two slices within
    a pair, matched by coil_density (J), as a consistency/validation check,
    not used to constrain or pool the fit itself, since the two slices are
    not exact mirror images.

    Returns
    -------
    dict of {coefficient_name: mean_abs_difference} across matched J values,
    or None if the pair doesn't have exactly 2 slices present.
    """
    if pid not in results:
        warnings.warn(f"Pair {pid} not found in results.")
        return None

    fit_df = results[pid]["fit_df"]
    by_slice = {sid: g for sid, g in fit_df.groupby("slice_id")}
    slices = sorted(by_slice)

    if len(slices) != 2:
        warnings.warn(
            f"Pair {pid} has {len(slices)} slice(s) present ({slices}), "
            f"need exactly 2 for a symmetry check. Skipping."
        )
        return None

    a, b = by_slice[slices[0]], by_slice[slices[1]]
    merged = a.merge(b, on="coil_density", suffixes=("_s1", "_s2"))
    if merged.empty:
        warnings.warn(f"Pair {pid}: no matching coil_density values between slices "
                      f"{slices[0]} and {slices[1]} — cannot compare.")
        return None
    if len(merged) < len(a):
        warnings.warn(
            f"Pair {pid}: only {len(merged)}/{len(a)} J values matched between "
            f"slices {slices[0]} and {slices[1]}. Check for missing scenarios."
        )

    exclude = {"scenario_id", "coil_density", "slice_id", "z",
               "rmse_sensor", "rmse_tube", "condition_number", "n_iterations_used"}
    coeff_cols = [c for c in a.columns if c not in exclude]

    resid = {}
    for c in coeff_cols:
        c1, c2 = f"{c}_s1", f"{c}_s2"
        if c1 in merged.columns and c2 in merged.columns:
            resid[c] = float(np.mean(np.abs(merged[c1] - merged[c2])))

    return resid


def print_symmetry_report(results: dict) -> pd.DataFrame:
    # Run pair_symmetry_check for all pairs and print a combined table.
    rows = []
    for pid in PAIR_IDS:
        if pid not in results:
            continue
        resid = pair_symmetry_check(results, pid)
        if resid is None:
            continue
        row = {"pair_id": pid, **resid}
        rows.append(row)

    if not rows:
        print("No pairs had exactly 2 slices present — no symmetry report available.")
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("pair_id")
    print("\n" + "="*80)
    print("  Within-pair coefficient consistency check (mean |Δ| across matched J)")
    print("  (diagnostic only — pairs are not assumed exactly mirror-symmetric)")
    print("="*80)
    print(out.to_string(float_format=lambda v: f"{v:.4e}"))
    return out


if __name__ == "__main__":
    import sys
    from magis_pca_pipeline import load_and_reclassify

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "parsed_fields.csv"
    print(f"Loading: {csv_path}")
    df = load_and_reclassify(csv_path)

    results = run_all_pairs(df)
    print_symmetry_report(results)
