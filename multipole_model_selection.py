"""
multipole_model_selection.py
=============================
Phase 4 — Sensor count vs multipole order guidance table
Phase 5 — PCA cross-validation sweep: does multipole RMSE approach PCA RMSE
           as n_max increases, or does it plateau (indicating a Mode A
           linearisation limitation rather than order truncation)?

Phase 4: Sensor/Order Decision Table
--------------------------------------
With K 3-axis magnetometers you have 3K scalar measurements.
To reliably resolve multipole order n_max you need 2*n_max+2 coefficients.
The table prints the maximum resolvable order and recommended n_max for
sensor counts 1-6, mirroring g-2 Table I but adapted for 3-axis sensors:

  n_sensors | n_meas | n_coeffs_max | recommended_n_max | notes
  -----------------------------------------------------------------
      1      |   3    |      2       |        0          | underdetermined for n=1
      2      |   6    |      4       |        1          | tight; n=2 underdetermined
      3      |   9    |      6       |        2          | good headroom
      4      |  12    |      8       |        3          | matches your n_max=3 default
      6      |  18    |     12       |        5          | approaches g-2 level
      ...

The "recommended" n_max is the largest n where n_meas >= 2*(2*n_max+2),
i.e. at least 2x oversampling for stable conditioning, following the
g-2 guideline of "only use moments the probe geometry can resolve".

Phase 5: RMSE vs n_max sweep
-------------------------------
For a fixed sensor layout (from the PCA layout search) and each n_max in
[1..n_max_max]:
  1. Fit multipole coefficients for all scenarios.
  2. Compute cross-validated tube reconstruction RMSE.
  3. Compare against PCA CV RMSE (the Bayes-optimal benchmark
     given perfect sensor readings).

Interpretation:
  - If multipole RMSE decreases as n_max increases and converges toward
    the PCA RMSE: the main limitation is order truncation. Increase n_max
    (or equivalently, add sensors so the higher orders are resolvable).
  - If multipole RMSE plateaus ABOVE the PCA RMSE even as n_max grows:
    the linearisation approximation (Mode A — fitting |H| magnitude rather
    than vector components) is the bottleneck. The residual error cannot be
    removed by adding more multipole orders; it requires Mode B (vector H
    export from Opera) or at minimum a smarter initialisation strategy.
  - The "elbow" in the RMSE vs n_max curve is the practical optimal order:
    the point where additional orders give diminishing returns relative to
    the conditioning cost of more coefficients with the same sensor count.

Both phases are fully self-contained: they import from the existing pipeline
and produce tables + a visualisation-ready dict so the caller can plot.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from multipole_basis import (
    coefficient_names, build_design_matrix, R0_DEFAULT,
    check_design_matrix_rank,
)
from multipole_fitter import (
    fit_all_scenarios, cross_validate_multipole,
    DEFAULT_ALPHA, DEFAULT_N_ITER, DEFAULT_TOL, MU0,
)
from sensor_orientation import (
    compare_named_orientations, condition_number,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Sensor count vs multipole order decision table
# ═══════════════════════════════════════════════════════════════════════════════

def sensor_order_decision_table(
    n_sensors_range: range | list = range(1, 7),
    axes_per_sensor: int = 3,
    oversampling_factor: float = 2.0,
    print_table: bool = True,
) -> pd.DataFrame:
    """
    Print and return the sensor-count vs multipole-order decision table.

    For each sensor count, computes:
      - n_measurements  = n_sensors * axes_per_sensor
      - n_coeffs_max    = n_measurements  (largest n_coeffs solvable at all)
      - n_max_raw       = (n_coeffs_max - 2) / 2  (solve for n from 2n+2=C)
      - recommended_n_max = largest n where n_meas >= oversampling_factor
                            * (2*n_max+2)  (stable conditioning, not just
                            formally solvable). Default oversampling = 2x,
                            matching g-2's implicit criterion in Table I.
      - condition_impact: qualitative note about conditioning at recommended
                          n_max, derived from the oversampling ratio.

    Parameters
    ----------
    n_sensors_range     : sensor counts to tabulate
    axes_per_sensor     : 3 for your 3-axis magnetometers (can be set to 1
                          to compare against g-2's scalar NMR probes)
    oversampling_factor : minimum n_meas / n_coeffs ratio to declare an order
                          "recommended" (default 2.0)

    Returns
    -------
    DataFrame with one row per sensor count.
    """
    rows = []
    for n_s in n_sensors_range:
        n_meas = n_s * axes_per_sensor

        # Largest n_max that is at all solvable (n_meas >= 2*n_max+2)
        n_max_solvable = max(0, (n_meas - 2) // 2)

        # Largest n_max with stable conditioning (n_meas >= factor*(2*n_max+2))
        n_max_rec = 0
        for n in range(n_max_solvable, -1, -1):
            n_coeffs = 2*n + 2
            if n_meas >= oversampling_factor * n_coeffs:
                n_max_rec = n
                break

        # Effective oversampling at recommended n_max
        n_coeffs_rec = 2*n_max_rec + 2
        ratio_rec = n_meas / n_coeffs_rec if n_coeffs_rec > 0 else float('inf')

        if ratio_rec >= 3:
            cond_note = "well-conditioned"
        elif ratio_rec >= 2:
            cond_note = "acceptable"
        elif ratio_rec >= 1:
            cond_note = "marginal — regularise"
        else:
            cond_note = "underdetermined"

        # g-2 equivalent: how many scalar NMR probes needed for same resolution
        g2_equiv = n_coeffs_rec  # g-2 used one scalar probe per measurement

        rows.append({
            "n_sensors"       : n_s,
            "n_meas_total"    : n_meas,
            "n_max_solvable"  : n_max_solvable,
            "n_max_recommended": n_max_rec,
            "n_coefficients"  : n_coeffs_rec,
            "oversampling"    : round(ratio_rec, 2),
            "conditioning"    : cond_note,
            "g2_NMR_equiv"    : g2_equiv,
        })

    df = pd.DataFrame(rows)

    if print_table:
        print("\n" + "═"*80)
        print("  PHASE 4 — Sensor Count vs Multipole Order Decision Table")
        print(f"  (3-axis magnetometers, oversampling threshold = {oversampling_factor:.1f}x)")
        print("═"*80)
        print(f"\n  {'n_sensors':>9} {'n_meas':>7} {'n_max_solv':>11} "
              f"{'n_max_rec':>10} {'n_coeffs':>9} {'oversamp':>9} "
              f"{'conditioning':<18} {'g2_NMR_equiv':>13}")
        print("  " + "-"*90)
        for _, r in df.iterrows():
            flag = "  ◄ your current default" if r["n_max_recommended"] == 3 else ""
            print(f"  {int(r['n_sensors']):>9} {int(r['n_meas_total']):>7} "
                  f"{int(r['n_max_solvable']):>11} {int(r['n_max_recommended']):>10} "
                  f"{int(r['n_coefficients']):>9} {r['oversampling']:>9.2f} "
                  f"{r['conditioning']:<18} {int(r['g2_NMR_equiv']):>13}{flag}")

        print("\n  Notes:")
        print("  • n_max_solvable  : largest order with at least 1 measurement per coefficient")
        print("  • n_max_recommended: largest order with ≥{:.0f}x oversampling (stable fitting)".format(
              oversampling_factor))
        print("  • g2_NMR_equiv    : scalar NMR probes needed for same reconstruction order")
        print("    (3-axis sensors give 3x more info per physical location than scalar NMR)")
        print("  • With 2 sensors (your target) n_max_rec=1; with 3 sensors n_max_rec=2.")
        print("    Your current n_max=3 default is only appropriate with ≥4 sensors OR")
        print("    with strong regularisation (ridge α) accepted as a prior on small")
        print("    higher-order coefficients.\n")

    return df


def sensor_order_table_with_geometry(
    sensor_positions: list[tuple],
    n_max_range: range | list = range(1, 6),
    noise_sigma_T: float = 1e-9,
    r0: float = R0_DEFAULT,
    print_table: bool = True,
) -> pd.DataFrame:
    """
    Extend the decision table with ACTUAL condition numbers and D-optimality
    values for a SPECIFIC sensor layout, across a range of n_max values.
    This connects the abstract sensor-count analysis (above) to the concrete
    geometry of your best layout from the PCA search.

    For each n_max:
      - Builds the design matrix for the best physical orientation
        (cartesian, i.e. aligned to x/y/z, since it's the most neutral
        baseline; compare_named_orientations will show if another is better)
      - Reports condition number, D-optimality, and the number of singular
        values above the numerical threshold (effective DOF)
    """
    from sensor_orientation import compare_named_orientations
    sigma_H = noise_sigma_T / MU0
    rows = []
    for n_max in n_max_range:
        n_meas   = len(sensor_positions) * 3
        n_coeffs = 2*n_max + 2
        ratio    = n_meas / n_coeffs

        # Use cartesian (lab-aligned) as the neutral reference orientation
        orients  = [None for _ in sensor_positions]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = build_design_matrix(sensor_positions, orients, n_max, r0=r0)

        sv       = np.linalg.svd(A, compute_uv=False)
        cond     = float(sv[0]/sv[-1]) if sv[-1] > 1e-15 else float('inf')
        dof      = int(np.sum(sv > 1e-10 * sv[0]))
        F        = A.T @ A / sigma_H**2
        sign, ld = np.linalg.slogdet(F)
        d_opt    = float(ld) if sign > 0 else float('-inf')

        rows.append({
            "n_max"          : n_max,
            "n_coefficients" : n_coeffs,
            "n_measurements" : n_meas,
            "oversampling"   : round(ratio, 2),
            "effective_dof"  : dof,
            "condition_number": round(cond, 1) if np.isfinite(cond) else float('inf'),
            "log_D_optimality": round(d_opt, 3),
            "status"         : ("ok" if ratio >= 2 else
                               "marginal" if ratio >= 1 else "underdetermined"),
        })

    df = pd.DataFrame(rows)
    if print_table:
        print(f"\n  Geometry-specific table for positions {sensor_positions}:")
        print(f"  (cartesian orientation, r0={r0} in, σ_sensor={noise_sigma_T:.1e} T)\n")
        print(f"  {'n_max':>6} {'n_coeffs':>9} {'n_meas':>7} {'oversamp':>9} "
              f"{'eff_dof':>8} {'cond_num':>12} {'log_D':>10} {'status':<14}")
        print("  " + "-"*80)
        for _, r in df.iterrows():
            flag = " ←" if r["status"] == "ok" else ("  !" if r["status"] == "marginal" else "  ✗")
            print(f"  {int(r['n_max']):>6} {int(r['n_coefficients']):>9} "
                  f"{int(r['n_measurements']):>7} {r['oversampling']:>9.2f} "
                  f"{int(r['effective_dof']):>8} {r['condition_number']:>12.1f} "
                  f"{r['log_D_optimality']:>10.3f} {r['status']:<14}{flag}")
        print()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — RMSE vs n_max sweep: multipole convergence to PCA benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def rmse_vs_nmax_sweep(
    df: pd.DataFrame,
    sensor_positions: list[tuple],
    n_max_range: range | list = range(1, 6),
    pca_rmse: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    r0: float = R0_DEFAULT,
    n_folds: int = 10,
    n_iter: int = DEFAULT_N_ITER,
    tol: float = DEFAULT_TOL,
    print_results: bool = True,
    noise_sigma_T: float = 1e-9,
) -> dict:
    """
    Sweep over multipole orders and measure cross-validated tube reconstruction
    RMSE at each order, comparing against the PCA pipeline benchmark.

    This is the key diagnostic for Phase 5:
      - Converging toward PCA RMSE  → order truncation is the limit
      - Plateau above PCA RMSE      → Mode A linearisation is the limit

    Parameters
    ----------
    df              : full dataset from load_and_reclassify
    sensor_positions: list of (x,y) tuples — the layout under evaluation
    n_max_range     : multipole orders to sweep (default 1..5)
    pca_rmse        : PCA CV RMSE for comparison. If None, computed here.
    alpha           : ridge regularisation
    r0              : normalisation radius
    n_folds         : CV folds
    print_results   : print the sweep table and interpretation

    Returns
    -------
    dict with keys:
      sweep_df      : DataFrame with per-n_max results
      pca_rmse      : float, the PCA benchmark used
      elbow_n_max   : int, the recommended n_max (elbow of RMSE curve)
      plateau_detected : bool, True if RMSE plateaued above PCA
      plateau_floor : float, the plateau RMSE level if detected
      mode_a_bias   : float, estimated bias from linearisation (plateau - PCA)
    """

    # ── PCA benchmark ─────────────────────────────────────────────────────────
    if pca_rmse is None:
        try:
            from magis_pca_pipeline import (
                build_tube_matrix, fit_pca, build_sensor_matrix,
                fit_regression, evaluate,
            )
            T, scenarios, _ = build_tube_matrix(df)
            pca_obj, K, A_pca, _ = fit_pca(T, variance_threshold=0.99)
            S, _ = build_sensor_matrix(df, scenarios)
            model, scaler, cv_preds = fit_regression(S, A_pca)
            res = evaluate(pca_obj, T, A_pca, cv_preds, S, model, scaler,
                           noise_levels=(0.0,), n_noise_trials=1)
            pca_rmse = float(res["field_rmse_cv"])
            if print_results:
                print(f"  PCA benchmark (K={K}): CV RMSE = {pca_rmse:.6f} A/m")
        except Exception as e:
            if print_results:
                print(f"  PCA benchmark unavailable: {e}; proceeding without.")
            pca_rmse = None

    # ── Per-order sweep ───────────────────────────────────────────────────────
    records = []
    n_meas = len(sensor_positions) * 3

    for n_max in n_max_range:
        n_coeffs = 2 * n_max + 2
        ratio    = n_meas / n_coeffs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate_multipole(
                df, sensor_xy=sensor_positions, n_max=n_max,
                alpha=alpha, r0=r0, n_folds=n_folds,
            )

            # Also get the in-sample condition number and fit RMSE
            try:
                fit_df = fit_all_scenarios(
                    df, sensor_xy=sensor_positions, n_max=n_max,
                    alpha=alpha, r0=r0, n_iter=n_iter, tol=tol,
                )
                fit_rmse_sensor = float(fit_df["rmse_sensor"].mean())
                fit_rmse_tube   = float(fit_df["rmse_tube"].mean())
                cond_median     = float(
                    fit_df["condition_number"].replace([np.inf,-np.inf], np.nan)
                    .dropna().median()
                )
            except Exception:
                fit_rmse_sensor = np.nan
                fit_rmse_tube   = np.nan
                cond_median     = np.nan

        pca_ratio = (cv["mean_rmse"] / pca_rmse
                    if pca_rmse and pca_rmse > 0 else np.nan)

        records.append({
            "n_max"              : n_max,
            "n_coefficients"     : n_coeffs,
            "oversampling"       : round(ratio, 2),
            "cv_rmse"            : cv["mean_rmse"],
            "cv_rmse_std"        : cv["std_rmse"],
            "fit_rmse_sensor"    : fit_rmse_sensor,
            "fit_rmse_tube"      : fit_rmse_tube,
            "condition_number"   : cond_median,
            "ratio_to_pca"       : pca_ratio,
        })

    sweep_df = pd.DataFrame(records)

    # ── Convergence analysis ──────────────────────────────────────────────────
    cv_rmses = sweep_df["cv_rmse"].values

    # First check whether RMSE is monotonically increasing (diverging) —
    # this means the system is underdetermined and higher orders are hurting
    # by introducing unconstrained degrees of freedom that overfit the noise.
    # In this case the "elbow" is simply the minimum-RMSE order.
    diverging = bool(len(cv_rmses) >= 2 and cv_rmses[-1] > cv_rmses[0] * 1.05)

    if diverging:
        best_idx    = int(np.argmin(cv_rmses))
        elbow_n_max = list(n_max_range)[best_idx]
    elif len(cv_rmses) >= 2:
        improvements   = cv_rmses[:-1] - cv_rmses[1:]  # positive = better
        total_improve  = max(cv_rmses[0] - cv_rmses.min(), 1e-15)
        cumulative     = np.cumsum(np.clip(improvements, 0, None))
        elbow_idx      = int(np.searchsorted(cumulative, 0.9 * total_improve))
        elbow_idx      = min(elbow_idx, len(n_max_range)-1)
        elbow_n_max    = list(n_max_range)[elbow_idx]
    else:
        elbow_n_max = list(n_max_range)[0]

    # Plateau detection: check whether the last 3 (or available) RMSE values
    # are within 5% of each other AND above PCA RMSE by more than 10%.
    plateau_detected = False
    plateau_floor    = float(cv_rmses.min())
    mode_a_bias      = 0.0
    window = min(3, len(cv_rmses))
    if window >= 2:
        tail_rmse = cv_rmses[-window:]
        tail_var  = (tail_rmse.max() - tail_rmse.min()) / (tail_rmse.mean() + 1e-15)
        plateau_detected = bool(tail_var < 0.05)  # last N values within 5% of each other
        plateau_floor = float(tail_rmse.mean())
        if pca_rmse and pca_rmse > 0:
            mode_a_bias = max(0., plateau_floor - pca_rmse)
            if mode_a_bias / pca_rmse < 0.1:
                plateau_detected = False  # plateau is essentially at PCA level — converged

    # ── Print table ───────────────────────────────────────────────────────────
    if print_results:
        print(f"\n{'═'*80}")
        print(f"  PHASE 5 — Multipole RMSE vs Order Sweep")
        print(f"  Sensor positions: {sensor_positions}")
        print(f"  n_measurements = {n_meas}  (3 axes × {len(sensor_positions)} sensors)")
        if pca_rmse:
            print(f"  PCA benchmark RMSE = {pca_rmse:.6f} A/m  (lower bound)")
        print(f"{'═'*80}\n")

        hdr = (f"  {'n_max':>6} {'n_coeffs':>9} {'oversamp':>9} "
               f"{'CV RMSE':>12} {'±std':>10} "
               f"{'sensor_RMSE':>12} {'cond':>10} {'÷PCA':>8}")
        print(hdr)
        print("  " + "-"*80)

        for _, r in sweep_df.iterrows():
            nm     = int(r["n_max"])
            ratio_s = f"  {'✓' if r['oversampling']>=2 else ('~' if r['oversampling']>=1 else '✗')}"
            pca_col = (f"{r['ratio_to_pca']:>7.3f}x" if np.isfinite(r['ratio_to_pca'])
                       else "     —")
            cond_s  = (f"{r['condition_number']:>10.1f}" if np.isfinite(r['condition_number'])
                       else "       inf")
            elbow_flag = " ← elbow" if nm == elbow_n_max else ""
            print(f"  {nm:>6} {int(r['n_coefficients']):>9} "
                  f"{r['oversampling']:>8.2f}{ratio_s}"
                  f" {r['cv_rmse']:>12.6f} {r['cv_rmse_std']:>10.6f} "
                  f"{r['fit_rmse_sensor']:>12.6f} {cond_s} {pca_col}{elbow_flag}")

        print(f"\n  Legend: ✓ oversampling≥2 (recommended)  ~ marginal  ✗ underdetermined")
        diverge_note = "minimum RMSE order (RMSE diverges above this)" if diverging else "diminishing returns above this order"
        print(f"  Elbow  : n_max={elbow_n_max} — {diverge_note}")

        print(f"\n  {'─'*70}")
        print(f"  Convergence interpretation:")
        if diverging:
            print(f"  ⚠  DIVERGING: RMSE increases with n_max beyond n_max={elbow_n_max}.")
            print(f"     The system becomes underdetermined at higher orders ({n_meas} measurements")
            print(f"     vs {2*list(n_max_range)[-1]+2} coefficients at n_max={list(n_max_range)[-1]}).")
            print(f"     Unconstrained modes absorb noise rather than signal.")
            print(f"     → Use n_max={elbow_n_max} ({2*elbow_n_max+2} coefficients, "
                  f"oversampling={n_meas/(2*elbow_n_max+2):.2f}x)")
            print(f"     → OR add a 3rd sensor (9 meas) to stably resolve n_max=2")
            if pca_rmse:
                best = float(cv_rmses[int(np.argmin(cv_rmses))])
                frac = (best - pca_rmse) / pca_rmse * 100
                print(f"     Best achievable RMSE with current layout: {best:.6f} A/m")
                print(f"     ({frac:.1f}% above PCA — from underdetermination + Mode A bias)")
        elif plateau_detected and pca_rmse:
            print(f"  ⚠  PLATEAU DETECTED: RMSE stopped improving at ~{plateau_floor:.6f} A/m,")
            print(f"     which is {mode_a_bias:.6f} A/m above the PCA benchmark ({pca_rmse:.6f} A/m).")
            print(f"     Mode A linearisation bias ≈ {mode_a_bias/pca_rmse*100:.1f}% of PCA RMSE.")
            print(f"     Adding more multipole orders will NOT close this gap.")
            print(f"     → To reach PCA-level reconstruction accuracy, consider:")
            print(f"       1. Export Hx, Hy, Hz vector components from Opera (Mode B)")
            print(f"       2. Use a smarter Gauss-Newton initialisation seeded")
            print(f"          from PCA-predicted field directions at sensor locations")
            print(f"       3. Accept the {mode_a_bias*MU0*1e9:.3f} nT bias as negligible")
            print(f"          (compare to your {1e-6/MU0:.2f} A/m experiment threshold)")
        elif pca_rmse and cv_rmses.min() < 2 * pca_rmse:
            improvement = (cv_rmses[0] - cv_rmses.min()) / cv_rmses[0] * 100
            print(f"  ✓  CONVERGING: RMSE fell {improvement:.1f}% across the sweep.")
            gap = cv_rmses.min() - pca_rmse
            if gap / pca_rmse < 0.2:
                print(f"     Minimum RMSE {cv_rmses.min():.6f} is within 20% of PCA benchmark.")
                print(f"     Recommended n_max={elbow_n_max} gives a good accuracy/complexity trade-off.")
            else:
                print(f"     Gap to PCA ({gap:.6f} A/m) suggests more orders or sensors would help.")
                print(f"     Recommended n_max={elbow_n_max} balances accuracy vs conditioning.")
        else:
            print(f"  ℹ  Insufficient data to determine convergence pattern.")
            print(f"     Recommended n_max={elbow_n_max} based on elbow criterion.")

        print(f"\n  Summary recommendation:")
        print(f"    n_max = {elbow_n_max}  ({2*elbow_n_max+2} coefficients, "
              f"oversampling = {n_meas/(2*elbow_n_max+2):.2f}x)")
        if pca_rmse:
            best_rmse = sweep_df.loc[sweep_df["n_max"]==elbow_n_max, "cv_rmse"].values[0]
            print(f"    Expected tube RMSE ≈ {best_rmse:.6f} A/m  "
                  f"({best_rmse*MU0*1e9:.3f} nT,  "
                  f"{best_rmse/pca_rmse:.2f}× PCA benchmark)")
        print()

    return {
        "sweep_df"         : sweep_df,
        "pca_rmse"         : pca_rmse,
        "elbow_n_max"      : elbow_n_max,
        "plateau_detected" : plateau_detected,
        "plateau_floor"    : plateau_floor,
        "mode_a_bias"      : mode_a_bias,
        "mode_a_bias_nT"   : mode_a_bias * MU0 * 1e9,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Combined entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_model_selection(
    df: pd.DataFrame,
    sensor_positions: list[tuple],
    n_max_range: range | list = range(1, 6),
    pca_rmse: float | None = None,
    noise_sigma_T: float = 1e-9,
    r0: float = R0_DEFAULT,
    alpha: float = DEFAULT_ALPHA,
    n_folds: int = 10,
    save_dir: Path | None = None,
) -> dict:
    """
    Run both Phase 4 (decision table) and Phase 5 (RMSE sweep) and return
    a combined payload.

    Typical usage after the PCA layout search:

        from multipole_model_selection import run_model_selection
        results = run_model_selection(
            df,
            sensor_positions=[(x1, y1), (x2, y2)],
            pca_rmse=pca_ref["field_rmse_cv"],   # from run_pca_reference()
        )
        # results["recommended_n_max"] is the integer to pass to generate_full_report

    Returns
    -------
    dict with keys:
        decision_table_df  : Phase 4 generic table
        geometry_table_df  : Phase 4 geometry-specific table
        sweep              : Phase 5 sweep dict (see rmse_vs_nmax_sweep)
        recommended_n_max  : int, the elbow n_max from Phase 5
    """
    print("\n" + "═"*80)
    print("  MODEL SELECTION: Phase 4 + Phase 5")
    print("  Sensor layout: " + str(sensor_positions))
    print("═"*80)

    # Phase 4a — generic table
    print("\n── Phase 4a: Generic sensor-count / order table ──")
    decision_df = sensor_order_decision_table(
        n_sensors_range=range(1, max(len(sensor_positions)+3, 7)),
        axes_per_sensor=3,
        oversampling_factor=2.0,
    )

    # Phase 4b — geometry-specific table for this layout
    print("── Phase 4b: Geometry-specific table for this layout ──")
    geometry_df = sensor_order_table_with_geometry(
        sensor_positions=sensor_positions,
        n_max_range=n_max_range,
        noise_sigma_T=noise_sigma_T,
        r0=r0,
    )

    # Phase 5 — sweep
    print("── Phase 5: CV RMSE sweep ──")
    sweep = rmse_vs_nmax_sweep(
        df=df,
        sensor_positions=sensor_positions,
        n_max_range=n_max_range,
        pca_rmse=pca_rmse,
        alpha=alpha,
        r0=r0,
        n_folds=n_folds,
        noise_sigma_T=noise_sigma_T,
    )

    recommended = sweep["elbow_n_max"]
    print(f"\n{'═'*80}")
    print(f"  FINAL RECOMMENDATION: n_max = {recommended}")
    print(f"  Use this value in generate_full_report(... n_max={recommended} ...)")
    print(f"{'═'*80}\n")

    if save_dir is not None:
        save_dir = Path(save_dir)
        decision_df.to_csv(save_dir / "phase4_decision_table.csv", index=False)
        geometry_df.to_csv(save_dir / "phase4_geometry_table.csv", index=False)
        sweep["sweep_df"].to_csv(save_dir / "phase5_rmse_sweep.csv", index=False)
        print(f"✓  Phase 4/5 tables saved to {save_dir}")

    return {
        "decision_table_df": decision_df,
        "geometry_table_df": geometry_df,
        "sweep"            : sweep,
        "recommended_n_max": recommended,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from magis_pca_pipeline import load_and_reclassify, build_tube_matrix, fit_pca

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "parsed_fields.csv"
    print(f"Loading: {csv_path}")

    df = load_and_reclassify(csv_path)

    # Default to the best 2-sensor layout from the PCA search if available,
    # otherwise fall back to a canonical example for demonstration
    try:
        from sensor_layout_search import search_layouts
        T, scenarios, _ = build_tube_matrix(df)
        pca_obj, K, A_pca, _ = fit_pca(T, variance_threshold=0.99)
        layout_df = search_layouts(df, T, A_pca, pca_obj, scenarios,
                                   n_sensors=2, noise_sigma_T=1e-9, verbose=False)
        best = layout_df.iloc[0]
        sensor_pos = [(float(best["sensor1_x"]), float(best["sensor1_y"])),
                      (float(best["sensor2_x"]), float(best["sensor2_y"]))]
        pca_rmse   = float(best["field_rmse_cv"])
        print(f"Best 2-sensor layout: {sensor_pos}  PCA RMSE={pca_rmse:.6f}")
    except Exception as e:
        print(f"Layout search unavailable ({e}), using example positions.")
        sensor_pos = [(-5., 0.), (2., 4.)]
        pca_rmse   = None

    results = run_model_selection(
        df=df,
        sensor_positions=sensor_pos,
        n_max_range=range(1, 5),
        pca_rmse=pca_rmse,
        noise_sigma_T=1e-9,
        save_dir=Path(csv_path).parent,
    )
