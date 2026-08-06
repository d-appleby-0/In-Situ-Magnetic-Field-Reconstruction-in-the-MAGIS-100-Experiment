"""
multipole_report.py — master analysis report
"""
import numpy as np
import pandas as pd
from pathlib import Path

from multipole_basis import (
    coefficient_names, reconstruct_field, reconstruct_magnitude,
    build_design_matrix, sensor_basis_row, phi_x_row, phi_y_row, phi_z_row,
    R0_DEFAULT,
)
from multipole_fitter import fit_all_scenarios, cross_validate_multipole, DEFAULT_N_MAX, DEFAULT_ALPHA
from sensor_orientation import (
    search_orientations, compare_named_orientations,
    fisher_matrix, condition_number, d_optimality, three_axis_orientations, MU0,
)
from plot_field_quiver import plot_vector_field_quiver
from plot_predicted_vs_actual import plot_predicted_vs_actual
from plot_gradient_heatmap import plot_gradient_heatmap
from sensor_misalignment import (
    compute_cov_theta_with_misalignment, MAX_SENSOR_MISALIGNMENT_DEG,
)


def run_pca_reference(df):
    from magis_pca_pipeline import build_tube_matrix, fit_pca, build_sensor_matrix, fit_regression, evaluate
    T, scenarios, _ = build_tube_matrix(df)
    pca, K, A, _    = fit_pca(T, variance_threshold=0.99)
    S, _            = build_sensor_matrix(df, scenarios)
    model, scaler, cv_preds = fit_regression(S, A)
    results = evaluate(pca, T, A, cv_preds, S, model, scaler, noise_levels=(0.0,), n_noise_trials=1)
    return {"K": K, "field_rmse_cv": results["field_rmse_cv"], "n_scenarios": len(scenarios)}


def print_field_equations(theta, n_max, scenario_label="", r0=R0_DEFAULT):
    label = f" [{scenario_label}]" if scenario_label else ""
    def fmt(v): return f"{v:+.6f}"
    print(f"\n  Field equations{label} (r0={r0} in):")
    print(f"  {'─'*64}")
    C0 = theta[0]
    lines_x = [f"  Hx(r,φ) = {fmt(C0)}  [uniform]"]
    for n in range(1, n_max+1):
        Cn=theta[2*n-1]; Sn=theta[2*n]
        if abs(Cn) > 1e-9 or abs(Sn) > 1e-9:
            lines_x.append(f"           {fmt(Cn)}·(r/r0)^{n}·cos({n}φ)  {fmt(-Sn)}·(r/r0)^{n}·sin({n}φ)")
    print("\n".join(lines_x))
    lines_y = [f"  Hy(r,φ) = 0.000000  [uniform→0]"]
    for n in range(1, n_max+1):
        Cn=theta[2*n-1]; Sn=theta[2*n]
        if abs(Cn) > 1e-9 or abs(Sn) > 1e-9:
            lines_y.append(f"           {fmt(Cn)}·(r/r0)^{n}·sin({n}φ)  {fmt(Sn)}·(r/r0)^{n}·cos({n}φ)")
    print("\n".join(lines_y))
    print(f"  Hz      = {fmt(theta[-1])}  [uniform within slice]")
    print(f"  {'─'*64}")


def print_sensor_equations(sensor_positions, orientations, n_max,
                           coeff_vals=None, r0=R0_DEFAULT):
    coeff_nms = coefficient_names(n_max)
    axis_labels = ["primary", "secondary", "tertiary"]
    for s_idx, ((x, y), orient) in enumerate(zip(sensor_positions, orientations)):
        r   = np.sqrt(x*x + y*y)
        phi = np.degrees(np.arctan2(y, x)) % 360
        axes = np.eye(3) if orient is None else (
               np.array(orient).reshape(1,3) if np.array(orient).ndim==1 else np.array(orient))
        print(f"\n  Sensor {s_idx+1} at (x={x:.2f}, y={y:.2f})  r={r:.3f} in  φ={phi:.1f}°")
        for a_idx, u in enumerate(axes):
            row = sensor_basis_row(x, y, u, n_max, r0=r0)
            lbl = axis_labels[a_idx] if a_idx < 3 else f"axis{a_idx+1}"
            print(f"    Axis {a_idx+1} ({lbl})  u=[{u[0]:+.3f}, {u[1]:+.3f}, {u[2]:+.3f}]:")
            first = True
            for name, val in zip(coeff_nms, row):
                if abs(val) < 1e-12: continue
                if first: print(f"      m = {val:+.6f}·{name}", end=""); first=False
                else: print(f"\n           {val:+.6f}·{name}", end="")
            print()
            if coeff_vals is not None:
                print(f"      → predicted: {float(row @ coeff_vals):.6f} A/m")


def build_measurement_labels(sensor_positions, orientations):
    axis_labels = ["primary", "secondary", "tertiary"]
    labels = []
    for s_idx, orient in enumerate(orientations):
        n_axes = 3 if orient is None else (1 if np.array(orient).ndim==1 else np.array(orient).shape[0])
        for a_idx in range(n_axes):
            lbl = axis_labels[a_idx] if a_idx < 3 else f"axis{a_idx+1}"
            labels.append(f"S{s_idx+1}.{lbl}")
    return labels


def print_mode_contributions(theta, n_max, tube_pts, r0=R0_DEFAULT):
    # quadrature power fractions, not linear magnitude sum
    order_names = {0:"n=0 uniform",1:"n=1 dipole",2:"n=2 quadrupole",
                   3:"n=3 sextupole",4:"n=4 octupole",5:"n=5 decapole",6:"n=6 dodecapole"}
    print(f"\n  {'─'*70}")
    print(f"  Multipole power contributions (RMS^2 quadrature, not linear |H| sum):")
    print(f"  {'─'*70}")

    H_total = reconstruct_field(theta, tube_pts, n_max, r0=r0)
    power_combined = float(np.mean(np.sum(H_total**2, axis=1)))

    mode_power = {}; mode_str = {}
    t = np.zeros_like(theta); t[0]=theta[0]
    mode_power["n=0 (uniform)"] = float(np.mean(np.sum(reconstruct_field(t,tube_pts,n_max,r0=r0)**2, axis=1)))
    mode_str["n=0 (uniform)"] = f"C0={theta[0]:+.4f}"
    for n in range(1, n_max+1):
        t = np.zeros_like(theta); t[2*n-1]=theta[2*n-1]; t[2*n]=theta[2*n]
        name = order_names.get(n, f"n={n}")
        mode_power[name] = float(np.mean(np.sum(reconstruct_field(t,tube_pts,n_max,r0=r0)**2, axis=1)))
        mode_str[name] = f"C{n}={theta[2*n-1]:+.4f} S{n}={theta[2*n]:+.4f}"
    t = np.zeros_like(theta); t[-1]=theta[-1]
    mode_power["Hz (axial)"] = float(np.mean(np.sum(reconstruct_field(t,tube_pts,n_max,r0=r0)**2, axis=1)))
    mode_str["Hz (axial)"] = f"Hz={theta[-1]:+.4f}"

    sum_power = sum(mode_power.values())
    print(f"  {'Order':<22} {'Coefficients':<28} {'RMS |H|':>10} {'% power':>10}")
    print(f"  {'-'*74}")
    for name, pwr in mode_power.items():
        frac = 100*pwr/sum_power if sum_power > 0 else 0.
        print(f"  {name:<22} {mode_str[name]:>28}   {np.sqrt(pwr):>10.6f}   {frac:>8.2f}%")
    print(f"  {'-'*74}")
    print(f"  {'Total (quadrature)':<22} {'':>28}   {np.sqrt(sum_power):>10.6f}   {'100.00%':>9}")
    orth = sum_power/power_combined if power_combined > 0 else float('nan')
    verdict = "   well-separated" if abs(orth-1) < 0.05 else "X  sampling not uniform in φ"
    print(f"\n  Orthogonality check: {orth:.4f}  {verdict}")
    return {"mode_power": mode_power, "sum_power": sum_power,
            "power_combined": power_combined, "orthogonality": orth}


def print_uncertainty_budget(A, theta, n_max, tube_pts,
                             sensor_positions=None, sensor_orientations=None,
                             delta_max_deg=MAX_SENSOR_MISALIGNMENT_DEG,
                             noise_sigma_T=1e-9, rcond=1e-10, r0=R0_DEFAULT):
    """
    SVD-truncated pseudoinverse, effective DOF, full-covariance
    delta method, now including sensor mounting misalignment uncertainty
    in quadrature with the sensor noise floor, via
    compute_cov_theta_with_misalignment().

    sensor_positions/sensor_orientations - required to include the
        misalignment contribution. If either is None, falls back to
        noise-only (equivalent to delta_max_deg=0), matching the original
        behavior
    """
    sigma_H   = noise_sigma_T / MU0
    coeff_nms = coefficient_names(n_max)

    if sensor_positions is not None and sensor_orientations is not None:
        cov, sigma_total_row, sigma_misalign_row = compute_cov_theta_with_misalignment(
            A, sensor_positions, sensor_orientations, theta, n_max, r0=r0,
            noise_sigma_T=noise_sigma_T, delta_max_deg=delta_max_deg, rcond=rcond)
        print(f"\n  {'─'*70}")
        print(f"  Coefficient uncertainties  σ_sensor={noise_sigma_T:.1e} T = {sigma_H:.4e} A/m"
              f"   +  mounting misalignment ≤{delta_max_deg}°")
        print(f"  Per-row total σ: median={np.median(sigma_total_row):.4e} A/m "
              f"(noise-only portion would be {sigma_H:.4e} A/m;"
              f" misalignment median={np.median(sigma_misalign_row):.4e} A/m)")
    else:
        # unchanged fallback
        U, s_thin, Vt = np.linalg.svd(A, full_matrices=True)
        s = np.zeros(2*n_max+2); s[:len(s_thin)] = s_thin
        s_max = s[0] if len(s) else 0.
        well = s > rcond * s_max
        s_inv2 = np.zeros_like(s); s_inv2[well] = 1./(s[well]**2)
        cov = (Vt.T * s_inv2) @ Vt * sigma_H**2
        print(f"\n  {'─'*70}")
        print(f"  Coefficient uncertainties  σ_sensor={noise_sigma_T:.1e} T = {sigma_H:.4e} A/m")
        print(f"  (sensor_positions/orientations not supplied -- misalignment NOT included)")

    dof = int(np.sum(np.linalg.eigvalsh(cov) > rcond))  # effective DOF from cov's own rank
    sigma_theta = np.sqrt(np.clip(np.diag(cov), 0, None))

    print(f"  Effective DOF: {dof}/{2*n_max+2}")
    print(f"  {'─'*70}")
    print(f"  {'Coefficient':<22} {'Best-fit':>14} {'1σ':>14} {'σ/|θ|':>10} {'Status':>14}")
    print(f"  {'-'*78}")
    unresolved = []
    for name, val, unc in zip(coeff_nms, theta, sigma_theta):
        ratio  = unc/(abs(val)+1e-15)
        status = "NOT RESOLVED" if ratio > 3 else "ok"
        if ratio > 3: unresolved.append(name)
        print(f"  {name:<22} {val:>14.4e} {unc:>14.4e} {ratio:>9.2f}x {status:>14}")
    if unresolved:
        print(f"\n  X Not resolved (σ > 3x fitted value): {', '.join(unresolved)}")
        print(f"    Add sensors, reduce n_max, or use a different orientation.")
    else:
        print(f"\n    All coefficients resolved within 3x uncertainty.")

    n_null = (2*n_max+2) - dof
    if n_null > 0:
        print(f"\n  X {n_null} unconstrained direction(s) in coefficient space:")
        for j in np.where(~well)[0]:
            direction = Vt[j]
            top = np.argsort(-np.abs(direction))[:4]
            terms = "  ".join(f"{direction[k]:+.3f}·{coeff_nms[k]}"
                              for k in top if abs(direction[k]) > 1e-9)
            print(f"      {terms}")

    # Full-covariance delta method for tube field uncertainty
    tube_unc = []
    for (x, y) in tube_pts:
        Jp = np.vstack([phi_x_row(x,y,n_max,r0=r0),
                        phi_y_row(x,y,n_max,r0=r0),
                        phi_z_row(n_max)])
        H_vec  = Jp @ theta; norm_H = np.linalg.norm(H_vec)
        cov_H  = Jp @ cov @ Jp.T
        if norm_H < 1e-15:
            sigma_pt = np.sqrt(max(np.trace(cov_H)/3., 0.))
        else:
            h_hat = H_vec/norm_H
            sigma_pt = np.sqrt(max(float(h_hat @ cov_H @ h_hat), 0.))
        tube_unc.append(sigma_pt)
    tube_unc = np.array(tube_unc)
    print(f"\n  Propagated |H| uncertainty in tube interior:")
    print(f"    mean : {tube_unc.mean():.4e} A/m  ({tube_unc.mean()*MU0*1e9:.4f} nT)")
    print(f"    max  : {tube_unc.max():.4e} A/m  ({tube_unc.max()*MU0*1e9:.4f} nT)")
    print(f"    Experiment threshold: several μT = {1e-6/MU0:.2f} A/m")
    return {"sigma_theta": sigma_theta, "cov_theta": cov, "dof_effective": dof,
            "unresolved": unresolved, "tube_field_unc": tube_unc}


def print_jacobian_matrix(A, n_max, measurement_labels=None, rcond=1e-10, top_k=4):
    # explicit dθ/dm Jacobian following g-2 Appendix B
    coeff_nms = coefficient_names(n_max)
    n_meas = A.shape[0]
    if measurement_labels is None:
        measurement_labels = [f"m{k}" for k in range(n_meas)]
    J = np.linalg.pinv(A, rcond=rcond)
    print(f"\n  {'─'*70}")
    print(f"  Jacobian θ = J @ m  (dθ_i/dm_k) — top {top_k} contributors per coefficient:")
    print(f"  {'─'*70}")
    for i, name in enumerate(coeff_nms):
        row = J[i]; top = np.argsort(-np.abs(row))[:top_k]
        terms = "  ".join(f"{row[k]:+.4f}·{measurement_labels[k]}"
                          for k in top if abs(row[k]) > 1e-12)
        print(f"  {name:<22} ‖row‖={np.linalg.norm(row):>8.4f}   {terms}")
    return J

def print_plug_and_play_equations(A, measurement_labels, target_points, n_max,
                                  r0=R0_DEFAULT, rcond=1e-10):
    """
    Step 11: direct measurement -> field formulas at specific target points.
    Composes the fit (theta = pinv(A)@m) with field reconstruction into ONE
    numeric equation per target point/component, plug in raw sensor
    readings and get the field, no intermediate fitting step for the reader
    to perform themselves.
    """
    from multipole_basis import build_measurement_to_field_map
    mapping = build_measurement_to_field_map(A, target_points, n_max, r0=r0, rcond=rcond)
    n_coeffs = A.shape[1]
    rank_deficient = mapping["dof_effective"] < n_coeffs
 
    print(f"\n  {'─'*70}")
    print(f"  Plug-and-play field equations:  H(target) = Σ_k M_k · m_k")
    print(f"  (straight from raw sensor readings m -- no separate fitting step)")
    print(f"  {'─'*70}")
    if rank_deficient:
        print(f"  X Design matrix is rank-deficient ({mapping['dof_effective']}/{n_coeffs} "
              f"effective DOF). These coefficients are the MINIMUM-NORM solution,")
        print(f"    not unique -- treat with the same caution as the Step 9 uncertainty budget.")
 
    for i, (x, y) in enumerate(target_points):
        print(f"\n  Target point (x={x:.3f}, y={y:.3f}):")
        for comp, M in (("Hx", mapping["M_x"]), ("Hy", mapping["M_y"]), ("Hz", mapping["M_z"])):
            row = M[i]
            terms = [f"{val:+.6f}·m({lbl})"
                     for val, lbl in zip(row, measurement_labels) if abs(val) > 1e-9]
            eq = "  ".join(terms) if terms else "0  (no measurement contributes above threshold)"
            print(f"    {comp} = {eq}")
 
    return mapping
 
 
def print_transverse_gradient(cov_theta, theta, n_max, tube_pts, r0=R0_DEFAULT):
    """
    Step 12: transverse field gradient at representative tube-interior
    points, a direct uniformity diagnostic. Reuses the fitted
    theta (no new fitting) and the coefficient covariance from Step 9 for
    error propagation via the same delta-method technique, using the
    derivative basis rows in place of the value rows.
    """
    from multipole_basis import (
        reconstruct_gradient, phi_x_dx_row, phi_x_dy_row, phi_y_dx_row, phi_y_dy_row,
    )
    G = reconstruct_gradient(theta, tube_pts, n_max, r0=r0)
    grad_frobenius = np.sqrt(np.sum(G**2, axis=(1, 2)))
 
    print(f"\n  {'─'*70}")
    print(f"  Transverse field gradient (uniformity diagnostic)")
    print(f"  {'─'*70}")
    print(f"  {'point':<14} {'dHx/dx':>11} {'dHx/dy':>11} {'dHy/dx':>11} "
          f"{'dHy/dy':>11} {'‖∇H‖':>11} {'σ(‖∇H‖)':>11}")
    print("  " + "-"*90)
 
    sigmas = []
    for i, (x, y) in enumerate(tube_pts):
        rows = [phi_x_dx_row(x, y, n_max, r0=r0), phi_x_dy_row(x, y, n_max, r0=r0),
                phi_y_dx_row(x, y, n_max, r0=r0), phi_y_dy_row(x, y, n_max, r0=r0)]
        R = np.vstack(rows)                      # (4, n_coeffs)
        g_vec  = R @ theta                        # matches G[i] flattened
        Cov_g  = R @ cov_theta @ R.T               # (4,4)
        norm_g = np.linalg.norm(g_vec)
        if norm_g < 1e-15:
            sigma_norm = float(np.sqrt(max(np.trace(Cov_g)/4., 0.)))
        else:
            ghat = g_vec / norm_g
            sigma_norm = float(np.sqrt(max(float(ghat @ Cov_g @ ghat), 0.)))
        sigmas.append(sigma_norm)
 
        g = G[i]
        print(f"  ({x:+.2f},{y:+.2f})    {g[0,0]:>11.4e} {g[0,1]:>11.4e} "
              f"{g[1,0]:>11.4e} {g[1,1]:>11.4e} {grad_frobenius[i]:>11.4e} {sigma_norm:>11.4e}")
 
    sigmas = np.array(sigmas)
    print(f"\n  Mean ‖∇H‖ across sampled points: {grad_frobenius.mean():.4e} A/m per inch")
    print(f"  Max  ‖∇H‖ across sampled points: {grad_frobenius.max():.4e} A/m per inch")
    print(f"  (smaller ‖∇H‖ = more spatially uniform field across the tube cross-section)")
 
    return {"gradient_tensors": G, "grad_frobenius": grad_frobenius,
            "sigma_grad_frobenius": sigmas}


def generate_full_report(df, sensor_positions, n_max=DEFAULT_N_MAX,
                         noise_sigma_T=1e-9, run_orientation_search=True,
                         r0=R0_DEFAULT, save_dir=None, orientation_cache=None,
                         plug_and_play_points=None, gradient_points=None,
                         pair_label=None, target_J=66.039, target_slice=None,
                         use_smoothed_theta=True):
    """
    orientation_cache : optional dict, previously returned as
        payload["orientation_cache"] from an earlier call to this function
        with the same (sensor_positions, n_max, r0, noise_sigma_T). Currently
        just a recommendation, Cartesian layout is always used
    plug_and_play_points : optional list of (x,y) target points for the
        Step 11 plug-and-play equations. Defaults to tube center + one
        edge point at (0.9*r0, 0) if not supplied.
    gradient_points : optional list of (x,y) target points for the Step 12
        transverse gradient diagnostic. Defaults to tube center plus 4
        edge points (N/E/S/W at 0.9*r0) if not supplied.
    """
    if plug_and_play_points is None:
        plug_and_play_points = [(0.0, 0.0), (0.9*r0, 0.0)]
    if gradient_points is None:
        gradient_points = [(0.0, 0.0), (0.9*r0, 0.0), (0.0, 0.9*r0),
                           (-0.9*r0, 0.0), (0.0, -0.9*r0)]
    
    sep = "═"*68
    print(f"\n{sep}")
    print(f"  MAGIS-100 FIELD RECONSTRUCTION REPORT")
    print(f"  Sensors: {sensor_positions}   n_max={n_max} ({2*n_max+2} coeffs)   r0={r0} in")
    print(f"  Scenarios: {df['scenario_id'].nunique()}")
    print(f"{sep}\n")

    print("STEP 1 — PCA Pipeline Reference"); print("─"*40)
    try:
        pca_ref = run_pca_reference(df)
        print(f"  K={pca_ref['K']}  CV RMSE={pca_ref['field_rmse_cv']:.3e} A/m")
    except Exception as e:
        print(f"  Skipped: {e}"); pca_ref = {}

    print("\nSTEP 2 — Multipole Fits"); print("─"*40)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_df = fit_all_scenarios(df, sensor_xy=sensor_positions, n_max=n_max, r0=r0, mode="auto")
    coeff_nms = coefficient_names(n_max)
    has_vector = ("field_Hx" in df.columns and df["field_Hx"].notna().any())
    fit_mode_used = "vector" if has_vector else "magnitude"
    print(f"  Fit mode: {fit_mode_used}"
          + ("" if has_vector else
             "  (no vector Hx/Hy/Hz data available - using scalar |H| readings)"))
    if fit_mode_used == "magnitude":
        print(f"  X NOTE: |H|=sqrt(Hx²+Hy²+Hz²) magnitude-only fitting has a structural")
        print(f"    degeneracy between C0 (uniform transverse) and Hz (axial): both add a")
        print(f"    position-independent contribution to the total magnitude, so when")
        print(f"    dipole/higher-order terms are weak, magnitude data cannot reliably")
        print(f"    tell you how much of that contribution is C0 vs Hz -- many different")
        print(f"    splits fit the sensor data almost equally well. The fit below uses a")
        print(f"    symmetry-broken initial guess so Hz is no longer trivially forced to")
        print(f"    exactly 0, but the specific Hz value reported should be treated with")
        print(f"    skepticism. If precise axial recovery matters, export Hx/Hy/Hz")
        print(f"    vector components from Opera (see parse_table_files.py docstring) -")
        print(f"    vector-mode fitting resolves Hz unambiguously via direct least squares.")
    print(f"  Fitted {len(fit_df)} scenarios")
    print(f"  Sensor RMSE : {fit_df['rmse_sensor'].mean():.6f} ± {fit_df['rmse_sensor'].std():.6f} A/m")
    print(f"  Tube RMSE   : {fit_df['rmse_tube'].mean():.6f} ± {fit_df['rmse_tube'].std():.6f} A/m")
    finite_cond = fit_df["condition_number"].replace([np.inf,-np.inf], np.nan).dropna()
    if len(finite_cond):
        print(f"  Cond number : median={finite_cond.median():.2f}  mean={finite_cond.mean():.2f}")

    print("\nSTEP 3 — Cross-Validation (Multipole vs PCA)"); print("─"*40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cross_validate_multipole(df, sensor_xy=sensor_positions, n_max=n_max, r0=r0, mode="auto")
    print(f"  Multipole CV RMSE : {cv['mean_rmse']:.6f} ± {cv['std_rmse']:.6f} A/m")
    if pca_ref and pca_ref.get('field_rmse_cv', 0) > 0:
        ratio   = cv['mean_rmse'] / pca_ref['field_rmse_cv']
        verdict = "  comparable" if ratio < 2 else "X worse than PCA"
        print(f"  Ratio to PCA      : {ratio:.3f}x  {verdict}")

    if orientation_cache is not None:
        print("\nSTEP 4/5 — Orientation Strategy Comparison + Grid Search"); print("─"*40)
        print(f"  Reusing cached orientation result (identical sensor geometry, "
              f"n_max={n_max}, r0={r0} to a previously processed pair).")
        print(f"  Steps 4-5 depend only on sensor positions/n_max/r0, not on field data,")
        print(f"  so this is an exact reuse, not an approximation.")
        orient_df          = orientation_cache["orient_df"]
        best_named         = orientation_cache["best_named"]
        best_orientations  = orientation_cache["best_orientations"]
        opt_df_out         = orientation_cache["opt_df_out"]
    else:
        print("\nSTEP 4 — Orientation Strategy Comparison"); print("─"*40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orient_df = compare_named_orientations(sensor_positions, n_max=n_max,
                                                   noise_sigma_T=noise_sigma_T, r0=r0)
        print(orient_df[["orientation_strategy","D_optimality","condition_number"]].to_string(index=False))
        best_named = orient_df.iloc[0]["orientation_strategy"]
        print(f"\n  Best named strategy: {best_named}")

        if run_orientation_search:
            print("\nSTEP 5 — Optimal Orientation Grid Search"); print("─"*40)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt_df = search_orientations(sensor_positions, n_max=n_max,
                                             theta_steps=9, phi_steps=12,
                                             criterion="D", noise_sigma_T=noise_sigma_T, r0=r0)
            best_orientations = []
            for _, row in opt_df.iterrows():
                best_orientations.append(np.array([
                    [row["u1_x"], row["u1_y"], row["u1_z"]],
                    [row["u2_x"], row["u2_y"], row["u2_z"]],
                    [row["u3_x"], row["u3_y"], row["u3_z"]],
                ]))
            opt_df_out = opt_df
        else:
            from sensor_orientation import radial_orientation, tangential_orientation, axial_orientation
            fn = {"radial": radial_orientation, "tangential": tangential_orientation,
                  "axial": lambda x,y: axial_orientation(), "cartesian": lambda x,y: None}
            best_orientations = [fn.get(best_named, radial_orientation)(x,y) for x,y in sensor_positions]
            opt_df_out = None

        print(f"\n  NOTE: the above is a theoretical free-orientation optimum, shown for "
              f"comparison only. Sensors are physically mounted along fixed x/y/z axes, "
              f"so the report below uses the Cartesian orientation for all downstream "
              f"steps and uncertainty propagation.")

        # This is what actually gets used from here on:
        best_orientations = [None] * len(sensor_positions)

    print("\nSTEP 6 — Field Equations (representative scenario)"); print("─"*40)
    from plot_field_quiver import select_theta_for_J
    from coefficient_smoothing import smooth_coefficients_vs_J

    # smooth once per slice present in this pair's fit_df, before selecting theta_rep
    coeff_nms = coefficient_names(n_max)
    if use_smoothed_theta:
        smoothed_frames = []
        for sid_num in sorted(fit_df["slice_id"].unique()):
            sdf, _ = smooth_coefficients_vs_J(fit_df, n_max, slice_id=sid_num,
                                              method="linear_through_origin",
                                              coeff_names_override=coeff_nms)
            smoothed_frames.append(sdf)
        fit_df = pd.concat(smoothed_frames, ignore_index=True)
    theta_rep, rep_row = select_theta_for_J(fit_df, coeff_nms, J_target=66.039, slice_id=target_slice,
                                            use_smoothed=use_smoothed_theta)
    print_field_equations(theta_rep, n_max,
                          f"{rep_row['scenario_id']} J={rep_row['coil_density']:.0f} A/in² z={rep_row['z']:.2f} in",
                          r0=r0)

    print("\nSTEP 7 — Sensor Measurement Equations"); print("─"*40)
    print_sensor_equations(sensor_positions, best_orientations, n_max, theta_rep, r0=r0)
    
    print("\nSTEP 8 — Multipole Order Contributions"); print("─"*40)
    tube_pts = list(zip(
        df[df["region"]=="tube_interior"]["x"].tolist()[:30],
        df[df["region"]=="tube_interior"]["y"].tolist()[:30]
    ))
    mode_contrib = print_mode_contributions(theta_rep, n_max, tube_pts, r0=r0)

    print("\nSTEP 9 — Uncertainty Budget"); print("─"*40)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A_best = build_design_matrix(sensor_positions, best_orientations, n_max, r0=r0)
    uncertainty = print_uncertainty_budget(A_best, theta_rep, n_max, tube_pts,
                                           sensor_positions=sensor_positions,
                                           sensor_orientations=best_orientations,
                                           delta_max_deg=MAX_SENSOR_MISALIGNMENT_DEG,
                                           noise_sigma_T=noise_sigma_T, r0=r0)

    print("\nSTEP 10 — Jacobian (measurements → coefficients)"); print("─"*40)
    meas_labels = build_measurement_labels(sensor_positions, best_orientations)
    jacobian    = print_jacobian_matrix(A_best, n_max, measurement_labels=meas_labels)

    print("\nSTEP 11 — Plug-and-Play Field Equations"); print("─"*40)
    plug_and_play = print_plug_and_play_equations(
        A_best, meas_labels, plug_and_play_points, n_max, r0=r0)
 
    print("\nSTEP 12 — Transverse Field Gradient (Uniformity Diagnostic)"); print("─"*40)
    gradient_result = print_transverse_gradient(
        uncertainty["cov_theta"], theta_rep, n_max, gradient_points, r0=r0)

    plot_gradient_heatmap(
        theta_rep, n_max, r0=r0, A=A_best,  # already built earlier in Step 9
        sensor_positions=sensor_positions, sensor_orientations=best_orientations,
        noise_sigma_T=noise_sigma_T,   # already a parameter of generate_full_report
        extend_to_sensor_zone=True,
        out_path=f"C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/gradient_heatmap_{pair_label}.pdf",   # use the per-pair pid, per the earlier filename-collision fix
        title=f"Transverse Field Gradient — {pair_label}",
    )


    print(f"\n{sep}\n  END OF REPORT\n{sep}\n")

    orientation_cache_out = {
        "orient_df": orient_df, "best_named": best_named,
        "best_orientations": best_orientations, "opt_df_out": opt_df_out,
        "_signature": (tuple(sensor_positions), n_max, r0, noise_sigma_T),
    }

    payload = {"pca_reference": pca_ref, "fit_df": fit_df, "cv_multipole": cv,
               "orientation_df": orient_df, "opt_orientation_df": opt_df_out,
               "best_orientations": best_orientations, "theta_representative": theta_rep,
               "design_matrix": A_best, "mode_contributions": mode_contrib,
               "uncertainty": uncertainty, "jacobian": jacobian,
               "orientation_cache": orientation_cache_out, "plug_and_play": plug_and_play,
               "gradient": gradient_result}

    plot_vector_field_quiver(
        theta_rep, n_max, r0,
        out_path=f"C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/pair_{pair_label}_field.pdf",
        title=f"Magnetic Field Map (J={rep_row['coil_density']:.2f} A/in², Pair {pair_label})",
        sensor_positions=sensor_positions,
    )

    plot_predicted_vs_actual(
        df, fit_df, n_max=n_max, r0=r0, color_by="J", noise_sigma_T=noise_sigma_T, 
        out_path=f"C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/Predicted_vs_actual_{pair_label}.pdf",
    )

    from plot_predicted_vs_actual_components import (
        plot_predicted_vs_actual_components, plot_diff_vs_J,
    )

    # Component-wise (only if you have vector-mode Opera exports)
    plot_predicted_vs_actual_components(
        df, fit_df, n_max=n_max, r0=r0, noise_sigma_T=noise_sigma_T,
        out_path=f"C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/components_{pair_label}.pdf",
    )

    # Difference vs J (works regardless of fit mode)
    plot_diff_vs_J(
        df, fit_df, n_max=n_max, r0=r0, noise_sigma_T=noise_sigma_T,
        out_path=f"C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/diff_vs_J_{pair_label}.pdf",
    )

    from plot_predicted_vs_actual_components import plot_diff_vs_J_components
    from plot_reconstruction_error_heatmap import (
        plot_reconstruction_error_heatmap, plot_reconstruction_error_heatmap_by_slice,
    )

    # Componentized difference vs J (requires vector-mode data)
    plot_diff_vs_J_components(df, fit_df, n_max, r0=r0, noise_sigma_T=noise_sigma_T,
                              out_path=f"C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/diff_components_{pair_label}.pdf")
    
    # Single-slice spatial error map
    plot_reconstruction_error_heatmap(df, fit_df, n_max, r0=r0, J_target=target_J, slice_id=1, theta=theta_rep, row=rep_row,
                                      out_path=f"C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/error_heatmap_{pair_label}_J66.039_s1.pdf")
    
    # Both slices at once, shared color scale
    plot_reconstruction_error_heatmap_by_slice(df, fit_df, n_max, r0=r0, J_target=target_J, theta=theta_rep, row=rep_row,
                                               out_path=f"C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/error_by_slice_{pair_label}_J66.039.pdf")

    

    if save_dir is not None:
        save_dir = Path(save_dir)
        fit_df.to_csv(save_dir/"multipole_coefficients.csv", index=False)
        orient_df.to_csv(save_dir/"orientation_comparison.csv", index=False)
        if opt_df_out is not None:
            opt_df_out.to_csv(save_dir/"optimal_orientations.csv", index=False)
        print(f"   Tables saved to {save_dir}")
    return payload
