"""
multipole_fitter.py — vectorised Gauss-Newton multipole fitting
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from recursive_order_refinement import fit_incremental_order
from multipole_basis import (
    build_design_matrix, reconstruct_magnitude, reconstruct_field,
    coefficient_names, phi_x_row, phi_y_row, phi_z_row, sensor_basis_row,
    R0_DEFAULT,
)

DEFAULT_N_MAX  = 3
DEFAULT_ALPHA  = 1e-3
DEFAULT_N_ITER = 8
DEFAULT_TOL    = 1e-8
MU0            = 4 * np.pi * 1e-7
"""
Symmetry-breaking seed for the initial Hz guess in magnitude-mode
Gauss-Newton fitting
"""
HZ_SEED_FRACTION = 0.05


def _magnitude_design_row(x, y, H0, n_max, r0=R0_DEFAULT):
    h_norm = np.linalg.norm(H0)
    h_hat  = H0 / h_norm if h_norm > 1e-15 else np.array([1.,0.,0.])
    return sensor_basis_row(x, y, h_hat, n_max, r0=r0)


def fit_slice_magnitude(sensor_xy, H_magnitudes, n_max=DEFAULT_N_MAX,
                        alpha=DEFAULT_ALPHA, n_iter=DEFAULT_N_ITER,
                        tol=DEFAULT_TOL, r0=R0_DEFAULT,
                        theta_init=None, return_diagnostics=False):
    n_coeffs = 2 * n_max + 2
    theta = (np.array(theta_init, dtype=float, copy=True)
             if theta_init is not None else np.zeros(n_coeffs))
    if theta_init is None:
        theta[0] = float(np.mean(H_magnitudes))
        theta[-1] = HZ_SEED_FRACTION * theta[0]

    prev_theta = theta.copy(); A = None; n_used = n_iter
    for iteration in range(n_iter):
        H0_vecs = reconstruct_field(theta, sensor_xy, n_max, r0=r0)
        A_rows  = [_magnitude_design_row(x, y, H0_vecs[i], n_max, r0=r0)
                   for i, (x, y) in enumerate(sensor_xy)]
        A = np.vstack(A_rows)
        reg = Ridge(alpha=alpha, fit_intercept=False)
        reg.fit(A, H_magnitudes)
        theta = reg.coef_
        rel_change = np.max(np.abs(theta - prev_theta) / (np.abs(prev_theta) + 1e-12))
        prev_theta = theta.copy()
        if rel_change < tol:
            n_used = iteration + 1; break

    H0_vecs = reconstruct_field(theta, sensor_xy, n_max, r0=r0)
    if not return_diagnostics:
        return theta, H0_vecs

    sensor_pred = np.linalg.norm(H0_vecs, axis=1)
    rmse_sensor = float(np.sqrt(np.mean((sensor_pred - H_magnitudes)**2)))
    try:    cond = float(np.linalg.cond(A))
    except: cond = float("inf")
    return theta, H0_vecs, {"rmse_sensor": rmse_sensor,
                             "condition_number": cond,
                             "n_iterations_used": n_used}


def _precompute_point_basis(points, n_max, r0=R0_DEFAULT):
    Phi_x = np.vstack([phi_x_row(x, y, n_max, r0=r0) for x, y in points])
    Phi_y = np.vstack([phi_y_row(x, y, n_max, r0=r0) for x, y in points])
    Phi_z = np.tile(phi_z_row(n_max), (len(points), 1))
    return Phi_x, Phi_y, Phi_z


def _pivot_scenario_matrix(region_df, points, value_col="field_value"):
    df = region_df.copy()
    df["x"] = df["x"].round(6); df["y"] = df["y"].round(6)
    df["xy"] = list(zip(df["x"], df["y"]))
    point_set = set((round(x,6), round(y,6)) for x,y in points)
    df = df[df["xy"].isin(point_set)]
    pivot = df.pivot_table(index="scenario_id", columns="xy", values=value_col)
    pts_r = [(round(x,6), round(y,6)) for x,y in points]
    pivot = pivot.reindex(columns=pts_r)
    if pivot.isna().any().any():
        raise ValueError(
            f"{int(pivot.isna().sum().sum())} (scenario,point) combos missing on {value_col} — "
            f"falling back to loop fit")
    return pivot.values, pivot.index.tolist()


def _fit_magnitude_batched(M, Phi_x, Phi_y, Phi_z,
                           alpha=DEFAULT_ALPHA, n_iter=DEFAULT_N_ITER,
                           tol=DEFAULT_TOL, theta_init=None):
    S, K = M.shape; C = Phi_x.shape[1]
    Theta = (np.array(theta_init, dtype=float, copy=True)
             if theta_init is not None else np.zeros((S, C)))
    if theta_init is None:
        Theta[:, 0]  = M.mean(axis=1)
        """
        same fixed-point trap as fit_slice_magnitude. Theta[:,-1]=0 makes
        every sensor's h_hat=[1,0,0], which zeroes the Hz column of the
        design matrix at every iteration.
        """
        Theta[:, -1] = HZ_SEED_FRACTION * Theta[:, 0]
    I_reg = alpha * np.eye(C); prev_theta = Theta.copy(); n_used = n_iter

    def _forward(Theta):
        Hx0 = Theta @ Phi_x.T; Hy0 = Theta @ Phi_y.T; Hz0 = Theta @ Phi_z.T
        norm0 = np.sqrt(Hx0**2 + Hy0**2 + Hz0**2)
        safe  = np.where(norm0 < 1e-15, 1., norm0)
        hx = np.where(norm0 < 1e-15, 1., Hx0/safe)
        hy = np.where(norm0 < 1e-15, 0., Hy0/safe)
        hz = np.where(norm0 < 1e-15, 0., Hz0/safe)
        return norm0, hx, hy, hz

    for iteration in range(n_iter):
        _, hx, hy, hz = _forward(Theta)
        A = (hx[:,:,None]*Phi_x[None,:,:] + hy[:,:,None]*Phi_y[None,:,:]
             + hz[:,:,None]*Phi_z[None,:,:])
        AtA = np.einsum('ski,skj->sij', A, A) + I_reg[None,:,:]
        Atm = np.einsum('ski,sk->si', A, M)
        Theta = np.linalg.solve(AtA, Atm[...,None])[...,0]
        delta = np.linalg.norm(Theta - prev_theta, axis=1)
        rel_change = np.max(delta / (np.linalg.norm(prev_theta, axis=1) + 1e-12))
        prev_theta = Theta.copy()
        if rel_change < tol:
            n_used = iteration + 1; break

    norm0, hx, hy, hz = _forward(Theta)
    A_final = (hx[:,:,None]*Phi_x[None,:,:] + hy[:,:,None]*Phi_y[None,:,:]
               + hz[:,:,None]*Phi_z[None,:,:])
    sv = np.linalg.svd(A_final, compute_uv=False)
    cond = np.where(sv[:,-1] > 1e-15, sv[:,0]/sv[:,-1], np.inf)
    return Theta, norm0, cond, n_used


def _resolve_mode(df: pd.DataFrame, mode: str) -> str:
    """
    Resolve mode='auto' to 'vector' if the dataframe actually has usable
    vector components (field_Hx/Hy/Hz, non-NaN), else falls back to
    'magnitude'. Any other mode value passes through unchanged so existing
    callers using mode='magnitude' explicitly are unaffected by not opting in.
    """
    if mode != "auto":
        return mode
    has_vector = (
        "field_Hx" in df.columns and df["field_Hx"].notna().any()
        and "field_Hy" in df.columns and df["field_Hy"].notna().any()
        and "field_Hz" in df.columns and df["field_Hz"].notna().any()
    )
    return "vector" if has_vector else "magnitude"


def fit_all_scenarios(df, sensor_xy=None, n_max=DEFAULT_N_MAX,
                      alpha=DEFAULT_ALPHA, mode="magnitude",
                      r0=R0_DEFAULT, n_iter=DEFAULT_N_ITER, tol=DEFAULT_TOL,
                      sensor_orientations=None):
    mode = _resolve_mode(df, mode)
    coeff_names = coefficient_names(n_max)
    sz = df[df["region"] == "sensor_zone"]
    tube_df = df[df["region"] == "tube_interior"]
    scenarios = sorted(df["scenario_id"].unique())

    if sensor_xy is None:
        ref = sz[sz["scenario_id"] == scenarios[0]][["x","y"]].drop_duplicates()
        sensor_xy = list(zip(ref["x"].round(6), ref["y"].round(6)))

    tube_ref = tube_df[tube_df["scenario_id"] == scenarios[0]][["x","y"]].drop_duplicates()
    tube_xy  = list(zip(tube_ref["x"].round(6), tube_ref["y"].round(6)))

    try:
        M_sensor, scen_order = _pivot_scenario_matrix(sz, sensor_xy, "field_value")
        M_tube, tube_scen    = _pivot_scenario_matrix(tube_df, tube_xy, "field_value")
    except ValueError as e:
        warnings.warn(f"Falling back to loop: {e}", stacklevel=2)
        return fit_all_scenarios_loop(df, sensor_xy=sensor_xy, n_max=n_max,
                                      alpha=alpha, mode=mode, r0=r0, n_iter=n_iter, tol=tol)

    if scen_order != tube_scen:
        lu = {sid: i for i, sid in enumerate(tube_scen)}
        M_tube = M_tube[[lu[s] for s in scen_order]]

    Phi_x, Phi_y, Phi_z = _precompute_point_basis(sensor_xy, n_max, r0=r0)
    Phi_xt, Phi_yt, Phi_zt = _precompute_point_basis(tube_xy, n_max, r0=r0)

    if mode == "magnitude":
        Theta, sensor_pred, cond, n_used = _fit_magnitude_batched(
            M_sensor, Phi_x, Phi_y, Phi_z, alpha=alpha, n_iter=n_iter, tol=tol)
        n_used_col = np.full(len(scen_order), n_used)
    elif mode == "vector":
        M_sensor_x, _ = _pivot_scenario_matrix(sz, sensor_xy, "field_Hx")
        M_sensor_y, _ = _pivot_scenario_matrix(sz, sensor_xy, "field_Hy")
        M_sensor_z, _ = _pivot_scenario_matrix(sz, sensor_xy, "field_Hz")

        Phi_stacked = np.vstack([Phi_x, Phi_y, Phi_z])
        M_stacked = np.hstack([M_sensor_x, M_sensor_y, M_sensor_z])

        theta_all, _, _, sv = np.linalg.lstsq(Phi_stacked, M_stacked.T, rcond=None)
        Theta = theta_all.T

        Hx_pred = Theta @ Phi_x.T
        Hy_pred = Theta @ Phi_y.T
        Hz_pred = Theta @ Phi_z.T
        sensor_pred = np.sqrt(Hx_pred**2 + Hy_pred**2 + Hz_pred**2)

        cond_scalar = float(sv[0]/sv[-1]) if sv[-1] > 1e-15 else float("inf")
        cond = np.full(len(scen_order), cond_scalar)
        n_used_col = np.full(len(scen_order), np.nan)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    rmse_sensor = np.sqrt(np.mean((sensor_pred - M_sensor)**2, axis=1))
    Hxt = Theta @ Phi_xt.T; Hyt = Theta @ Phi_yt.T; Hzt = Theta @ Phi_zt.T
    tube_pred = np.sqrt(Hxt**2 + Hyt**2 + Hzt**2)
    rmse_tube = np.sqrt(np.mean((tube_pred - M_tube)**2, axis=1))

    meta = (df.groupby("scenario_id")[["coil_density","slice_id","z"]]
              .first().reindex(scen_order))
    records = {"scenario_id": scen_order,
               "coil_density": meta["coil_density"].values,
               "slice_id": meta["slice_id"].values,
               "z": meta["z"].values}
    for j, name in enumerate(coeff_names):
        records[name] = Theta[:, j]
    records["rmse_sensor"]       = rmse_sensor
    records["rmse_tube"]         = rmse_tube
    records["condition_number"]  = cond
    records["n_iterations_used"] = n_used_col
    return pd.DataFrame(records)


def fit_all_scenarios_loop(df, sensor_xy=None, n_max=DEFAULT_N_MAX,
                           alpha=DEFAULT_ALPHA, mode="magnitude",
                           r0=R0_DEFAULT, n_iter=DEFAULT_N_ITER, tol=DEFAULT_TOL):
    mode = _resolve_mode(df, mode)
    coeff_names_ = coefficient_names(n_max)
    sz = df[df["region"] == "sensor_zone"]
    scenarios = sorted(df["scenario_id"].unique())
    if sensor_xy is None:
        ref = sz[sz["scenario_id"] == scenarios[0]][["x","y"]].drop_duplicates()
        sensor_xy = list(zip(ref["x"].round(6), ref["y"].round(6)))
    sensor_xy_set = set((round(x,6), round(y,6)) for x,y in sensor_xy)
    tube_df = df[df["region"] == "tube_interior"]
    records = []

    for sid in scenarios:
        sz_s = sz[sz["scenario_id"] == sid].copy()
        sz_s["x"] = sz_s["x"].round(6); sz_s["y"] = sz_s["y"].round(6)
        mask   = sz_s.apply(lambda r: (r["x"],r["y"]) in sensor_xy_set, axis=1)
        sz_sub = sz_s[mask].sort_values(["x","y"])
        if sz_sub.empty: continue

        s_xy = list(zip(sz_sub["x"], sz_sub["y"]))
        H_mag_true = sz_sub["field_value"].values

        if mode == "magnitude":
            theta, _, diag = fit_slice_magnitude(s_xy, H_mag_true, n_max, alpha,
                                                 n_iter=n_iter, tol=tol, r0=r0,
                                                 return_diagnostics=True)
        elif mode == "vector":
            Hx = sz_sub["field_Hx"].values
            Hy = sz_sub["field_Hy"].values
            Hz = sz_sub["field_Hz"].values

            Phi_x, Phi_y, Phi_z = _precompute_point_basis(s_xy, n_max, r0=r0)
            Phi_stacked = np.vstack([Phi_x, Phi_y, Phi_z])
            M_stacked = np.concatenate([Hx, Hy, Hz])

            theta, _, _, sv = np.linalg.lstsq(Phi_stacked, M_stacked, rcond=None)

            Hx_pred = Phi_x @ theta
            Hy_pred = Phi_y @ theta
            Hz_pred = Phi_z @ theta
            H_pred_mag = np.sqrt(Hx_pred**2 + Hy_pred**2 + Hz_pred**2)

            cond_scalar = float(sv[0]/sv[-1]) if sv[-1] > 1e-15 else float("inf")
            rmse_sensor = float(np.sqrt(np.mean((H_pred_mag - H_mag_true)**2)))
            diag = {"rmse_sensor": rmse_sensor, "condition_number": cond_scalar, "n_iterations_used": np.nan}
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        tube_s = tube_df[tube_df["scenario_id"] == sid].sort_values(["x","y"])
        tube_pts = list(zip(tube_s["x"], tube_s["y"]))
        tube_mag_true = tube_s["field_value"].values
        rmse_tube = (float(np.sqrt(np.mean(
            (reconstruct_magnitude(theta, tube_pts, n_max, r0=r0) - tube_mag_true)**2)))
            if len(tube_pts) > 0 else np.nan)

        row = {"scenario_id": sid,
               "coil_density": df[df["scenario_id"]==sid]["coil_density"].iloc[0],
               "slice_id": df[df["scenario_id"]==sid]["slice_id"].iloc[0],
               "z": df[df["scenario_id"]==sid]["z"].iloc[0]}
        for name, val in zip(coeff_names_, theta): row[name] = val
        row["rmse_sensor"] = diag["rmse_sensor"]
        row["rmse_tube"]   = rmse_tube
        row["condition_number"] = diag["condition_number"]
        row["n_iterations_used"] = diag["n_iterations_used"]
        records.append(row)

    return pd.DataFrame(records)


def fit_all_scenarios_adaptive_order(df, sensor_xy, n_max_max, r0=R0_DEFAULT,
                                     oversampling_threshold=2.0, verbose=False):
    """
    For each scenario, use greedy incremental fitting to determine the
    highest order the sensor count actually supports at good conditioning,
    rather than forcing every scenario to the same fixed n_max.
    """
    sz = df[df["region"] == "sensor_zone"]
    scenarios = sorted(df["scenario_id"].unique())
    records = []
    for sid in scenarios:
        sub = sz[sz["scenario_id"] == sid]
        sub = sub[sub.apply(lambda r: (round(r["x"],6), round(r["y"],6))
                            in set((round(x,6), round(y,6)) for x,y in sensor_xy), axis=1)]
        if sub.empty:
            continue
        theta, history = fit_incremental_order(
            sensor_xy, sub["field_Hx"].values, sub["field_Hy"].values, sub["field_Hz"].values,
            n_max_max, r0=r0, oversampling_threshold=oversampling_threshold, verbose=verbose)
        n_max_reached = len(history) - 1  # C0 and Hz don't count as "order", first n= group does
        records.append({"scenario_id": sid, "coil_density": df[df.scenario_id==sid]["coil_density"].iloc[0],
                        "slice_id": df[df.scenario_id==sid]["slice_id"].iloc[0],
                        "n_max_reached": n_max_reached, **{n: v for n, v in zip(coefficient_names(n_max_max), theta)}})
    return pd.DataFrame(records)


def cross_validate_multipole(df, sensor_xy=None, n_max=DEFAULT_N_MAX,
                             alpha=DEFAULT_ALPHA, r0=R0_DEFAULT, n_folds=10,
                             mode="magnitude"):
    from sklearn.model_selection import KFold
    scenarios = sorted(df["scenario_id"].unique())
    splitter  = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scen_arr  = np.array(scenarios)
    fold_rmses = []
    for _, test_idx in splitter.split(scen_arr):
        test_df = df[df["scenario_id"].isin(set(scen_arr[test_idx]))]
        if test_df.empty: continue
        try:
            fit_df = fit_all_scenarios(test_df, sensor_xy=sensor_xy,
                                       n_max=n_max, alpha=alpha, r0=r0, mode=mode)
        except Exception:
            fit_df = fit_all_scenarios_loop(test_df, sensor_xy=sensor_xy,
                                            n_max=n_max, alpha=alpha, r0=r0, mode=mode)
        if not fit_df.empty:
            fold_rmses.append(float(fit_df["rmse_tube"].mean()))
    return {
        "mean_rmse": float(np.mean(fold_rmses)) if fold_rmses else float("nan"),
        "std_rmse":  float(np.std(fold_rmses))  if fold_rmses else float("nan"),
        "per_fold_rmse": fold_rmses,
    }


if __name__ == "__main__":
    import sys
    from magis_pca_pipeline import load_and_reclassify
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "parsed_fields.csv"
    n_max    = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_MAX
    print(f"Loading: {csv_path}, n_max={n_max}")
    df = load_and_reclassify(csv_path)
    results = fit_all_scenarios(df, n_max=n_max)
    print(f"Fitted {len(results)} scenarios")
    print(f"  Sensor RMSE: {results['rmse_sensor'].mean():.6f} A/m")
    print(f"  Tube RMSE:   {results['rmse_tube'].mean():.6f} A/m")
    print(f"  Cond number: {results['condition_number'].median():.2f} (median)")
