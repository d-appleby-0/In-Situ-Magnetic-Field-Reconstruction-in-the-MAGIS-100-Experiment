"""
recursive_order_refinement.py
================================
greedy, residual-driven incremental multipole order fitting
(matching-pursuit style) for vector-mode data (direct Hx/Hy/Hz
measurements). Rather than jointly fitting all coefficients up to a
pre-chosen n_max at once, this fits one order group at a time,
C0, then Hz, then (C1,S1), then (C2,S2), etc., each time fitting only
the new group's basis columns to whatever residual is left after every
previously-fit group.

Greedy stagewise fitting is only equivalent to a joint fit when different
orders' basis columns are orthogonal under specific sensor sampling.
Always run compare_greedy_vs_joint() on the actual sensor layout before
trusting a greedy fit's coefficients for anything beyond diagnosing which
order stopped reducing residual.
"""

import numpy as np

from multipole_basis import phi_x_row, phi_y_row, phi_z_row, R0_DEFAULT


def _order_groups(n_max_max):
    """
    Order in which coefficient groups are introduced - C0 first (simplest),
    then Hz (also zero-order/simple, separate physical direction), then
    each transverse order n=1..n_max_max as a (Cn, Sn) pair.
    """
    groups = [("C0", [0]), ("Hz", [2 * n_max_max + 1])]
    for n in range(1, n_max_max + 1):
        groups.append((f"n={n}", [2 * n - 1, 2 * n]))
    return groups


def _full_basis(sensor_xy, n_max_max, r0):
    Phi_x = np.vstack([phi_x_row(x, y, n_max_max, r0=r0) for x, y in sensor_xy])
    Phi_y = np.vstack([phi_y_row(x, y, n_max_max, r0=r0) for x, y in sensor_xy])
    Phi_z = np.vstack([phi_z_row(n_max_max) for _ in sensor_xy])
    return Phi_x, Phi_y, Phi_z


def fit_incremental_order(sensor_xy, Hx_true, Hy_true, Hz_true, n_max_max,
                          r0=R0_DEFAULT, oversampling_threshold=2.0,
                          axes_per_sensor=3, verbose=True):
    """
    Greedy residual-driven order-by-order fit. Stops automatically before
    the oversampling ratio (n_measurements / n_coeffs_so_far) would drop
    below oversampling_threshold, rather than fitting an order the sensor
    count can't support.

    Returns
    -------
    theta   : coefficient vector (zeros for any never-reached groups)
    history : list of dicts, one per fitted group, with oversampling ratio
              and residual RMSE after that group was added, useful for
              seeing exactly which order stopped being worth adding.
    """
    n_sensors = len(sensor_xy)
    n_meas = n_sensors * axes_per_sensor
    M_stacked = np.concatenate([Hx_true, Hy_true, Hz_true])

    theta = np.zeros(2 * n_max_max + 2)
    groups = _order_groups(n_max_max)
    n_coeffs_used = 0
    history = []

    Phi_x_full, Phi_y_full, Phi_z_full = _full_basis(sensor_xy, n_max_max, r0)

    for label, idx in groups:
        n_coeffs_used += len(idx)
        oversampling = n_meas / n_coeffs_used
        if oversampling < oversampling_threshold:
            if verbose:
                print(f"  Stopping before group {label}: oversampling would drop to "
                      f"{oversampling:.2f}x (< {oversampling_threshold}x threshold) "
                      f"with {n_meas} measurements / {n_coeffs_used} total coeffs.")
            n_coeffs_used -= len(idx)
            break

        pred_so_far = np.concatenate([Phi_x_full @ theta, Phi_y_full @ theta, Phi_z_full @ theta])
        residual = M_stacked - pred_so_far

        A_group = np.vstack([Phi_x_full[:, idx], Phi_y_full[:, idx], Phi_z_full[:, idx]])
        coeffs_group, *_ = np.linalg.lstsq(A_group, residual, rcond=None)
        theta[idx] = coeffs_group

        pred_now = np.concatenate([Phi_x_full @ theta, Phi_y_full @ theta, Phi_z_full @ theta])
        rmse = float(np.sqrt(np.mean((M_stacked - pred_now) ** 2)))
        history.append({"group": label, "oversampling": oversampling, "rmse_after": rmse})
        if verbose:
            print(f"  Fit group {label:<6} ({len(idx)} coeff{'s' if len(idx) > 1 else ''})  "
                  f"oversampling={oversampling:.2f}x  residual RMSE={rmse:.6e}")

    return theta, history


def fit_joint(sensor_xy, Hx_true, Hy_true, Hz_true, n_max_max, r0=R0_DEFAULT):
    """
    Standard joint (all coefficients at once) vector-mode fit, for
    comparison against the greedy incremental result.
    """
    Phi_x, Phi_y, Phi_z = _full_basis(sensor_xy, n_max_max, r0)
    Phi_stacked = np.vstack([Phi_x, Phi_y, Phi_z])
    M_stacked = np.concatenate([Hx_true, Hy_true, Hz_true])
    theta, *_ = np.linalg.lstsq(Phi_stacked, M_stacked, rcond=None)
    return theta


def compare_greedy_vs_joint(sensor_xy, Hx_true, Hy_true, Hz_true, n_max_max,
                            r0=R0_DEFAULT, oversampling_threshold=2.0, verbose=True):
    """
    Fit both ways and report the discrepancy - quantifies exactly how
    much bias the greedy approach introduces for this sensor layout,
    rather than assuming it's negligible.
    """
    theta_greedy, history = fit_incremental_order(
        sensor_xy, Hx_true, Hy_true, Hz_true, n_max_max, r0=r0,
        oversampling_threshold=oversampling_threshold, verbose=verbose)
    theta_joint = fit_joint(sensor_xy, Hx_true, Hy_true, Hz_true, n_max_max, r0=r0)

    diff = theta_greedy - theta_joint
    rel_diff = np.linalg.norm(diff) / max(np.linalg.norm(theta_joint), 1e-15)

    if verbose:
        print(f"\n  Greedy vs. joint fit comparison:")
        print(f"    ||theta_greedy - theta_joint|| / ||theta_joint|| = {rel_diff:.4f}")
        if rel_diff > 0.1:
            print(f"    X Large discrepancy (>10%) - this sensor layout's basis columns are "
                  f"not close to orthogonal across orders. Greedy results are not reliable here.")

    return theta_greedy, theta_joint, rel_diff, history
