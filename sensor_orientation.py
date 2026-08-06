"""
sensor_orientation.py
=====================
Finds optimal 3-axis magnetometer orientations for a given set of sensor
positions using the Fisher information framework.

Theory
------
For a sensor array with design matrix A (n_measurements x n_coeffs), and
sensor noise covariance sigma^2 * I, the Fisher information matrix is:

    F = (1/sigma^2) * A^T A

The Cramer-Rao bound says the covariance of any unbiased estimator of theta
satisfies Cov(theta) >= F^{-1}. Good sensor orientations maximise information:

  D-optimality : maximise det(F)       — minimises volume of uncertainty ellipsoid
  A-optimality : minimise trace(F^{-1})— minimises mean variance across coefficients
  E-optimality : maximise lambda_min(F)— minimises worst-case coefficient uncertainty

For a 3-axis magnetometer at position (x, y), we can orient it along any
three orthogonal directions. The orientation is parameterised by:
  - theta_deg : polar angle from z-axis (0=axial, 90=transverse)
  - phi_deg   : azimuthal angle from x-axis in transverse plane

We search over a grid of (theta, phi) for each sensor and find the orientation
that maximises D-optimality of the joint design matrix.
"""

import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

from multipole_basis import (
    build_design_matrix, coefficient_names,
    unit_vector, sensor_basis_row, R0_DEFAULT,
)

MU0 = 4 * np.pi * 1e-7


# ── Fisher information ────────────────────────────────────────────────────────

def fisher_matrix(A: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """F = (1/sigma^2) * A^T A"""
    return (A.T @ A) / (sigma ** 2)


def d_optimality(F: np.ndarray) -> float:
    """log det(F) — maximise for D-optimality. Log used for numerical stability."""
    sign, logdet = np.linalg.slogdet(F)
    return logdet if sign > 0 else -np.inf


def a_optimality(F: np.ndarray) -> float:
    """Negative trace(F^{-1}) — maximise (i.e. minimise trace of Cramer-Rao bound)."""
    try:
        return -float(np.trace(np.linalg.inv(F)))
    except np.linalg.LinAlgError:
        return -np.inf


def e_optimality(F: np.ndarray) -> float:
    """Minimum eigenvalue of F — maximise for E-optimality."""
    eigs = np.linalg.eigvalsh(F)
    return float(eigs[0])   # smallest eigenvalue


def condition_number(A: np.ndarray) -> float:
    """Condition number of design matrix A."""
    svs = np.linalg.svd(A, compute_uv=False)
    return svs[0] / svs[-1] if svs[-1] > 1e-15 else np.inf


# ── Orientation parameterisation ──────────────────────────────────────────────

def three_axis_orientations(theta_deg: float,
                            phi_deg: float) -> np.ndarray:
    """
    Build a (3,3) orientation matrix for a 3-axis magnetometer whose
    primary axis points in direction (theta_deg, phi_deg).

    The three measurement axes are:
      axis 1: u1 = unit_vector(theta_deg, phi_deg)           — primary
      axis 2: u2 = normalised(u1 x z_hat) or (u1 x x_hat)  — transverse 1
      axis 3: u3 = u1 x u2                                   — transverse 2

    This gives a right-handed orthonormal frame aligned to the primary axis.
    When theta=90, phi=0 the primary axis is x̂ and the frame is (x̂, ŷ, ẑ).
    """
    u1 = unit_vector(theta_deg, phi_deg)

    # Choose a reference to cross with to get u2
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u1, ref)) > 0.9:   # u1 is nearly axial
        ref = np.array([1.0, 0.0, 0.0])

    u2 = np.cross(u1, ref)
    u2 /= np.linalg.norm(u2)
    u3  = np.cross(u1, u2)
    u3 /= np.linalg.norm(u3)

    return np.vstack([u1, u2, u3])   # (3,3)


def radial_orientation(x: float, y: float) -> np.ndarray:
    """Sensor oriented with primary axis pointing radially outward from beam axis."""
    r    = np.sqrt(x*x + y*y)
    phi  = np.degrees(np.arctan2(y, x))
    return three_axis_orientations(90.0, phi)   # transverse, radially outward


def tangential_orientation(x: float, y: float) -> np.ndarray:
    """Sensor oriented with primary axis tangent to the sensor ring."""
    phi = np.degrees(np.arctan2(y, x)) + 90.0
    return three_axis_orientations(90.0, phi)


def axial_orientation() -> np.ndarray:
    """Sensor oriented with primary axis along the beam axis (z)."""
    return three_axis_orientations(0.0, 0.0)


# ── Grid search over orientations ─────────────────────────────────────────────

def search_orientations(sensor_positions: list[tuple],
                        n_max: int = 3,
                        theta_steps: int = 7,
                        phi_steps: int = 12,
                        criterion: str = "D",
                        noise_sigma_T: float = 1e-9,
                        verbose: bool = True,
                        r0: float = None) -> pd.DataFrame:
    """
    Grid search over the orientation of each sensor to maximise the chosen
    optimality criterion for the joint design matrix.

    For each sensor independently, we test `theta_steps x phi_steps` orientations
    of its primary axis (the other two axes are derived orthogonally). We find
    the combination across all sensors that maximises the criterion.

    For n_sensors=2-3 with moderate grid resolution this is fast because we
    optimise one sensor at a time holding others fixed (coordinate descent).

    Parameters
    ----------
    sensor_positions : list of (x, y) tuples
    n_max            : multipole order
    theta_steps      : number of polar angle samples in [0, 180]
    phi_steps        : number of azimuthal angle samples in [0, 360)
    criterion        : 'D' (D-optimality), 'A' (A-optimality), 'E' (E-optimality)
    noise_sigma_T    : sensor noise in Tesla (converted to A/m via mu0)
    verbose          : print progress

    Returns
    -------
    DataFrame with one row per sensor, columns:
        sensor_x, sensor_y, r, phi_position,
        opt_theta_deg, opt_phi_deg,
        u1_x, u1_y, u1_z,  (primary axis)
        u2_x, u2_y, u2_z,  (secondary)
        u3_x, u3_y, u3_z,  (tertiary)
        criterion_value, condition_number,
        [sensitivity to each multipole coefficient]
    """
    sigma_H    = noise_sigma_T / MU0
    n_sensors  = len(sensor_positions)
    n_coeffs   = 2 * n_max + 2
    coeff_nms  = coefficient_names(n_max)

    criterion_fn = {"D": d_optimality, "A": a_optimality, "E": e_optimality}[criterion]

    thetas = np.linspace(0,   180, theta_steps, endpoint=True)
    phis   = np.linspace(0,   360, phi_steps,   endpoint=False)

    # Initialise with radial orientation
    current_orients = [radial_orientation(x, y) for x,y in sensor_positions]

    if verbose:
        print(f"Orientation search: {n_sensors} sensors, "
              f"{theta_steps}x{phi_steps} grid, criterion={criterion}-optimality")
        print(f"Total grid evaluations per sensor: {theta_steps*phi_steps}")

    # Coordinate descent: optimise one sensor at a time, iterate twice
    for outer in range(2):
        for s_idx, (x, y) in enumerate(sensor_positions):
            best_val    = -np.inf
            best_orient = current_orients[s_idx]

            for th, ph in product(thetas, phis):
                orient = three_axis_orientations(th, ph)
                trial  = list(current_orients)
                trial[s_idx] = orient

                A = build_design_matrix(sensor_positions, trial, n_max, r0=r0 or R0_DEFAULT)
                F = fisher_matrix(A, sigma=sigma_H)
                val = criterion_fn(F)

                if val > best_val:
                    best_val    = val
                    best_orient = orient

            current_orients[s_idx] = best_orient
            if verbose:
                print(f"  [pass {outer+1}] sensor {s_idx+1} at "
                      f"({x:.1f},{y:.1f}): best {criterion}={best_val:.4f}")

    # Build final design matrix and extract per-sensor metrics
    A_final = build_design_matrix(sensor_positions, current_orients, n_max, r0=r0 or R0_DEFAULT)
    F_final = fisher_matrix(A_final, sigma=sigma_H)
    cond    = condition_number(A_final)

    # Per-coefficient uncertainty: sqrt of diagonal of F^{-1}
    try:
        F_inv  = np.linalg.inv(F_final)
        uncert = np.sqrt(np.abs(np.diag(F_inv)))
    except np.linalg.LinAlgError:
        uncert = np.full(n_coeffs, np.inf)

    records = []
    for s_idx, ((x, y), orient) in enumerate(
            zip(sensor_positions, current_orients)):
        r   = np.sqrt(x*x + y*y)
        phi_pos = np.degrees(np.arctan2(y, x)) % 360

        # Sensitivity: norm of each coefficient's contribution to this sensor's rows
        # Rows for this sensor are at indices s_idx*3 : s_idx*3+3
        A_s = A_final[s_idx*3 : s_idx*3+3, :]   # (3, n_coeffs)
        sensitivity = np.linalg.norm(A_s, axis=0)   # per-coefficient sensitivity

        rec = {
            "sensor_idx"     : s_idx + 1,
            "sensor_x"       : x,
            "sensor_y"       : y,
            "r_inches"       : r,
            "phi_position_deg": phi_pos,
            "u1_x": orient[0,0], "u1_y": orient[0,1], "u1_z": orient[0,2],
            "u2_x": orient[1,0], "u2_y": orient[1,1], "u2_z": orient[1,2],
            "u3_x": orient[2,0], "u3_y": orient[2,1], "u3_z": orient[2,2],
            "condition_number": cond,
            f"{criterion}_optimality": d_optimality(F_final),
        }
        for name, sens, unc in zip(coeff_nms, sensitivity, uncert):
            rec[f"sens_{name}"]   = sens
            rec[f"uncert_{name}"] = unc * sigma_H   # back to A/m units

        records.append(rec)

    return pd.DataFrame(records)


# ── Named orientation presets ─────────────────────────────────────────────────

def compare_named_orientations(sensor_positions: list[tuple],
                               n_max: int = 3,
                               noise_sigma_T: float = 1e-9,
                               r0: float = None) -> pd.DataFrame:
    """
    Compare four physically motivated orientation strategies for a given
    sensor layout, without any grid search.

    Strategies:
      radial    : primary axis points away from beam axis
      tangential: primary axis tangent to sensor ring
      axial     : primary axis along z (beam direction)
      cartesian : aligned to lab x, y, z axes (no rotation)

    Returns DataFrame with one row per strategy and key metrics.
    """
    sigma_H   = noise_sigma_T / MU0
    n_coeffs  = 2 * n_max + 2
    coeff_nms = coefficient_names(n_max)

    strategies = {
        "radial"    : [radial_orientation(x,y)    for x,y in sensor_positions],
        "tangential": [tangential_orientation(x,y) for x,y in sensor_positions],
        "axial"     : [axial_orientation()         for _ in sensor_positions],
        "cartesian" : [None                        for _ in sensor_positions],
    }

    records = []
    for name, orients in strategies.items():
        A   = build_design_matrix(sensor_positions, orients, n_max, r0=r0 or R0_DEFAULT)
        F   = fisher_matrix(A, sigma=sigma_H)
        cond = condition_number(A)

        try:
            F_inv  = np.linalg.inv(F)
            uncert = np.sqrt(np.abs(np.diag(F_inv))) * sigma_H
        except np.linalg.LinAlgError:
            uncert = np.full(n_coeffs, np.inf)

        rec = {
            "orientation_strategy": name,
            "D_optimality"        : d_optimality(F),
            "A_optimality"        : a_optimality(F),
            "E_optimality"        : e_optimality(F),
            "condition_number"    : cond,
        }
        for coeff, unc in zip(coeff_nms, uncert):
            rec[f"uncert_{coeff}"] = unc

        records.append(rec)

    return pd.DataFrame(records).sort_values("D_optimality", ascending=False)
