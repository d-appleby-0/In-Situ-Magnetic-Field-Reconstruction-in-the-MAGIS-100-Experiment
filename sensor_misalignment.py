"""
sensor_misalignment.py
========================
analytic estimate of the uncertainty contribution from small,
unknown sensor mounting misalignment (a rigid tilt of a 3-axis
magnetometer package by up to MAX_SENSOR_MISALIGNMENT_DEG, direction
unknown), combined in quadrature with the existing sensor noise floor.

If a sensor's true mounted axis u differs from its nominal axis u0 by a
small rigid rotation of angle delta, then to first order the extra
measurement error is

    Delta_m ~= delta * (u_perp . H)

where u_perp is a unit vector perpendicular to u0 (the unknown tilt
direction) and H is the true field at that sensor location. Since the
tilt direction is unknown, we take the worst case - the full magnitude of
H's component perpendicular to u0,

    H_perp = H - (H . u0) u0
    sigma_misalign = delta_max * |H_perp|

This is not a fixed instrument constant like the sensor noise floor but
rather scales with the field magnitude/direction at that sensor and
scenario, so it must be recomputed per (sensor, scenario) using that
scenario's own theta.

Since each measurement can now have a different total noise sigma_i
(sensor noise combined in quadrature with this scenario/sensor-specific
misalignment term), the Fisher information generalizes from the scalar-
noise case F = A^T A / sigma^2 to the heteroscedastic case F = A^T W A,
W = diag(1/sigma_i^2).
"""

import numpy as np

from multipole_basis import reconstruct_field, R0_DEFAULT
from sensor_orientation import MU0

# Upper bound on mounting misalignment
MAX_SENSOR_MISALIGNMENT_DEG = 1.0


def compute_misalignment_sigma(sensor_positions, sensor_orientations, theta,
                               n_max, r0=R0_DEFAULT,
                               delta_max_deg=MAX_SENSOR_MISALIGNMENT_DEG):
    """
    Per-measurement-row misalignment uncertainty, matching the row order
    build_design_matrix() produces (one row per axis per sensor).

    Parameters
    ----------
    sensor_positions    : [(x,y), ...]
    sensor_orientations : list, one entry per sensor - None (Cartesian
                          x/y/z), a single (3,) unit vector, or a (3,3)
                          orthonormal frame, matching build_design_matrix's
                          convention
    theta               : the fitted coefficient vector for the scenario
                          being evaluated (misalignment error depends on
                          the field at each sensor, not just geometry)
    delta_max_deg       : maximum mounting misalignment, in degrees

    Returns
    -------
    sigma_misalign : 1-D array, one entry per measurement row, in the same
                     field units as sigma_H (A/m)
    """
    delta_max_rad = np.radians(delta_max_deg)
    sigma_rows = []

    for (x, y), orient in zip(sensor_positions, sensor_orientations):
        if orient is None:
            axes = np.eye(3)
        elif np.array(orient).ndim == 1:
            axes = np.array(orient).reshape(1, 3)
        else:
            axes = np.array(orient)

        H_here = reconstruct_field(theta, [(x, y)], n_max, r0=r0)[0]  # (3,)

        for u0 in axes:
            u0 = u0 / np.linalg.norm(u0)
            H_along = np.dot(H_here, u0) * u0
            H_perp = H_here - H_along
            sigma_rows.append(delta_max_rad * np.linalg.norm(H_perp))

    return np.array(sigma_rows)


def build_whitened_design_matrix(A, sigma_per_row):
    """
    Divide each row of A by its own total noise sigma_i, so the
    heteroscedastic Fisher information F = A^T diag(1/sigma_i^2) A becomes
    a plain F = A_tilde^T A_tilde on the whitened matrix, letting the
    SVD-truncated-pseudoinverse covariance code used everywhere else in the
    pipeline apply unchanged
    """
    sigma_per_row = np.asarray(sigma_per_row)
    if np.any(sigma_per_row <= 0):
        raise ValueError("All per-row sigma values must be positive.")
    return A / sigma_per_row[:, None]


def compute_total_sigma_per_row(A, sensor_positions, sensor_orientations, theta,
                                n_max, r0=R0_DEFAULT, noise_sigma_T=1e-9,
                                delta_max_deg=MAX_SENSOR_MISALIGNMENT_DEG):
    """
    Combine the flat sensor noise floor (converted to A/m) with the
    scenario/sensor-specific misalignment uncertainty, in quadrature, per
    measurement row.
    """
    sigma_H = noise_sigma_T / MU0
    sigma_misalign = compute_misalignment_sigma(
        sensor_positions, sensor_orientations, theta, n_max, r0=r0,
        delta_max_deg=delta_max_deg)
    n_rows = A.shape[0]
    sigma_noise_row = np.full(n_rows, sigma_H)
    return np.sqrt(sigma_noise_row**2 + sigma_misalign**2), sigma_misalign, sigma_noise_row


def compute_cov_theta_with_misalignment(A, sensor_positions, sensor_orientations,
                                        theta, n_max, r0=R0_DEFAULT,
                                        noise_sigma_T=1e-9,
                                        delta_max_deg=MAX_SENSOR_MISALIGNMENT_DEG,
                                        rcond=1e-10):
    """
    Coefficient covariance including both sensor noise and misalignment
    uncertainty, via the whitened-design-matrix generalization

    Returns
    -------
    cov_theta       : (n_coeffs, n_coeffs) covariance matrix
    sigma_total_row : per-row total sigma used (for diagnostics)
    sigma_misalign  : per-row misalignment-only contribution (for diagnostics)
    """
    n_coeffs = 2 * n_max + 2
    sigma_total_row, sigma_misalign, sigma_noise_row = compute_total_sigma_per_row(
        A, sensor_positions, sensor_orientations, theta, n_max, r0=r0,
        noise_sigma_T=noise_sigma_T, delta_max_deg=delta_max_deg)

    A_tilde = build_whitened_design_matrix(A, sigma_total_row)

    U, s_thin, Vt = np.linalg.svd(A_tilde, full_matrices=True)
    s = np.zeros(n_coeffs); s[:len(s_thin)] = s_thin
    s_max = s[0] if len(s) else 0.0
    well = s > rcond * s_max

    s_inv2 = np.zeros_like(s)
    s_inv2[well] = 1.0 / (s[well] ** 2)
    cov_theta = (Vt.T * s_inv2) @ Vt   # no extra sigma^2 factor anymore

    return cov_theta, sigma_total_row, sigma_misalign
