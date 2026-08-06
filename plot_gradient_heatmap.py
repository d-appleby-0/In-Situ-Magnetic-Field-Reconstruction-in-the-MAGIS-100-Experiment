"""
plot_gradient_heatmap.py
==========================
plot the gradient over the dimensions of the vacuume tube as well as the
correlated 1σ uncertainty from the covariance of the theta coefficient vector

Found in STEP 12 of generate_full_report in multipole_report.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.colors import LogNorm, Normalize
from sensor_misalignment import compute_cov_theta_with_misalignment, MAX_SENSOR_MISALIGNMENT_DEG
from multipole_basis import (
    reconstruct_gradient, phi_x_dx_row, phi_x_dy_row, phi_y_dx_row, phi_y_dy_row,
    R0_DEFAULT,
)
from plot_field_quiver import (
    R_TUBE_INNER, R_TUBE_OUTER, R_SHIELD, SHIELD_N_SIDES, _in_regular_polygon,
)

MU0 = 4 * np.pi * 1e-7


def _build_grid_and_mask(grid_n, r_max, extend_to_sensor_zone, shield_orientation):
    xs = np.linspace(-r_max, r_max, grid_n)
    ys = np.linspace(-r_max, r_max, grid_n)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X**2 + Y**2)
    if extend_to_sensor_zone:
        inside_shield = _in_regular_polygon(X, Y, R_SHIELD, orientation=shield_orientation)
        keep = ((R <= R_TUBE_INNER) | ((R > R_TUBE_OUTER) & inside_shield))
    else:
        keep = (R <= R_TUBE_INNER)
    return X, Y, keep


def compute_gradient_magnitude_grid(theta, n_max, r0=R0_DEFAULT, grid_n=150,
                                    extend_to_sensor_zone=False, shield_orientation=0.0):
    r_max = (R_SHIELD / np.cos(np.pi / SHIELD_N_SIDES)) if extend_to_sensor_zone else R_TUBE_INNER
    X, Y, keep = _build_grid_and_mask(grid_n, r_max, extend_to_sensor_zone, shield_orientation)
    pts = list(zip(X.ravel(), Y.ravel()))
    G = reconstruct_gradient(theta, pts, n_max, r0=r0)
    grad_mag = np.sqrt(np.sum(G**2, axis=(1, 2))).reshape(X.shape)
    return X, Y, np.where(keep, grad_mag, np.nan)


def compute_cov_theta(A, n_max, noise_sigma_T=1e-9, rcond=1e-10):
    """
    SVD-truncated pseudoinverse covariance computation, same convention as
    multipole_report.print_uncertainty_budget. Factored out here (rather
    than imported from multipole_report) to avoid a circular import.
    """
    sigma_H = noise_sigma_T / MU0
    n_coeffs = 2 * n_max + 2

    U, s_thin, Vt = np.linalg.svd(A, full_matrices=True)
    s = np.zeros(n_coeffs); s[:len(s_thin)] = s_thin
    s_max = s[0] if len(s) else 0.0
    well = s > rcond * s_max

    s_inv2 = np.zeros_like(s)
    s_inv2[well] = 1.0 / (s[well]**2)
    cov = (Vt.T * s_inv2) @ Vt * sigma_H**2
    return cov


def compute_gradient_uncertainty_grid(cov_theta, n_max, r0=R0_DEFAULT, grid_n=150,
                                      extend_to_sensor_zone=False,
                                      shield_orientation=0.0, theta=None):
    """
    Propagate coefficient covariance through the gradient basis to get
    sigma(||∇H||) at every grid point, via the delta method, vectorized
    across the whole grid via einsum.
    """
    if theta is None:
        raise ValueError("theta is required to linearize the delta-method "
                         "uncertainty propagation around the fitted gradient value.")

    r_max = (R_SHIELD / np.cos(np.pi / SHIELD_N_SIDES)) if extend_to_sensor_zone else R_TUBE_INNER
    X, Y, keep = _build_grid_and_mask(grid_n, r_max, extend_to_sensor_zone, shield_orientation)
    pts = list(zip(X.ravel(), Y.ravel()))

    Phi_xdx = np.vstack([phi_x_dx_row(x, y, n_max, r0=r0) for x, y in pts])
    Phi_xdy = np.vstack([phi_x_dy_row(x, y, n_max, r0=r0) for x, y in pts])
    Phi_ydx = np.vstack([phi_y_dx_row(x, y, n_max, r0=r0) for x, y in pts])
    Phi_ydy = np.vstack([phi_y_dy_row(x, y, n_max, r0=r0) for x, y in pts])
    R_all = np.stack([Phi_xdx, Phi_xdy, Phi_ydx, Phi_ydy], axis=1)

    g_vec_all = np.einsum('pij,j->pi', R_all, theta)
    Cov_g_all = np.einsum('pik,kl,pjl->pij', R_all, cov_theta, R_all)

    norm_g_all = np.linalg.norm(g_vec_all, axis=1)
    safe_norm = np.where(norm_g_all < 1e-15, 1.0, norm_g_all)
    ghat_all = g_vec_all / safe_norm[:, None]

    sigma_sq = np.einsum('pi,pij,pj->p', ghat_all, Cov_g_all, ghat_all)
    trace_over_4 = np.einsum('pii->p', Cov_g_all) / 4.0
    sigma_sq = np.where(norm_g_all < 1e-15, trace_over_4, sigma_sq)
    sigma_norm_all = np.sqrt(np.clip(sigma_sq, 0, None)).reshape(X.shape)

    return X, Y, np.where(keep, sigma_norm_all, np.nan)


def _auto_color_norm(data, color_scale="auto", vmin=None, vmax=None,
                     clip_percentile=(1, 99), dynamic_range_threshold=10):
    """
    Choose a sensible color normalization for a field that may span orders
    of magnitude (common for gradient magnitude: small near the center,
    large near the boundary/higher-order terms). A linear scale over the
    full min-max range compresses almost everything into one end of the
    colormap when the dynamic range is large, making the plot look
    artificially uniform even when the underlying values vary.

    The log-vs-linear decision is based on the true (unclipped) min/max
    spread of the data, not the percentile-clipped display range, which
    is a separate concern (guarding a single outlier pixel from stretching
    the displayed color limits). Using the clipped range for the decision
    itself can under-trigger log scale in exactly the cases it's meant to
    catch, since clipping away the extreme low end can accidentally make
    the remaining spread look artificially modest.
    """
    finite = data[np.isfinite(data)]
    finite_positive = finite[finite > 0]
    if finite_positive.size == 0:
        return Normalize(vmin=0, vmax=1)

    true_lo, true_hi = finite_positive.min(), finite_positive.max()
    true_ratio = true_hi / true_lo if true_lo > 0 else float("inf")

    disp_lo = np.percentile(finite_positive, clip_percentile[0]) if vmin is None else vmin
    disp_hi = np.percentile(finite_positive, clip_percentile[1]) if vmax is None else vmax
    disp_lo = max(disp_lo, 1e-15)

    use_log = (color_scale == "log") or (
        color_scale == "auto" and true_ratio > dynamic_range_threshold
    )

    print(f"  [color scale] true data spread: [{true_lo:.3e}, {true_hi:.3e}] "
          f"({true_ratio:.1f}x)  |  display range ({clip_percentile[0]}-{clip_percentile[1]}%ile): "
          f"[{disp_lo:.3e}, {disp_hi:.3e}]")

    if use_log:
        print(f"  [color scale] -> using LOG normalization (true spread {true_ratio:.1f}x "
              f"exceeds threshold {dynamic_range_threshold}x)")
        return LogNorm(vmin=disp_lo, vmax=disp_hi, clip=True)
    else:
        print(f"  [color scale] -> using linear normalization")
        return Normalize(vmin=disp_lo, vmax=disp_hi, clip=True)


def _draw_geometry(ax, extend_to_sensor_zone, shield_orientation, dark_background):
    ring_color = "white" if dark_background else "black"
    circ = Circle((0, 0), R_TUBE_INNER, fill=False, edgecolor=ring_color, linewidth=1.1, alpha=0.7)
    ax.add_patch(circ)
    if extend_to_sensor_zone:
        circ2 = Circle((0, 0), R_TUBE_OUTER, fill=False, edgecolor=ring_color,
                       linestyle="--", linewidth=0.9, alpha=0.5)
        ax.add_patch(circ2)
        shield_circumradius = R_SHIELD / np.cos(np.pi / SHIELD_N_SIDES)
        octagon = RegularPolygon((0, 0), numVertices=SHIELD_N_SIDES, radius=shield_circumradius,
                                 orientation=shield_orientation + np.pi/SHIELD_N_SIDES,
                                 fill=False, edgecolor=ring_color, linestyle=":", linewidth=0.9, alpha=0.5)
        ax.add_patch(octagon)


def plot_gradient_heatmap(theta, n_max, r0=R0_DEFAULT, A=None,
                          sensor_positions=None, sensor_orientations=None,
                          delta_max_deg=MAX_SENSOR_MISALIGNMENT_DEG,
                          noise_sigma_T=1e-9, rcond=1e-10,
                          include_uncertainty_panel=True,
                          grid_n=150, extend_to_sensor_zone=False,
                          shield_orientation=0.0,
                          cmap="plasma", dark_background=True,
                          color_scale="auto", vmin=None, vmax=None,
                          clip_percentile=(1, 99),
                          out_path="gradient_heatmap.png", dpi=300,
                          title="Transverse Field Gradient Magnitude",
                          figsize=None):
    """
    Parameters
    ----------
    A       : design matrix (sensor_positions + orientations) - required
              if include_uncertainty_panel=True, since the uncertainty
              panel needs the coefficient covariance derived from it.
              Build via multipole_basis.build_design_matrix(...) with the
              same sensor layout/orientations used for the fit.
    noise_sigma_T : sensor noise floor in Tesla, used for the uncertainty
              panel's covariance propagation.
    include_uncertainty_panel : if True (default), renders a second panel
              showing σ(||∇H||) across the same grid. Requires A -
              falls back to single-panel output with a note if
              A is not supplied.
    color_scale : "auto" (default) picks LogNorm automatically if the TRUE
              data spread exceeds ~10x, else linear - applied
              independently to each panel, since the magnitude and
              uncertainty grids can have very different dynamic ranges.
    vmin, vmax  : override the automatic percentile-based color limits
              (applied to both panels if include_uncertainty_panel=True;
              pass separately per-panel by calling with
              include_uncertainty_panel=False twice if you need different
              overrides for each).
    clip_percentile : (low, high) percentiles used as default vmin/vmax
              instead of the true min/max, so a single outlier pixel can't
              stretch the whole color scale by itself.
    """
    X, Y, grad_mag = compute_gradient_magnitude_grid(
        theta, n_max, r0=r0, grid_n=grid_n,
        extend_to_sensor_zone=extend_to_sensor_zone,
        shield_orientation=shield_orientation)

    show_uncertainty = (include_uncertainty_panel and A is not None
                        and sensor_positions is not None and sensor_orientations is not None)
    if include_uncertainty_panel and A is None:
        print("  NOTE: include_uncertainty_panel=True but no design matrix A was "
              "supplied, skipping the uncertainty panel (single-panel output).")

    r_max = (R_SHIELD / np.cos(np.pi / SHIELD_N_SIDES)) if extend_to_sensor_zone else R_TUBE_INNER
    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")

    with style_ctx:
        if show_uncertainty:
            cov_theta, _, _ = compute_cov_theta_with_misalignment(A, sensor_positions, sensor_orientations, theta,
                                                                  n_max, noise_sigma_T=noise_sigma_T,
                                                                  delta_max_deg=delta_max_deg, rcond=rcond)
            _, _, sigma_grid = compute_gradient_uncertainty_grid(
                cov_theta, n_max, r0=r0, grid_n=grid_n,
                extend_to_sensor_zone=extend_to_sensor_zone,
                shield_orientation=shield_orientation, theta=theta)

            figsize = figsize or (12.5, 5.8)
            fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

            norm0 = _auto_color_norm(grad_mag, color_scale=color_scale, vmin=vmin,
                                     vmax=vmax, clip_percentile=clip_percentile)
            mesh0 = axes[0].pcolormesh(X, Y, grad_mag, cmap=cmap, norm=norm0, shading="auto")
            cbar0 = fig.colorbar(mesh0, ax=axes[0], fraction=0.045, pad=0.03)
            cbar0.set_label("||∇H||  (A/m per inch)", fontsize=10.5)
            axes[0].set_title("Gradient magnitude", fontsize=12, fontweight="bold")

            norm1 = _auto_color_norm(sigma_grid, color_scale=color_scale, vmin=vmin,
                                     vmax=vmax, clip_percentile=clip_percentile)
            mesh1 = axes[1].pcolormesh(X, Y, sigma_grid, cmap=cmap, norm=norm1, shading="auto")
            cbar1 = fig.colorbar(mesh1, ax=axes[1], fraction=0.045, pad=0.03)
            cbar1.set_label("σ(||∇H||)  (A/m per inch)", fontsize=10.5)
            axes[1].set_title("Propagated uncertainty", fontsize=12, fontweight="bold")

            for ax in axes:
                _draw_geometry(ax, extend_to_sensor_zone, shield_orientation, dark_background)
                ax.set_xlim(-r_max, r_max); ax.set_ylim(-r_max, r_max)
                ax.set_aspect("equal")
                ax.set_xlabel("x (in)", fontsize=10.5)
                ax.set_ylabel("y (in)", fontsize=10.5)

            fig.suptitle(title, fontsize=14, fontweight="bold")
        else:
            figsize = figsize or (7, 7)
            fig, ax = plt.subplots(figsize=figsize)

            norm = _auto_color_norm(grad_mag, color_scale=color_scale, vmin=vmin,
                                    vmax=vmax, clip_percentile=clip_percentile)
            mesh = ax.pcolormesh(X, Y, grad_mag, cmap=cmap, norm=norm, shading="auto")
            cbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.04)
            cbar.set_label("||∇H||  (A/m per inch)", fontsize=11)

            _draw_geometry(ax, extend_to_sensor_zone, shield_orientation, dark_background)
            ax.set_xlim(-r_max, r_max); ax.set_ylim(-r_max, r_max)
            ax.set_aspect("equal")
            ax.set_xlabel("x (in)", fontsize=11)
            ax.set_ylabel("y (in)", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight="bold")

        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path
