"""
plot_reconstruction_error_heatmap.py
======================================
Spatial heatmap of (reconstructed - true) field magnitude across the tube
interior cross-section, at a specific J/slice, answers "where in the
cross-section does the reconstruction fail."

Unlike plot_gradient_heatmap (which evaluates on an arbitrary fine mesh,
since it only needs theta), this evaluates at the actual simulated
tube-interior (x,y) points for that scenario, so it needs both df (for
true field_value) and fit_df (for the fitted theta), joined the same way
as the rest of the predicted-vs-actual family. Since those points are
scattered rather than a clean grid, rendering uses tricontourf/tripcolor.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from multipole_basis import reconstruct_magnitude, coefficient_names, R0_DEFAULT
from plot_field_quiver import R_TUBE_INNER, select_theta_for_J


def compute_reconstruction_error_grid(df, fit_df, n_max, r0=R0_DEFAULT,
                                      J_target=None, slice_id=None,
                                      theta=None, row=None,
                                      coeff_names=None):
    """
    Select the scenario nearest J_target (optionally restricted to
    slice_id), and compute (x, y, diff, error_pct) at every actual tube-interior grid
    point simulated for that scenario, where diff = predicted - true and
    error_pct = percent error.

    Returns
    -------
    x, y, diff, error_pct : 1-D arrays (raw simulation points, not a regular mesh)
    row : the selected fit_df row (for labeling - actual J, slice, etc.)
    """
    coeff_nms = coeff_names or coefficient_names(n_max)

    actual_J = float(row["coil_density"])
    actual_slice = int(row["slice_id"])

    tube_df = df[df["region"] == "tube_interior"].copy()
    sub = tube_df[(np.isclose(tube_df["coil_density"], actual_J, rtol=1e-6)) &
                  (tube_df["slice_id"] == actual_slice)].sort_values(["x", "y"])

    if sub.empty:
        raise ValueError(
            f"No tube_interior rows found in df for J={actual_J}, slice_id={actual_slice} "
            f"- df and fit_df may not correspond to the same run."
        )

    pts = list(zip(sub["x"], sub["y"]))
    true_vals = sub["field_value"].values
    pred_vals = reconstruct_magnitude(theta, pts, n_max, r0=r0)
    diff = pred_vals - true_vals
    error_pct = abs(pred_vals - true_vals) / true_vals * 100

    return sub["x"].values, sub["y"].values, diff, error_pct, row


def plot_reconstruction_error_heatmap(df, fit_df, n_max, r0=R0_DEFAULT,
                                      J_target=None, slice_id=None,
                                      coeff_names=None, #cmap="RdBu_r",
                                      cmap="plasma",
                                      theta=None, row=None,
                                      dark_background=False,
                                      symmetric_scale=True,
                                      n_contour_levels=20,
                                      out_path="reconstruction_error_heatmap.png",
                                      dpi=300, title=None, figsize=(7, 7)):
    """
    Single-slice reconstruction-error heatmap at the scenario nearest
    J_target (and slice_id, if given).
    """
    x, y, diff, error_pct, row = compute_reconstruction_error_grid(
        df, fit_df, n_max, r0=r0, J_target=J_target, slice_id=slice_id,
        theta=theta, row=row, coeff_names=coeff_names)

    actual_J = float(row["coil_density"])
    actual_slice = int(row["slice_id"])
    if title is None:
        # title = f"Reconstruction Error (Predicted − True) — J={actual_J:.3f} A/in², slice {actual_slice}"
        title = f"Reconstruction Error Percent — J={actual_J:.3f} A/in², slice {actual_slice}"

        
    if symmetric_scale:
        # vmax = np.max(np.abs(diff))
        vmax = np.max(error_pct)
        # vmin = -vmax
        vmin = 0
    else:
        # vmin, vmax = diff.min(), diff.max()
        vmin, vmax = error_pct.min(), error_pct.max()

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"

    with style_ctx:
        fig, ax = plt.subplots(figsize=figsize)
        levels = np.linspace(vmin, vmax, n_contour_levels)
        # tpc = ax.tricontourf(x, y, diff, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
        tpc = ax.tricontourf(x, y, error_pct, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
        cbar = fig.colorbar(tpc, ax=ax, fraction=0.045, pad=0.04)
        # cbar.set_label("Reconstructed − True |H|  (A/m)", fontsize=10.5)
        cbar.set_label("Percent Error |H|  (A/m)", fontsize=10.5)

        ax.scatter(x, y, s=4, color=ring_color, alpha=0.3, zorder=3)  # show actual sim point locations

        circ = Circle((0, 0), R_TUBE_INNER, fill=False, edgecolor=ring_color,
                      linewidth=1.2, alpha=0.8, zorder=4)
        ax.add_patch(circ)

        ax.set_aspect("equal")
        ax.set_xlabel("x (in)", fontsize=11)
        ax.set_ylabel("y (in)", fontsize=11)
        ax.set_title(title, fontsize=12.5, fontweight="bold")

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path


def plot_reconstruction_error_heatmap_by_slice(df, fit_df, n_max, r0=R0_DEFAULT,
                                               J_target=None, slice_ids=(1, 2),
                                               coeff_names=None, #cmap="RdBu_r",
                                               cmap="plasma",
                                               dark_background=False,
                                               n_contour_levels=20,
                                               theta=None,
                                               row=None,
                                               out_path="reconstruction_error_by_slice.png",
                                               dpi=300,
                                               title="Reconstruction Error by Slice",
                                               figsize=(13, 6)):
    """
    Side-by-side reconstruction-error heatmaps for each slice in a z-pair
    at the same target J, sharing one symmetric color scale so the two
    slices' error magnitudes are directly comparable
    """
    coeff_nms = coeff_names or coefficient_names(n_max)

    panel_data = []
    all_diffs = []
    all_error_pcts = []
    for sid in slice_ids:
        x, y, diff, error_pct, row = compute_reconstruction_error_grid(
            df, fit_df, n_max, r0=r0, J_target=J_target, slice_id=sid,
            theta=theta, row=row, coeff_names=coeff_nms)
        # panel_data.append((x, y, diff, row))
        panel_data.append((x, y, error_pct, row))
        all_diffs.append(diff)
        all_error_pcts.append(error_pct)

    all_diffs_concat = np.concatenate(all_diffs)
    all_error_pcts_concat = np.concatenate(all_error_pcts)
    # vmax = np.percentile(np.abs(all_diffs_concat), 99)
    vmax = np.percentile(all_error_pcts_concat, 99)
    # vmin = -vmax
    vmin = 0

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"

    with style_ctx:
        fig, axes = plt.subplots(1, len(slice_ids), figsize=figsize, constrained_layout=True)
        if len(slice_ids) == 1:
            axes = [axes]

        levels = np.linspace(vmin, vmax, n_contour_levels)
        tpc = None
        for ax, (x, y, # diff,
                 error_pct, row) in zip(axes, panel_data):
            actual_J = float(row["coil_density"])
            actual_slice = int(row["slice_id"])
            """
            tpc = ax.tricontourf(x, y, diff, levels=levels, cmap=cmap,
                                 vmin=vmin, vmax=vmax, extend="both")
            """
            tpc = ax.tricontourf(x, y, error_pct, levels=levels, cmap=cmap,
                                 vmin=vmin, vmax=vmax, extend="both")
            ax.scatter(x, y, s=4, color=ring_color, alpha=0.3, zorder=3)
            circ = Circle((0, 0), R_TUBE_INNER, fill=False, edgecolor=ring_color,
                          linewidth=1.2, alpha=0.8, zorder=4)
            ax.add_patch(circ)
            ax.set_aspect("equal")
            ax.set_xlabel("x (in)", fontsize=10.5)
            ax.set_ylabel("y (in)", fontsize=10.5)
            ax.set_title(f"Slice {actual_slice}  (J={actual_J:.3f} A/in²)",
                         fontsize=11.5, fontweight="bold")

        cbar = fig.colorbar(tpc, ax=axes, fraction=0.03, pad=0.02, shrink=0.9)
        # cbar.set_label("Reconstructed − True |H|  (A/m)", fontsize=10.5)
        cbar.set_label("Percent Error |H|  (A/m)", fontsize=10.5)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path
