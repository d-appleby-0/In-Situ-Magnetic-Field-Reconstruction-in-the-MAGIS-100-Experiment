"""
plot_field_quiver.py
======================
utilizes quiver to output a vector field map and chosen sensor configuration,
minimizing Hz into a component of the magnitude corresponding to each vector
in the transverse

Found in the plot block immediately after the first END OF REPORT in STEP 12
of generate_full_report in multipole_report.py

select_theta_for_J is a utility script implemented in STEP 6 of the same
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon

from multipole_basis import reconstruct_field, R0_DEFAULT

R_TUBE_INNER = 2.75
R_TUBE_OUTER = 3.0
R_SHIELD = 7.0
SHIELD_N_SIDES = 8


def _in_regular_polygon(x, y, apothem, n_sides=SHIELD_N_SIDES, orientation=0.0):
    x = np.asarray(x); y = np.asarray(y)
    inside = np.ones_like(x, dtype=bool)
    for k in range(n_sides):
        theta_k = orientation + 2*np.pi*k/n_sides
        proj = x*np.cos(theta_k) + y*np.sin(theta_k)
        inside &= (proj <= apothem + 1e-9)
    return inside


def plot_vector_field_quiver(theta, n_max, r0=R0_DEFAULT,
                             out_path="field_quiver.png",
                             title="Reconstructed Transverse Field",
                             grid_n=25, extend_to_sensor_zone=True,
                             sensor_positions=None,
                             cmap="plasma", dark_background=False,
                             shield_orientation=0.0,
                             dpi=300, figsize=(7, 7)):
    shield_circumradius = R_SHIELD / np.cos(np.pi / SHIELD_N_SIDES)
    r_max = shield_circumradius if extend_to_sensor_zone else R_TUBE_INNER
    xs = np.linspace(-r_max, r_max, grid_n)
    ys = np.linspace(-r_max, r_max, grid_n)
    X, Y = np.meshgrid(xs, ys)
    pts = list(zip(X.ravel(), Y.ravel()))

    H = reconstruct_field(theta, pts, n_max, r0=r0)
    Hx, Hy, Hz = H[:, 0], H[:, 1], H[:, 2]
    mag = np.sqrt(Hx**2 + Hy**2 + Hz**2)

    R = np.sqrt(X.ravel()**2 + Y.ravel()**2)
    if extend_to_sensor_zone:
        inside_shield = _in_regular_polygon(X.ravel(), Y.ravel(), R_SHIELD,
                                            orientation=shield_orientation)
        keep = ((R <= R_TUBE_INNER) | ((R > R_TUBE_OUTER) & inside_shield))
    else:
        keep = (R <= R_TUBE_INNER)

    Xk, Yk = X.ravel()[keep], Y.ravel()[keep]
    Hxk, Hyk = Hx[keep], Hy[keep]
    magk = mag[keep]

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    with style_ctx:
        fig, ax = plt.subplots(figsize=figsize)
        q = ax.quiver(Xk, Yk, Hxk, Hyk, magk, cmap=cmap,
                      angles="xy", scale_units="xy", scale=None, width=0.004, pivot="mid")
        cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|H|  (A/m)", fontsize=11)

        ring_color = "white" if dark_background else "black"
        for radius, ls in [(R_TUBE_INNER, "-"), (R_TUBE_OUTER, "--")]:
            circ = Circle((0, 0), radius, fill=False, edgecolor=ring_color,
                          linestyle=ls, linewidth=1.1, alpha=0.6)
            ax.add_patch(circ)
        if extend_to_sensor_zone:
            octagon = RegularPolygon((0, 0), numVertices=SHIELD_N_SIDES,
                                     radius=shield_circumradius,
                                     orientation=shield_orientation + np.pi/SHIELD_N_SIDES,
                                     fill=False, edgecolor=ring_color,
                                     linestyle=":", linewidth=1.1, alpha=0.6)
            ax.add_patch(octagon)

        if sensor_positions:
            sx = [p[0] for p in sensor_positions]
            sy = [p[1] for p in sensor_positions]
            ax.scatter(sx, sy, marker="*", s=220, color="red",
                      edgecolor="white", linewidth=0.8, zorder=5, label="sensor location")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.3)

        ax.set_xlim(-r_max*1.05, r_max*1.05)
        ax.set_ylim(-r_max*1.05, r_max*1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("x (in)", fontsize=11)
        ax.set_ylabel("y (in)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path


def select_theta_for_J(fit_df, coeff_names, J_target=None, slice_id=None,
                       tolerance_frac=0.05, use_smoothed=True):
    """
    Find the fit_df row whose coil_density is closest to J_target, and
    return (theta, row) so callers know exactly which scenario was used.

    This replaces the previous "middle index" selection
    (fit_df.iloc[len(fit_df)//2]), which picked a scenario purely by its
    position in the sorted table rather than by an actual physical J value
    and which, for any pair with an even row count, always landed on
    slice 2 by construction (never slice 1), an unintentional bias

    Parameters
    ----------
    fit_df     : per-scenario coefficient table (from fit_all_scenarios())
    coeff_names : coefficient_names(n_max) for that fit
    J_target   : the J value you actually want to look at. If None,
                 defaults to the true median J present in fit_df.
    slice_id   : optional - restrict the search to one slice. If None,
                 searches across whatever slices are present in fit_df and
                 picks whichever (J, slice) pair is numerically closest.
    tolerance_frac : print a note if the nearest available J differs from
                 J_target by more than this fraction (J grid is
                 discrete (e.g. geomspace-sampled) so an exact match
                 usually won't exist).

    Returns
    -------
    theta : coefficient vector for the selected row
    row   : the full fit_df row (gives you row["coil_density"],
            row["slice_id"], row["scenario_id"], etc. for labeling)
    """
    candidates = fit_df
    if slice_id is not None:
        candidates = candidates[candidates["slice_id"] == slice_id]
        if candidates.empty:
            raise ValueError(f"No rows with slice_id={slice_id} in fit_df. "
                             f"Available slice_id values: "
                             f"{sorted(fit_df['slice_id'].unique())}")

    if J_target is None:
        J_target = float(candidates["coil_density"].median())

    idx = (candidates["coil_density"] - J_target).abs().idxmin()
    row = candidates.loc[idx]
    actual_J = float(row["coil_density"])

    if J_target > 0 and abs(actual_J - J_target) > tolerance_frac * J_target:
        print(f"  NOTE: requested J={J_target:.4f}, nearest available is "
              f"J={actual_J:.4f} (slice_id={int(row['slice_id'])}) - "
              f"no exact match in the sampled J grid.")

    if use_smoothed:
        smoothed_cols = [f"{c}_smoothed" for c in coeff_names]
        if all(c in row.index for c in smoothed_cols):
            theta = row[smoothed_cols].values.astype(float)
        else:
            print(f"  NOTE: use_smoothed=True but smoothed columns not found in fit_df - "
                  f"falling back to raw fitted coefficients. Run "
                  f"coefficient_smoothing.smooth_coefficients_vs_J() first.")
            theta = row[coeff_names].values.astype(float)
    else:
        theta = row[coeff_names].values.astype(float)

    return theta, row


def plot_vector_field_quiver_for_J(fit_df, coeff_names, n_max, r0=R0_DEFAULT,
                                   J_target=None, slice_id=None,
                                   sensor_positions=None,
                                   out_path="field_quiver.png",
                                   title=None, **kwargs):
    """
    Convenience wrapper: pick the scenario nearest a specific J (and,
    optionally, a specific slice) and plot its reconstructed field directly
    without needing to run the full generate_full_report pipeline just
    to look at one J value.

    Example
    -------
        plot_vector_field_quiver_for_J(
            fit_df, coefficient_names(n_max), n_max, r0=2.75,
            J_target=50, slice_id=1,
            sensor_positions=sensor_positions,
            out_path="field_at_J50.pdf",
        )
    """
    theta, row = select_theta_for_J(fit_df, coeff_names, J_target=J_target,
                                    slice_id=slice_id)
    actual_J = float(row["coil_density"])
    actual_slice = int(row["slice_id"])

    if title is None:
        title = f"MAGIS-100 Tube Field — J={actual_J:.3f} A/in², slice {actual_slice}"

    return plot_vector_field_quiver(
        theta, n_max, r0=r0, out_path=out_path, title=title,
        sensor_positions=sensor_positions, **kwargs)
