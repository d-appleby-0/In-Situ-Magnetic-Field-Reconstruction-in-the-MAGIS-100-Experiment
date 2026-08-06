"""
plot_mode_gallery.py
======================
Two pedagogical visuals for showing "what each multipole order looks like",
built directly from the coefficient_names()/reconstruct_field() basis in
multipole_basis.py - no fitted data required, these are pure illustrations
of the basis functions themselves (set one coefficient to 1, zero the rest).

1. plot_mode_gallery() - a small-multiples grid, one streamplot panel per
   basis coefficient (C0, C1, C2, C3, Hz, S1, S2, S3 for n_max=3), each
   panel colored by that mode's own field magnitude.

2. plot_mode_overlay() - a single shared panel overlaying a handful of
   representative field lines from each order n=0..n_max, color-coded with
   a legend, so the "n-fold" symmetry of each order is directly comparable
   at a glance. (Uses only the canonical Cn=1 term per order - Sn is just
   a rotated copy, and Hz has no transverse streamline to show.)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from multipole_basis import reconstruct_field, coefficient_names, R0_DEFAULT
from plot_field_quiver import R_TUBE_INNER

ORDER_NAMES = {0: "uniform", 1: "dipole", 2: "quadrupole", 3: "sextupole",
              4: "octupole", 5: "decapole", 6: "dodecapole"}


def _one_hot_theta(n_max, index):
    theta = np.zeros(2*n_max + 2)
    theta[index] = 1.0
    return theta


def _field_grid(theta, n_max, r0, r_max, grid_n):
    xs = np.linspace(-r_max, r_max, grid_n)
    ys = np.linspace(-r_max, r_max, grid_n)
    X, Y = np.meshgrid(xs, ys)
    pts = list(zip(X.ravel(), Y.ravel()))
    H = reconstruct_field(theta, pts, n_max, r0=r0)
    Hx = H[:, 0].reshape(X.shape)
    Hy = H[:, 1].reshape(X.shape)
    Hz = H[:, 2].reshape(X.shape)
    return X, Y, Hx, Hy, Hz


def plot_mode_gallery(n_max=3, r0=R0_DEFAULT, r_max=None, grid_n=120,
                      cmap="plasma", dark_background=False,
                      out_path="mode_gallery.png", dpi=300,
                      suptitle="Multipole Basis Function Gallery"):
    """
    Small-multiples gallery: one streamplot panel per basis coefficient,
    colored by that panel's own field magnitude, thus each plot is not
    absolutely comparable to the other.

    Layout is 2 rows x (n_max+1) columns:
        row 1: C0, C1, C2, ..., C(n_max)
        row 2: Hz, S1, S2, ..., S(n_max)
    (C0's row-2 slot holds Hz since the uniform order has no sine partner.)
    """
    r_max = r_max if r_max is not None else 1.3 * R_TUBE_INNER
    coeff_nms = coefficient_names(n_max)
    n_cols = n_max + 1

    # panel spec: (row, col, coeff_index, label)
    panels = [(0, 0, 0, coeff_nms[0])]                    # C0
    panels.append((1, 0, 2*n_max + 1, "Hz(axial)"))       # Hz in C0's sin slot
    for n in range(1, n_max + 1):
        idx_c = 2*n - 1
        idx_s = 2*n
        panels.append((0, n, idx_c, coeff_nms[idx_c]))
        panels.append((1, n, idx_s, coeff_nms[idx_s]))

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"

    with style_ctx:
        fig, axes = plt.subplots(2, n_cols, figsize=(2.6*n_cols, 5.4),
                                 constrained_layout=True)

        for row, col, idx, label in panels:
            ax = axes[row, col]
            theta = _one_hot_theta(n_max, idx)
            X, Y, Hx, Hy, Hz = _field_grid(theta, n_max, r0, r_max, grid_n)

            if label.startswith("Hz"):
                """
                No transverse component by construction -- show the
                out-of-page field using standard EM "dot in circle"
                (toward viewer) notation instead of an empty streamplot.
                """
                ax.set_facecolor("#202020" if dark_background else "#f0f0f0")
                n_dots = 5
                dot_xs = np.linspace(-r_max*0.7, r_max*0.7, n_dots)
                for dx in dot_xs:
                    for dy in dot_xs:
                        if dx*dx + dy*dy <= (r_max*0.75)**2:
                            ax.plot(dx, dy, marker="o", markersize=7,
                                   markerfacecolor="none",
                                   markeredgecolor=ring_color, linewidth=1.0)
                            ax.plot(dx, dy, marker=".", markersize=2.5,
                                   color=ring_color)
            else:
                mag = np.sqrt(Hx**2 + Hy**2)
                speed = np.clip(mag, 1e-12, None)
                try:
                    ax.streamplot(X, Y, Hx, Hy, color=speed, cmap=cmap,
                                 density=1.1, linewidth=1.0, arrowsize=0.9)
                except ValueError:
                    # fallback since streamplot is finicky
                    ax.quiver(X[::8, ::8], Y[::8, ::8], Hx[::8, ::8], Hy[::8, ::8],
                             speed[::8, ::8], cmap=cmap, scale_units="xy")

            circ = Circle((0, 0), R_TUBE_INNER, fill=False, edgecolor=ring_color,
                          linewidth=0.9, alpha=0.55, linestyle="--")
            ax.add_patch(circ)
            ax.set_xlim(-r_max, r_max); ax.set_ylim(-r_max, r_max)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontsize=10.5)

        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path


def plot_mode_overlay(n_max=3, r0=R0_DEFAULT, r_max=None, grid_n=200,
                      n_seeds=8, seed_radius_frac=0.55,
                      dark_background=False,
                      out_path="mode_overlay.png", dpi=300,
                      title="Multipole Orders — Overlaid Field Lines"):
    """
    Single shared panel: a handful of representative field lines from each
    order's canonical Cn=1 term (n=0...n_max), color-coded with a legend, so
    the n-fold angular symmetry of each order is directly comparable.

    Only the Cn terms are shown (Sn is the same shape rotated by
    90/n degrees, and Hz has no transverse streamline), keeping the
    overlay legible rather than doubling the line count.
    """
    r_max = r_max if r_max is not None else 1.3 * R_TUBE_INNER
    coeff_nms = coefficient_names(n_max)

    palette = plt.get_cmap("tab10").colors
    orders = list(range(0, n_max + 1))
    colors = {n: palette[n % len(palette)] for n in orders}

    seed_r = seed_radius_frac * r0
    seed_angles = np.linspace(0, 2*np.pi, n_seeds, endpoint=False)
    seed_points = [(seed_r*np.cos(a), seed_r*np.sin(a)) for a in seed_angles]

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"

    with style_ctx:
        fig, ax = plt.subplots(figsize=(7, 7))

        for n in orders:
            idx = 0 if n == 0 else (2*n - 1)   # C0, or Cn for n>=1
            theta = _one_hot_theta(n_max, idx)
            X, Y, Hx, Hy, Hz = _field_grid(theta, n_max, r0, r_max, grid_n)

            try:
                ax.streamplot(X, Y, Hx, Hy, color=colors[n],
                             start_points=seed_points,
                             density=30,             # high density needed w/ start_points
                             linewidth=1.6, arrowsize=1.1,
                             integration_direction="both",
                             minlength=0.02)
            except ValueError:
                # n=0 uniform field (perfectly straight parallel lines) fallback
                for (sx, sy) in seed_points:
                    ax.annotate("", xy=(sx + 0.6, sy), xytext=(sx - 0.6, sy),
                               arrowprops=dict(arrowstyle="-|>", color=colors[n], lw=1.6))

        circ = Circle((0, 0), R_TUBE_INNER, fill=False, edgecolor=ring_color,
                      linewidth=1.1, alpha=0.6)
        ax.add_patch(circ)

        legend_handles = [
            Line2D([0], [0], color=colors[n], lw=2.2,
                  label=f"n={n}  ({ORDER_NAMES.get(n, f'n={n}')})")
            for n in orders
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9.5,
                 framealpha=0.35, title="Multipole order")

        ax.set_xlim(-r_max, r_max); ax.set_ylim(-r_max, r_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x (in)"); ax.set_ylabel("y (in)")
        ax.set_title(title, fontsize=13, fontweight="bold")

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path


if __name__ == "__main__":
    out1 = plot_mode_gallery(
        n_max=3, r0=R0_DEFAULT,
        out_path="C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/Mode_gallery.png",
    )
    print(f"Saved: {out1}")

    out2 = plot_mode_overlay(
        n_max=3, r0=R0_DEFAULT,
        out_path="C:/Users/dappleby/Desktop/Results_Files_no_long_H/Polar_Method/Mode_overlay.png",
    )
    print(f"Saved: {out2}")
