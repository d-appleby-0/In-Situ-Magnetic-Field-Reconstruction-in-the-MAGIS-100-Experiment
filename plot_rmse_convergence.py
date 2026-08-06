"""
plot_rmse_convergence.py
==========================
Overlays each z-pair's multipole RMSE-vs-n_max sweep on one set of axes,
with that pair's PCA benchmark drawn as a matching-colored horizontal
dashed line and its recommended (elbow) n_max marked with a star. Visualizes
the elbow-detection logic from multipole_model_selection and makes cross-pair
consistency or lack thereof in recommended order immediately visible.

Input is the per-pair sweep dict as already returned by
multipole_model_selection.rmse_vs_nmax_sweep() / run_model_selection()
e.g. model_selection_results["per_pair"][pair_label]["sweep"] from
multipole_model_selection_multi.run_multi_pair_model_selection().
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_rmse_vs_nmax_convergence(per_pair_sweeps: dict,
                                  yscale="log",
                                  dark_background=False,
                                  out_path="rmse_convergence.png", dpi=300,
                                  title="Multipole RMSE vs. Order — Cross-Pair Convergence",
                                  figsize=(8, 6)):
    """
    Parameters
    ----------
    per_pair_sweeps : dict {pair_id: sweep_dict}, where sweep_dict is the
        return value of rmse_vs_nmax_sweep() (has keys "sweep_df",
        "pca_rmse", "elbow_n_max", ...). Typically:
            model_selection_results["per_pair"][pid]["sweep"]
        for each pid in your run_multi_pair_model_selection() output.
    yscale : "log" (default, recommended - RMSE often spans orders of
        magnitude across n_max/pairs) or "linear"
    """
    palette = plt.get_cmap("tab10").colors
    pair_ids = sorted(per_pair_sweeps.keys())
    colors = {pid: palette[i % len(palette)] for i, pid in enumerate(pair_ids)}

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"

    with style_ctx:
        fig, ax = plt.subplots(figsize=figsize)

        n_max_all = set()
        for pid in pair_ids:
            sweep = per_pair_sweeps[pid]
            sweep_df = sweep["sweep_df"]
            color = colors[pid]
            n_max_all.update(sweep_df["n_max"].tolist())

            ax.plot(sweep_df["n_max"], sweep_df["cv_rmse"], marker="o",
                   color=color, linewidth=1.8, markersize=5.5, label=pid, zorder=3)
            """
            pca_rmse = sweep.get("pca_rmse")
            if pca_rmse is not None and np.isfinite(pca_rmse):
                ax.axhline(pca_rmse, color=color, linestyle="--", linewidth=1.1,
                          alpha=0.6, zorder=2)
            """
            elbow_n = sweep.get("elbow_n_max")
            if elbow_n is not None and elbow_n in set(sweep_df["n_max"]):
                elbow_row = sweep_df.loc[sweep_df["n_max"] == elbow_n].iloc[0]
                ax.scatter([elbow_n], [elbow_row["cv_rmse"]], marker="*", s=260,
                          color=color, edgecolor=ring_color, linewidth=0.8, zorder=4)

        ax.set_xlabel("Multipole order  n_max", fontsize=11)
        ax.set_ylabel("Cross-validated tube RMSE  (A/m)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xticks(sorted(n_max_all))
        ax.grid(True, alpha=0.25, which="both")

        # which color = which pair, and what solid/dashed/star mean
        pair_legend = ax.legend(loc="upper right", fontsize=9.5, framealpha=0.4,
                                title="Z-pair")
        ax.add_artist(pair_legend)

        style_handles = [
            Line2D([0], [0], color=ring_color, marker="o", linewidth=1.8,
                  markersize=5.5, label="multipole CV RMSE"),
            # Line2D([0], [0], color=ring_color, linestyle="--", linewidth=1.1,
            #       label="PCA benchmark (lower bound)"),
            Line2D([0], [0], color=ring_color, marker="*", markersize=13,
                  linewidth=0, label="recommended (elbow) n_max"),
        ]
        ax.legend(handles=style_handles, loc="lower right", fontsize=9,
                 framealpha=0.4, title="Line/marker key")

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path
