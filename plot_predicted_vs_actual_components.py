"""
plot_predicted_vs_actual_components.py
=========================================
Two extensions to the |H|-magnitude predicted-vs-actual plot:

1. plot_predicted_vs_actual_components() - separate Hx/Hy/Hz scatter
   panels (each with its own y=x line, RMSE/R^2, noise-floor/CV
   diagnostics), so a component-specific mismatch (e.g. the Hz/C0
   degeneracy found in magnitude-mode fitting) is visible directly rather
   than hidden inside a combined |H| magnitude.

2. plot_diff_vs_J() - (predicted - true) plotted directly against J, with
   a zero-reference line and a binned mean+/-std summary trend, so the
   J-dependence of reconstruction error is visible on its own rather than
   folded into a single aggregate RMSE number.

Both rewuire vector-mode data (field_Hx/Hy/Hz columns in df) for the
component-wise plot; plot_diff_vs_J can work on the scalar |H| difference
even in magnitude-mode, or on a specific component if vector data exists.

Reuses the exact same numeric-key (coil_density, slice_id) join logic
already validated in compute_predicted_vs_actual, rather than reinventing
a new join.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multipole_basis import reconstruct_field, reconstruct_magnitude, coefficient_names, R0_DEFAULT


def has_vector_data(df: pd.DataFrame) -> bool:
    return ("field_Hx" in df.columns and df["field_Hx"].notna().any()
           and "field_Hy" in df.columns and df["field_Hy"].notna().any()
           and "field_Hz" in df.columns and df["field_Hz"].notna().any())


def _matched_keys(df, fit_df, round_decimals=6, match_rate_threshold=0.9):
    """Shared join-key logic, factored out from compute_predicted_vs_actual
    so both the magnitude and component-wise paths use the identical,
    already-validated numeric join rather than two separate copies."""
    tube_df = df[df["region"] == "tube_interior"].copy()
    fit_df = fit_df.copy()
    tube_df["_J_key"] = tube_df["coil_density"].round(round_decimals)
    fit_df["_J_key"] = fit_df["coil_density"].round(round_decimals)

    tube_keys = set(zip(tube_df["_J_key"], tube_df["slice_id"]))
    fit_keys = set(zip(fit_df["_J_key"], fit_df["slice_id"]))
    matched_keys = tube_keys & fit_keys
    match_rate = len(matched_keys) / max(len(fit_keys), 1)

    print(f"  Matched {len(matched_keys)}/{len(fit_keys)} fit_df scenarios to "
          f"tube-interior data  ({match_rate*100:.1f}%).")
    if match_rate < match_rate_threshold:
        only_in_fit = sorted(fit_keys - tube_keys)[:5]
        only_in_tube = sorted(tube_keys - fit_keys)[:5]
        raise ValueError(
            f"Only {match_rate*100:.1f}% of fit_df scenarios matched -- refusing "
            f"to plot a partial/misleading subset.\n"
            f"  Example (J, slice_id) in fit_df but NOT in df:  {only_in_fit}\n"
            f"  Example (J, slice_id) in df but NOT in fit_df: {only_in_tube}\n")

    return tube_df, fit_df, matched_keys


# component-wise (Hx, Hy, Hz) reconstruction

def compute_predicted_vs_actual_components(df, fit_df, n_max, r0=R0_DEFAULT,
                                           coeff_names=None, round_decimals=6,
                                           match_rate_threshold=0.9):
    """
    Same join as compute_predicted_vs_actual, but returns per-component
    (Hx, Hy, Hz) true/predicted arrays instead of the combined |H|
    magnitude. Requires vector-mode data (field_Hx/Hy/Hz in df).

    Returns
    -------
    dict with keys "Hx", "Hy", "Hz", each mapping to
    (true_vals, pred_vals) arrays, plus "J_all" and "scenario_all".
    """
    if not has_vector_data(df):
        raise ValueError(
            "Component-wise reconstruction requires vector-mode data "
            "(field_Hx/Hy/Hz columns in df), which is not present here. "
            "Magnitude-mode data only has the scalar field_value column, "
            "use plot_predicted_vs_actual() (the |H| magnitude version) "
            "instead, or re-export from Opera with vector components."
        )

    coeff_nms = coeff_names or coefficient_names(n_max)
    tube_df, fit_df, matched_keys = _matched_keys(
        df, fit_df, round_decimals=round_decimals, match_rate_threshold=match_rate_threshold)

    true_Hx, true_Hy, true_Hz = [], [], []
    pred_Hx, pred_Hy, pred_Hz = [], [], []
    scen_chunks, J_chunks = [], []

    for _, row in fit_df.iterrows():
        key = (row["_J_key"], row["slice_id"])
        if key not in matched_keys:
            continue
        sub = tube_df[(tube_df["_J_key"] == key[0]) &
                      (tube_df["slice_id"] == key[1])].sort_values(["x", "y"])
        if sub.empty:
            continue
        theta = row[coeff_nms].values.astype(float)
        pts = list(zip(sub["x"], sub["y"]))
        H_pred = reconstruct_field(theta, pts, n_max, r0=r0)  # (n_pts, 3)

        true_Hx.append(sub["field_Hx"].values)
        true_Hy.append(sub["field_Hy"].values)
        true_Hz.append(sub["field_Hz"].values)
        pred_Hx.append(H_pred[:, 0])
        pred_Hy.append(H_pred[:, 1])
        pred_Hz.append(H_pred[:, 2])
        scen_chunks.append(np.full(len(pts), row["scenario_id"], dtype=object))
        J_chunks.append(np.full(len(pts), float(row["coil_density"])))

    if not true_Hx:
        raise ValueError("No matching tube-interior points found for any scenario in fit_df.")

    return {
        "Hx": (np.concatenate(true_Hx), np.concatenate(pred_Hx)),
        "Hy": (np.concatenate(true_Hy), np.concatenate(pred_Hy)),
        "Hz": (np.concatenate(true_Hz), np.concatenate(pred_Hz)),
        "J_all": np.concatenate(J_chunks),
        "scenario_all": np.concatenate(scen_chunks),
    }


def _rank_colors(J_all, rng_seed=42):
    """Percentile-rank color values (0-1), matching the fix already applied
    to the |H| magnitude plot, guaranteeing full colormap usage regardless
    of how skewed the J sampling is."""
    unique_J_sorted = np.unique(J_all)
    rank_lookup = {v: i for i, v in enumerate(unique_J_sorted)}
    ranks = np.array([rank_lookup[v] for v in J_all], dtype=float)
    return ranks / max(len(unique_J_sorted) - 1, 1), unique_J_sorted


def plot_predicted_vs_actual_components(df, fit_df, n_max, r0=R0_DEFAULT,
                                        cmap="plasma", dark_background=False,
                                        point_size=6, alpha=0.4,
                                        noise_sigma_T=None,
                                        out_path="predicted_vs_actual_components.png",
                                        dpi=300, figsize=(16, 5.5),
                                        shuffle_plot_order=True, rng_seed=42):
    """
    Three-panel scatter: Hx, Hy, Hz reconstructed vs. true, each with its
    own y=x line, RMSE/R^2, and (if noise_sigma_T given) noise-floor ratio
    and true-value CV since each component can have a very different
    variance structure (e.g. Hz degenerate while Hx/Hy are fine).
    """
    comp_data = compute_predicted_vs_actual_components(df, fit_df, n_max, r0=r0)
    J_all = comp_data["J_all"]
    ranks_norm, unique_J_sorted = _rank_colors(J_all)

    plot_idx = np.arange(len(J_all))
    if shuffle_plot_order:
        np.random.default_rng(rng_seed).shuffle(plot_idx)

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"
    MU0 = 4 * np.pi * 1e-7

    with style_ctx:
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

        for ax, comp in zip(axes, ["Hx", "Hy", "Hz"]):
            true_vals, pred_vals = comp_data[comp]
            true_p = true_vals[plot_idx]
            pred_p = pred_vals[plot_idx]
            ranks_p = ranks_norm[plot_idx]

            rmse = float(np.sqrt(np.mean((pred_vals - true_vals)**2)))
            ss_res = float(np.sum((true_vals - pred_vals)**2))
            ss_tot = float(np.sum((true_vals - true_vals.mean())**2))
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")
            true_cv = (float(np.std(true_vals) / abs(np.mean(true_vals)))
                      if np.mean(true_vals) != 0 else float("nan"))

            lo = min(true_vals.min(), pred_vals.min())
            hi = max(true_vals.max(), pred_vals.max())
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            lims = (lo - pad, hi + pad)

            ax.plot(lims, lims, linestyle="--", color=ring_color, linewidth=1.2, alpha=0.8, zorder=1)
            sca = ax.scatter(true_p, pred_p, c=ranks_p, cmap=cmap, s=point_size,
                            alpha=alpha, linewidths=0, zorder=2, vmin=0, vmax=1)

            stats_txt = f"RMSE = {rmse:.3e}\nR² = {r2:.4f}\nCV = {true_cv*100:.1f}%"
            if noise_sigma_T is not None:
                sigma_H = noise_sigma_T / MU0
                stats_txt += f"\nRMSE/noise = {rmse/sigma_H:.3f}x"
            ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=8.5,
                   va="top", ha="left",
                   bbox=dict(boxstyle="round", facecolor=("black" if dark_background else "white"),
                            alpha=0.55, edgecolor=ring_color, linewidth=0.5))

            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_aspect("equal")
            ax.set_xlabel(f"True {comp}  (A/m)", fontsize=10.5)
            ax.set_ylabel(f"Reconstructed {comp}  (A/m)", fontsize=10.5)
            ax.set_title(comp, fontsize=12, fontweight="bold")

        cbar = fig.colorbar(sca, ax=axes, fraction=0.025, pad=0.02, shrink=0.9)
        tick_fracs = np.linspace(0, 1, 6)
        tick_idx = np.round(tick_fracs * (len(unique_J_sorted) - 1)).astype(int)
        cbar.set_ticks(tick_fracs)
        cbar.set_ticklabels([f"{unique_J_sorted[i]:.3g}" for i in tick_idx])
        cbar.set_label("J  (A/in², percentile-ranked)", fontsize=10)

        fig.suptitle("Reconstructed vs. True Field Components", fontsize=14, fontweight="bold")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path


# difference vs J

def compute_diff_vs_J(df, fit_df, n_max, r0=R0_DEFAULT, component=None,
                      coeff_names=None, round_decimals=6, match_rate_threshold=0.9):
    """
    Compute (predicted - true) per tube point, alongside the J value of
    the scenario it came from.

    Parameters
    ----------
    component : None -> uses |H| magnitude difference (works for both
                magnitude-mode and vector-mode data).
                "Hx"/"Hy"/"Hz" -> specific vector component's SIGNED difference
                (requires vector-mode data).

    Returns
    -------
    J_all, diff_all : parallel 1-D arrays (signed difference, not absolute
    -- sign matters for spotting systematic bias)
    """
    if component is None:
        coeff_nms = coeff_names or coefficient_names(n_max)
        tube_df, fit_df_j, matched_keys = _matched_keys(
            df, fit_df, round_decimals=round_decimals, match_rate_threshold=match_rate_threshold)
        diff_chunks, J_chunks = [], []
        for _, row in fit_df_j.iterrows():
            key = (row["_J_key"], row["slice_id"])
            if key not in matched_keys:
                continue
            sub = tube_df[(tube_df["_J_key"] == key[0]) &
                          (tube_df["slice_id"] == key[1])].sort_values(["x", "y"])
            if sub.empty:
                continue
            theta = row[coeff_nms].values.astype(float)
            pts = list(zip(sub["x"], sub["y"]))
            true_vals = sub["field_value"].values
            pred_vals = reconstruct_magnitude(theta, pts, n_max, r0=r0)
            diff_chunks.append(pred_vals - true_vals)
            J_chunks.append(np.full(len(pts), float(row["coil_density"])))
        return np.concatenate(J_chunks), np.concatenate(diff_chunks)
    else:
        comp_data = compute_predicted_vs_actual_components(
            df, fit_df, n_max, r0=r0, coeff_names=coeff_names,
            round_decimals=round_decimals, match_rate_threshold=match_rate_threshold)
        true_vals, pred_vals = comp_data[component]
        return comp_data["J_all"], (pred_vals - true_vals)


def _compute_binned_trend(J_all, diff_all, n_bins=15):
    """Shared binning logic: log-spaced bins across the J range, returning
    (bin_centers, bin_means, bin_stds) for the summary trend overlay."""
    J_positive = J_all[J_all > 0]
    if len(J_positive) > 1:
        bins = np.geomspace(J_positive.min(), J_positive.max(), n_bins + 1)
    else:
        bins = np.linspace(J_all.min(), J_all.max() + 1e-9, n_bins + 1)
    bin_idx = np.digitize(J_all, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_centers, bin_means, bin_stds = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_centers.append(np.sqrt(bins[b] * bins[b+1]))  # geometric mean of bin edges
        bin_means.append(diff_all[mask].mean())
        bin_stds.append(diff_all[mask].std())
    return np.array(bin_centers), np.array(bin_means), np.array(bin_stds)


def _render_diff_panel(ax, J_all, diff_all, label, ring_color,
                       noise_sigma_T=None, n_bins=15,
                       point_size=5, alpha=0.25, show_legend=True):
    """
    Shared rendering logic for a single (predicted - true) vs. J panel --
    used by both plot_diff_vs_J (standalone) and plot_diff_vs_J_components
    (small-multiples), so the two never drift out of sync with each other.
    """
    bin_centers, bin_means, bin_stds = _compute_binned_trend(J_all, diff_all, n_bins=n_bins)

    ax.scatter(J_all, diff_all, s=point_size, alpha=alpha, color="tab:cyan",
              linewidths=0, zorder=1, label="per-point difference")
    ax.axhline(0, linestyle="--", color=ring_color, linewidth=1.2, alpha=0.8, zorder=2)

    ax.plot(bin_centers, bin_means, color="tab:orange", linewidth=2.0,
           marker="o", markersize=4, zorder=3, label="binned mean")
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                    color="tab:orange", alpha=0.25, zorder=2, label="binned ±1σ")

    if noise_sigma_T is not None:
        MU0 = 4 * np.pi * 1e-7
        sigma_H = noise_sigma_T / MU0
        ax.axhspan(-sigma_H, sigma_H, color="tab:green", alpha=0.12, zorder=0,
                  label=f"±noise floor ({sigma_H:.2e} A/m)")

    ax.set_xscale("log")
    ax.set_xlabel("J  (A/in²)", fontsize=10.5)
    ax.set_ylabel(f"Reconstructed − True {label}  (A/m)", fontsize=10.5)
    ax.set_title(label, fontsize=12, fontweight="bold")
    if show_legend:
        ax.legend(loc="best", fontsize=8, framealpha=0.4)


def plot_diff_vs_J(df, fit_df, n_max, r0=R0_DEFAULT, component=None,
                   noise_sigma_T=None, n_bins=15,
                   dark_background=False, point_size=5, alpha=0.25,
                   out_path="diff_vs_J.png", dpi=300,
                   title=None, figsize=(8, 5.5)):
    """
    (predicted - true) plotted directly against J (log x-axis), with a
    zero-reference line and a binned mean +/- std trend overlaid on the
    raw per-point scatter, so the J-dependence of reconstruction error is
    visible directly rather than folded into one aggregate RMSE number.
    """
    J_all, diff_all = compute_diff_vs_J(df, fit_df, n_max, r0=r0, component=component)
    label = f"H{component[-1]}" if component else "|H|"
    if title is None:
        title = f"Reconstruction Error vs. Coil Current Density ({label})"

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"

    with style_ctx:
        fig, ax = plt.subplots(figsize=figsize)
        _render_diff_panel(ax, J_all, diff_all, label, ring_color,
                          noise_sigma_T=noise_sigma_T, n_bins=n_bins,
                          point_size=point_size, alpha=alpha, show_legend=True)
        ax.set_title(title, fontsize=13, fontweight="bold")  # override panel's short label title

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path


def plot_diff_vs_J_components(df, fit_df, n_max, r0=R0_DEFAULT,
                              noise_sigma_T=None, n_bins=15,
                              dark_background=False, point_size=5, alpha=0.25,
                              out_path="diff_vs_J_components.png", dpi=300,
                              title="Reconstruction Error vs. J — All Components",
                              figsize=(16, 5.0)):
    """
    Small-multiples version of plot_diff_vs_J: one panel each for Hx, Hy,
    Hz, side by side, sharing the same binning/zero-line/noise-band
    rendering logic so they're directly comparable. Requires vector-mode
    data (field_Hx/Hy/Hz in df).
    """
    if not has_vector_data(df):
        raise ValueError(
            "plot_diff_vs_J_components requires vector-mode data "
            "(field_Hx/Hy/Hz columns in df). Use plot_diff_vs_J(component=None) "
            "for the |H| magnitude version instead, which works with "
            "magnitude-mode data too."
        )

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"

    with style_ctx:
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

        for ax, comp in zip(axes, ["Hx", "Hy", "Hz"]):
            J_all, diff_all = compute_diff_vs_J(df, fit_df, n_max, r0=r0, component=comp)
            _render_diff_panel(ax, J_all, diff_all, comp, ring_color,
                              noise_sigma_T=noise_sigma_T, n_bins=n_bins,
                              point_size=point_size, alpha=alpha,
                              show_legend=(comp == "Hx"))  # legend once is enough

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path
