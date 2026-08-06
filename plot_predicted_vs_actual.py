"""
plot_predicted_vs_actual.py
=============================
reconstructed vs. true |H| at every tube-interior location, across every
scenario, with a y=x reference line. Cheap to produce as it reuses the
fitted per-scenario coefficients in fit_df (from fit_all_scenarios()) and
recomputes reconstruct_magnitude() at each tube point rather than needing
any new fitting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style_utils import clean_decimal_axis

from multipole_basis import reconstruct_magnitude, coefficient_names, R0_DEFAULT


def compute_predicted_vs_actual(df, fit_df, n_max, r0=R0_DEFAULT, coeff_names=None,
                                round_decimals=6, match_rate_threshold=0.9):
    """
    Recompute per-point (true, predicted) |H| pairs at every tube-interior
    location, for every scenario present in fit_df.

    Parameters
    ----------
    df      : reclassified dataset (from load_and_reclassify) containing the
              true field_value at tube_interior points
    fit_df  : per-scenario coefficient table (from fit_all_scenarios())
    n_max   : multipole order used for that fit
    coeff_names : optional override of coefficient_names(n_max)
    round_decimals : rounding tolerance for matching J values numerically
    match_rate_threshold : raise a loud error if fewer than this fraction of
              fit_df's scenarios can be matched to tube-interior data,
              rather than silently plotting a partial/misleading subset

    Returns
    -------
    true_all, pred_all, scenario_all, J_all : parallel 1-D arrays, one entry
    per (scenario, tube point)
    """
    coeff_nms = coeff_names or coefficient_names(n_max)
    tube_df = df[df["region"] == "tube_interior"].copy()
    fit_df = fit_df.copy()

    if "slice_id" not in tube_df.columns or "slice_id" not in fit_df.columns:
        raise ValueError("Both df and fit_df must have a 'slice_id' column to join on.")

    tube_df["_J_key"] = tube_df["coil_density"].round(round_decimals)
    fit_df["_J_key"]  = fit_df["coil_density"].round(round_decimals)

    tube_keys = set(zip(tube_df["_J_key"], tube_df["slice_id"]))
    fit_keys  = set(zip(fit_df["_J_key"],  fit_df["slice_id"]))
    matched_keys = tube_keys & fit_keys
    match_rate = len(matched_keys) / max(len(fit_keys), 1)

    print(f"  Matched {len(matched_keys)}/{len(fit_keys)} fit_df scenarios to "
          f"tube-interior data by (coil_density, slice_id)  ({match_rate*100:.1f}%).")

    if match_rate < match_rate_threshold:
        only_in_fit  = sorted(fit_keys - tube_keys)[:5]
        only_in_tube = sorted(tube_keys - fit_keys)[:5]
        raise ValueError(
            f"Only {match_rate*100:.1f}% of fit_df scenarios matched tube-interior "
            f"data by (coil_density, slice_id) - refusing to plot a partial/"
            f"misleading subset. This usually means df and fit_df came from "
            f"different pipeline runs (e.g. re-parsed with different J precision) "
            f"or df has been filtered to a subset that doesn't cover fit_df's "
            f"scenarios.\n"
            f"  Example (J, slice_id) in fit_df but not in df:  {only_in_fit}\n"
            f"  Example (J, slice_id) in df but not in fit_df: {only_in_tube}\n"
            f"Re-run df and fit_df together from the same parsed dataset, or "
            f"pass round_decimals= higher/lower if J values are being rounded "
            f"to a collision or a near-miss."
        )

    true_chunks, pred_chunks, scen_chunks, J_chunks = [], [], [], []
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
        true_vals = sub["field_value"].values
        pred_vals = reconstruct_magnitude(theta, pts, n_max, r0=r0)

        true_chunks.append(true_vals)
        pred_chunks.append(pred_vals)
        scen_chunks.append(np.full(len(true_vals), row["scenario_id"], dtype=object))
        J_chunks.append(np.full(len(true_vals), float(row["coil_density"])))

    if not true_chunks:
        raise ValueError("No matching tube-interior points found for any scenario in fit_df.")

    return (np.concatenate(true_chunks), np.concatenate(pred_chunks),
            np.concatenate(scen_chunks), np.concatenate(J_chunks))


def plot_predicted_vs_actual(df, fit_df, n_max, r0=R0_DEFAULT,
                             color_by="J_rank", cmap="plasma",
                             dark_background=False, point_size=7, alpha=0.45,
                             out_path="predicted_vs_actual.png", dpi=300,
                             title="Reconstructed vs. True Tube Field",
                             figsize=(6.5, 6.5), sensor_positions=None,
                             axes_per_sensor=3, oversampling_threshold=2.0,
                             noise_sigma_T=None, shuffle_plot_order=True,
                             rng_seed=42):
    """
    Scatter of reconstructed vs. true |H| at every tube-interior point,
    across every scenario in fit_df, with a y=x reference line and overall
    RMSE/R^2 annotated.

    Parameters
    ----------
    color_by : "J_rank" (default) colors points by the percentile rank of
               their scenario's J among all unique J's present, not the raw
               value. 
    shuffle_plot_order : if True (default), randomizes the order points are
               drawn in before scatter. With many overlapping points,
               whichever scenario happens to be drawn last visually
               dominates the overlap regardless of its actual color
               diversity. Shuffling removes this systematic z-order bias.
    noise_sigma_T : sensor noise floor in Tesla. If supplied, annotates RMSE
               as a multiple of the noise floor (converted to A/m via mu0)
               directly on the plot.
    sensor_positions : the sensor layout used for this fit.
    """
    true_all, pred_all, scenario_all, J_all = compute_predicted_vs_actual(
        df, fit_df, n_max, r0=r0)

    # Diagnostic: does J_all actually span the full range
    unique_J = np.unique(J_all)
    print(f"  J_all diagnostic: {len(unique_J)} unique J value(s) actually "
          f"present in the plotted data (out of {fit_df['coil_density'].nunique()} "
          f"unique J in fit_df).")
    print(f"    range: [{J_all.min():.6g}, {J_all.max():.6g}]")
    if len(unique_J) <= 20:
        print(f"    unique values: {np.round(unique_J, 4).tolist()}")
    else:
        """
        coarse log-spaced histogram so a collapsed/lopsided distribution
        is visible even with too many unique values to list individually
        """
        finite_J = unique_J[unique_J > 0]
        if len(finite_J) > 1:
            bins = np.geomspace(finite_J.min(), finite_J.max(), 8)
            counts, edges = np.histogram(J_all[J_all > 0], bins=bins)
            print(f"    log-spaced point-count histogram (not unique-J count):")
            for c, lo_edge, hi_edge in zip(counts, edges[:-1], edges[1:]):
                print(f"      [{lo_edge:8.3g}, {hi_edge:8.3g}) : {c:>6,} points")
    if len(unique_J) < fit_df["coil_density"].nunique() * 0.9:
        print(f"    X Fewer unique J values reached the plot than exist in fit_df -- "
              f"some scenarios' points may be getting dropped, overwritten, or "
              f"collapsed together. Check the match-rate line above and the "
              f"df/fit_df construction upstream of this call.")

    overall_rmse = float(np.sqrt(np.mean((pred_all - true_all)**2)))
    ss_res = float(np.sum((true_all - pred_all)**2))
    ss_tot = float(np.sum((true_all - true_all.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")

    # Noise-floor comparison + true-value CV
    true_cv = float(true_all.std() / true_all.mean()) if true_all.mean() != 0 else float("nan")
    noise_floor_lines = []
    low_variance_caveat = None
    if noise_sigma_T is not None:
        MU0 = 4 * np.pi * 1e-7
        sigma_H = noise_sigma_T / MU0
        ratio_to_noise = overall_rmse / sigma_H if sigma_H > 0 else float("nan")
        noise_floor_lines.append(f"RMSE / noise floor = {ratio_to_noise:.3f}x")
        print(f"  [noise floor] sigma_H={sigma_H:.4e} A/m (from noise_sigma_T={noise_sigma_T:.1e} T)  "
              f"RMSE/noise floor = {ratio_to_noise:.3f}x")
    print(f"  [true-value spread] coefficient of variation = {true_cv*100:.2f}%  "
          f"(std/mean of true |H| across all plotted points)")
    if true_cv < 0.15:
        low_variance_caveat = (
            f"True values vary only {true_cv*100:.1f}% (CV) - R² is unreliable\n"
            f"here (tiny denominator inflates any absolute error). Judge fit\n"
            f"quality by absolute RMSE" +
            (" vs. noise floor above." if noise_sigma_T is not None else "."))
        print(f"  X Low true-value variance (CV={true_cv*100:.2f}% < 15%): R² is not a "
              f"reliable fit-quality metric here, judge by absolute RMSE"
              + (" relative to the noise floor instead." if noise_sigma_T is not None else " instead."))

    # Conditioning diagnostic: is this fit well-posed?
    n_coeffs = 2*n_max + 2
    conditioning_lines = []
    if sensor_positions is not None:
        n_meas = len(sensor_positions) * axes_per_sensor
        oversampling = n_meas / n_coeffs
        status = ("well-conditioned" if oversampling >= 3 else
                 "acceptable" if oversampling >= oversampling_threshold else
                 "marginal" if oversampling >= 1 else "UNDERDETERMINED")
        flag = "" if oversampling >= oversampling_threshold else "  ⚠"
        conditioning_lines.append(
            f"n_sensors={len(sensor_positions)}  n_max={n_max}  "
            f"oversampling={oversampling:.2f}x ({status}){flag}")
        print(f"  [conditioning] n_meas={n_meas}  n_coeffs={n_coeffs}  "
              f"oversampling={oversampling:.2f}x  -> {status}{flag}")

    if "condition_number" in fit_df.columns:
        finite_cond = fit_df["condition_number"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite_cond):
            cond_med, cond_max = finite_cond.median(), finite_cond.max()
            print(f"  [conditioning] condition_number: median={cond_med:.1f}  "
                  f"max={cond_max:.1f}  (n={len(finite_cond)} finite of {len(fit_df)} scenarios)")

    lo = min(true_all.min(), pred_all.min())
    hi = max(true_all.max(), pred_all.max())
    pad = 0.03 * (hi - lo if hi > lo else 1.0)
    lims = (lo - pad, hi + pad)

    style_ctx = plt.style.context("dark_background") if dark_background else plt.style.context("default")
    ring_color = "white" if dark_background else "black"

    with style_ctx:
        fig, ax = plt.subplots(figsize=figsize)
        clean_decimal_axis(ax=ax)

        ax.plot(lims, lims, linestyle="--", color=ring_color, linewidth=1.3,
               alpha=0.8, zorder=1, label="y = x  (perfect prediction)")

        # Shuffle draw order
        plot_idx = np.arange(len(true_all))
        if shuffle_plot_order:
            np.random.default_rng(rng_seed).shuffle(plot_idx)
        true_p, pred_p, J_p = true_all[plot_idx], pred_all[plot_idx], J_all[plot_idx]

        if color_by == "J_rank":
            # Color by percentile rank of J
            unique_J_sorted = np.unique(J_all)
            rank_lookup = {v: i for i, v in enumerate(unique_J_sorted)}
            ranks = np.array([rank_lookup[v] for v in J_p], dtype=float)
            ranks_norm = ranks / max(len(unique_J_sorted) - 1, 1)
            sca = ax.scatter(true_p, pred_p, c=ranks_norm, cmap=cmap,
                            s=point_size, alpha=alpha, linewidths=0, zorder=2,
                            vmin=0, vmax=1)
            cbar = fig.colorbar(sca, ax=ax, fraction=0.045, pad=0.03)
            # label ticks
            tick_fracs = np.linspace(0, 1, 6)
            tick_idx = np.round(tick_fracs * (len(unique_J_sorted) - 1)).astype(int)
            cbar.set_ticks(tick_fracs)
            cbar.set_ticklabels([f"{unique_J_sorted[i]:.3g}" for i in tick_idx])
            cbar.set_label("J  (A/in², percentile-ranked color scale)", fontsize=10.5)
        elif color_by == "J":
            J_nonzero = J_all[J_all > 0]
            if len(J_nonzero) > 0 and (J_all.max() / max(J_nonzero.min(), 1e-12)) > 20:
                # use log color normalization
                from matplotlib.colors import LogNorm
                norm = LogNorm(vmin=max(J_nonzero.min(), 1e-6), vmax=J_all.max(), clip=True)
                cbar_label = "J  (A/in², log scale)"
            else:
                norm = None
                cbar_label = "J  (A/in²)"
            sca = ax.scatter(true_p, pred_p, c=J_p, cmap=cmap, norm=norm,
                            s=point_size, alpha=alpha, linewidths=0, zorder=2)
            cbar = fig.colorbar(sca, ax=ax, fraction=0.045, pad=0.03)
            cbar.set_label(cbar_label, fontsize=10.5)
        else:
            ax.scatter(true_p, pred_p, color="tab:cyan", s=point_size,
                      alpha=alpha, linewidths=0, zorder=2)

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("True |H|  (A/m)", fontsize=11)
        ax.set_ylabel("Reconstructed |H|  (A/m)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        stats_txt = f"RMSE = {overall_rmse:.4e} A/m\nR² = {r2:.5f}\nn = {len(true_all):,} points"
        if noise_floor_lines:
            stats_txt += "\n" + "\n".join(noise_floor_lines)
        stats_txt += f"\ntrue-value CV = {true_cv*100:.2f}%"
        if conditioning_lines:
            stats_txt += "\n" + "\n".join(conditioning_lines)
        ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
               fontsize=9.5, va="top", ha="left",
               bbox=dict(boxstyle="round", facecolor=("black" if dark_background else "white"),
                        alpha=0.55, edgecolor=ring_color, linewidth=0.6))

        if low_variance_caveat is not None:
            caveat_color = "#FFD54A" if dark_background else "#8A6D00"
            ax.text(0.03, 0.03, low_variance_caveat, transform=ax.transAxes,
                   fontsize=8.5, va="bottom", ha="left", color=caveat_color,
                   bbox=dict(boxstyle="round", facecolor=("black" if dark_background else "white"),
                            alpha=0.55, edgecolor=caveat_color, linewidth=0.6))

        ax.legend(loc="lower right" if low_variance_caveat is None else "center right",
                 fontsize=9.5, framealpha=0.4)

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        

    return {"out_path": out_path, "rmse": overall_rmse, "r2": r2, "n_points": len(true_all)}
