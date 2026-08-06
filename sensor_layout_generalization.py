"""
sensor_layout_generalization.py
================================
Searches for an optimal sensor layout on one reference z-pair, then checks
whether that layout generalizes to the other three pairs. Falls back to a
pair-specific layout wherever generalization is poor.

This deliberately reuses sensor_layout_search.py's existing functions
unchanged (search_layouts, build_sensor_subset_matrix, score_layout) —
the only new logic here is the cross-pair evaluation + fallback decision.
"""

import warnings
import numpy as np
import pandas as pd

from z_pair_utils import add_pair_id, validate_pair_coverage, PAIR_IDS

from magis_pca_pipeline import build_tube_matrix, fit_pca
from sensor_layout_search import (
    get_candidate_sensor_points, build_sensor_subset_matrix, score_layout,
    search_layouts, search_layout_greedy,
)


def _layout_from_row(row: pd.Series, n_sensors: int) -> list:
    """
    Cast to native float explicitly - otherwise these stay np.float64,
    which under NumPy>=2.0 repr prints as literal "np.float64(3.25)" any
    time the layout list/tuple is printed (reports, summaries, etc).
    """
    xs = [float(row[f"sensor{j+1}_x"]) for j in range(n_sensors)]
    ys = [float(row[f"sensor{j+1}_y"]) for j in range(n_sensors)]
    return list(zip(xs, ys))


def _points_df_from_layout(layout: list) -> pd.DataFrame:
    return pd.DataFrame({"x": [p[0] for p in layout], "y": [p[1] for p in layout]})


def _run_search(method, df, T, A, pca, scenarios, n_sensors, max_candidates,
                ridge_alpha, noise_sigma_T, verbose):
    """
    Dispatch to the requested search strategy, all returning the same
    result_df schema (row 0 = recommended layout).
    """
    if method == "greedy":
        return search_layout_greedy(df, T, A, pca, scenarios, n_sensors=n_sensors,
                                    ridge_alpha=ridge_alpha, noise_sigma_T=noise_sigma_T,
                                    verbose=verbose)
    elif method in ("exhaustive", "random"):
        """
        "exhaustive" vs "random" is just whether max_candidates caps it -
        both go through search_layouts, which already handles both cases.
        """
        return search_layouts(df, T, A, pca, scenarios, n_sensors=n_sensors,
                              max_candidates=max_candidates, ridge_alpha=ridge_alpha,
                              noise_sigma_T=noise_sigma_T, verbose=verbose)
    else:
        raise ValueError(f"Unknown method={method!r}. Use 'greedy', 'exhaustive', or 'random'.")


def search_and_validate_layout(df: pd.DataFrame,
                               n_sensors: int = 2,
                               reference_pair: str = "P1",
                               method: str = "greedy",
                               max_candidates: int | None = None,
                               ridge_alpha: float = 0.1,
                               noise_sigma_T: float = 1e-9,
                               degradation_tolerance: float = 0.5,
                               verbose: bool = True) -> dict:
    """
    Search sensor layout on `reference_pair`, then validate on all other
    pairs. Falls back to a pair-specific layout if the global layout
    degrades that pair's field_rmse_cv by more than `degradation_tolerance`
    relative to that pair's own best-achievable layout.

    Parameters
    ----------
    df                    : full dataset (all slices, all pairs)
    n_sensors             : number of magnetometer positions to search for
    reference_pair        : which pair_id to search the layout on (default P1)
    method                : "greedy" (default) - fast forward selection,
                             O(n_pts*n_sensors) evaluations instead of
                             O(C(n_pts,n_sensors)); not guaranteed globally
                             optimal but empirically close in testing (see
                             search_layout_greedy's docstring). "exhaustive"
                             or "random" fall back to the original
                             search_layouts() combinatorial search (capped
                             by max_candidates if given).
    max_candidates         : cap on combinations tried per search when
                             method is "exhaustive"/"random" (None =
                             exhaustive, ignored for method="greedy", which
                             is always O(n_pts*n_sensors) regardless).
    degradation_tolerance : max allowed fractional RMSE increase of the
                             global layout vs. that pair's own optimum
                             before falling back to a pair-specific layout.
                             e.g. 0.5 = global layout may be up to 50% worse.

    Returns
    -------
    dict with:
        reference_pair        : the pair used for the initial search
        global_layout          : [(x,y), ...] found on the reference pair
        final_layouts          : {pair_id: layout} — global or fallback, per pair
        generalization_report  : {pair_id: {rmse_with_global, rmse_best_own,
                                             degradation, used_fallback}}
    """
    df = add_pair_id(df)
    validate_pair_coverage(df)

    pairs_present = [p for p in PAIR_IDS if (df["pair_id"] == p).any()]
    if reference_pair not in pairs_present:
        raise ValueError(
            f"reference_pair={reference_pair!r} has no data. "
            f"Pairs present: {pairs_present}"
        )

    # Step 1: search on the reference pair
    ref_df = df[df["pair_id"] == reference_pair]
    T_ref, scen_ref, _ = build_tube_matrix(ref_df)
    pca_ref, K_ref, A_ref, _ = fit_pca(T_ref, variance_threshold=0.99)

    if verbose:
        print(f"\n{'='*80}\n  Searching {n_sensors}-sensor layout on reference pair "
              f"{reference_pair}\n{'='*80}")

    ref_search = _run_search(method, ref_df, T_ref, A_ref, pca_ref, scen_ref,
                             n_sensors, max_candidates, ridge_alpha, noise_sigma_T, verbose)
    best_row = ref_search.iloc[0]
    global_layout = _layout_from_row(best_row, n_sensors)

    if verbose:
        print(f"\n  Best layout on {reference_pair}: {global_layout}  "
              f"(field_rmse_cv={best_row['field_rmse_cv']:.6f})")

    # Step 2: validate on every other pair, with fallback
    generalization = {}
    final_layouts = {reference_pair: global_layout}

    for pid in pairs_present:
        if pid == reference_pair:
            generalization[pid] = {
                "rmse_with_global": float(best_row["field_rmse_cv"]),
                "rmse_best_own": float(best_row["field_rmse_cv"]),
                "degradation": 0.0,
                "used_fallback": False,
            }
            continue

        if verbose:
            print(f"\n{'-'*80}\n  Validating global layout on pair {pid}\n{'-'*80}")

        sub_df = df[df["pair_id"] == pid]
        T_s, scen_s, _ = build_tube_matrix(sub_df)
        pca_s, K_s, A_s, _ = fit_pca(T_s, variance_threshold=0.99)

        # Evaluate the GLOBAL layout on this pair's own field model
        subset_pts = _points_df_from_layout(global_layout)
        try:
            S_global = build_sensor_subset_matrix(sub_df, scen_s, subset_pts)
            metrics_global = score_layout(S_global, A_s, pca_s, T_s,
                                          ridge_alpha=ridge_alpha,
                                          noise_sigma_T=noise_sigma_T)
            rmse_global = metrics_global["field_rmse_cv"]
        except ValueError as e:
            warnings.warn(
                f"Pair {pid}: global layout points not present on this pair's "
                f"sensor grid ({e}). Treating as maximal degradation -> fallback."
            )
            rmse_global = float("inf")

        # This pair's own best-achievable layout (independent search)
        own_search = _run_search(method, sub_df, T_s, A_s, pca_s, scen_s,
                                 n_sensors, max_candidates, ridge_alpha, noise_sigma_T,
                                 verbose=False)
        own_best_row = own_search.iloc[0]
        rmse_own_best = float(own_best_row["field_rmse_cv"])

        if rmse_own_best > 0 and np.isfinite(rmse_global):
            degradation = (rmse_global - rmse_own_best) / rmse_own_best
        else:
            degradation = float("inf")

        use_fallback = degradation > degradation_tolerance

        if use_fallback:
            final_layouts[pid] = _layout_from_row(own_best_row, n_sensors)
        else:
            final_layouts[pid] = global_layout

        generalization[pid] = {
            "rmse_with_global": rmse_global,
            "rmse_best_own": rmse_own_best,
            "degradation": degradation,
            "used_fallback": use_fallback,
        }

        if verbose:
            flag = "fallback to pair-specific layout" if use_fallback else "global layout retained"
            print(f"  Pair {pid}: global RMSE={rmse_global:.6f}  "
                  f"own-best RMSE={rmse_own_best:.6f}  "
                  f"degradation={degradation:.1%}  -> {flag}")

    if verbose:
        print_generalization_summary(reference_pair, global_layout,
                                     final_layouts, generalization,
                                     degradation_tolerance)

    return {
        "reference_pair": reference_pair,
        "global_layout": global_layout,
        "final_layouts": final_layouts,
        "generalization_report": generalization,
    }


def print_generalization_summary(reference_pair, global_layout, final_layouts,
                                 generalization, degradation_tolerance) -> None:
    print(f"\n{'='*80}")
    print(f"  SENSOR LAYOUT GENERALIZATION SUMMARY")
    print(f"  Reference pair: {reference_pair}   Global layout: {global_layout}")
    print(f"  Degradation tolerance for fallback: {degradation_tolerance:.0%}")
    print(f"{'='*80}")
    print(f"\n  {'pair':<6} {'rmse_global':>13} {'rmse_own_best':>15} "
          f"{'degradation':>13} {'decision':<28}")
    print("  " + "-"*80)
    for pid, g in generalization.items():
        decision = "fallback (pair-specific)" if g["used_fallback"] else "global layout"
        rmse_g = f"{g['rmse_with_global']:.6f}" if np.isfinite(g["rmse_with_global"]) else "inf"
        print(f"  {pid:<6} {rmse_g:>13} {g['rmse_best_own']:>15.6f} "
              f"{g['degradation']:>12.1%} {decision:<28}")
    n_fallback = sum(1 for g in generalization.values() if g["used_fallback"])
    print(f"\n  {n_fallback}/{len(generalization)} pairs required fallback to a "
          f"pair-specific layout.")
    if n_fallback > 0:
        print(f"  Final layouts differ across pairs — this is an expected outcome, "
              f"not an error.")
    print()


def generalization_report_to_df(generalization: dict) -> pd.DataFrame:
    """Convenience: flatten the generalization dict into a DataFrame for saving/plotting."""
    rows = [{"pair_id": pid, **metrics} for pid, metrics in generalization.items()]
    return pd.DataFrame(rows)
