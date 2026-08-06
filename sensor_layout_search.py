"""
sensor_layout_search.py
========================
Exhaustive (or sampled) search over candidate magnetometer placements to find
the 2-3 sensor layout that best reconstructs the PCA tube field coefficients.

Candidate positions: (x, y) points in the sensor_zone of one reference slice.
"""

import itertools
import math
import numpy as np
import pandas as pd
from pathlib import Path

from magis_pca_pipeline import (
    fit_regression, evaluate, classify_point,
    build_tube_matrix, fit_pca, load_and_reclassify,
)


# Candidate sensor locations

def get_candidate_sensor_points(df: pd.DataFrame, scenarios: list) -> pd.DataFrame:
    """
    Return candidate magnetometer positions: the (x, y) sensor_zone grid
    from the first scenario (all scenarios share the same spatial grid).
    """
    sz      = df[df["region"] == "sensor_zone"]
    ref_sid = scenarios[0]
    ref     = sz[sz["scenario_id"] == ref_sid][["x", "y", "z", "slice_id"]]
    return (
        ref.drop_duplicates(subset=["x", "y"])
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )


# Sensor subset matrix

def build_sensor_subset_matrix(df: pd.DataFrame, scenarios: list,
                               point_subset: pd.DataFrame) -> np.ndarray:
    """
    Build S (n_scenarios x n_subset_points).
    """
    sz = df[df["region"] == "sensor_zone"].copy()
    sz["x"] = sz["x"].round(6)
    sz["y"] = sz["y"].round(6)

    sz["_key"] = list(zip(sz["scenario_id"], sz["x"], sz["y"]))
    lookup     = sz.set_index("_key")["field_value"]

    xs = point_subset["x"].round(6).tolist()
    ys = point_subset["y"].round(6).tolist()

    rows = []
    for sid in scenarios:
        vals = []
        for x, y in zip(xs, ys):
            result = lookup.get((sid, x, y))
            if result is None:
                raise ValueError(
                    f"Missing sensor reading for {sid} at (x={x}, y={y}). "
                    f"Check all scenarios share the same spatial grid."
                )
            vals.append(float(result))
        rows.append(vals)
    return np.array(rows, dtype=float)


# Score a single layout

def score_layout(S: np.ndarray, A: np.ndarray, pca, T: np.ndarray,
                 ridge_alpha: float = 0.1,
                 noise_sigma_T: float = 1e-9,
                 mu0: float = 4 * np.pi * 1e-7,
                 n_noise_trials: int = 30) -> dict:
    """
    Fit and evaluate ridge regression for one candidate sensor layout.
    """
    model, scaler, cv_preds = fit_regression(S, A, alpha=ridge_alpha)
    sigma_H = noise_sigma_T / mu0   # convert T -> A/m

    results = evaluate(
        pca, T, A, cv_preds, S, model, scaler,
        noise_levels=(0.0, sigma_H, sigma_H * 10),
        n_noise_trials=n_noise_trials,
    )
    return {
        "n_sensors"        : S.shape[1],
        "condition_number" : results["condition_number"],
        "field_rmse_cv"    : results["field_rmse_cv"],
        "coeff_rmse_cv"    : results["coeff_rmse_cv"],
        "noise_rmse_floor" : results["noise_robustness"][sigma_H]["mean_rmse"],
        "noise_rmse_10x"   : results["noise_robustness"][sigma_H * 10]["mean_rmse"],
        "worst_case_error" : results["worst_case_tube_error_max"],
    }


# Combinatorics helpers

def _sample_combos(n_pts: int, n_sensors: int, max_candidates: int,
                   rng_seed: int = 42, exhaustive_fraction_threshold: float = 0.3) -> list:
    """
    Draw up to max_candidates unique n_sensors-point combinations from
    range(n_pts), without materializing the full C(n_pts, n_sensors) space
    first.
    Falls back to exhaustive generation when max_candidates is close to (or
    exceeds) the true combinatorial count, since rejection sampling becomes
    inefficient as the sampled fraction approaches the full space.
    """
    n_combos_true = math.comb(n_pts, n_sensors)
    if max_candidates >= n_combos_true:
        return list(itertools.combinations(range(n_pts), n_sensors))
    if max_candidates >= exhaustive_fraction_threshold * n_combos_true:
        # sampling a large fraction of a space that's itself not huge --
        # just generate exhaustively and subsample, cheaper than rejection
        all_combos = list(itertools.combinations(range(n_pts), n_sensors))
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(all_combos), size=max_candidates, replace=False)
        return [all_combos[i] for i in idx]

    rng = np.random.default_rng(rng_seed)
    seen = set()
    while len(seen) < max_candidates:
        combo = tuple(sorted(rng.choice(n_pts, size=n_sensors, replace=False).tolist()))
        seen.add(combo)
    return list(seen)


# Exhaustive / sampled layout search

def search_layouts(df: pd.DataFrame, T: np.ndarray, A: np.ndarray, pca,
                   scenarios: list, n_sensors: int = 2,
                   max_candidates: int | None = None,
                   ridge_alpha: float = 0.1,
                   noise_sigma_T: float = 1e-9,
                   rng_seed: int = 42,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Search over n_sensors-point subsets of the sensor_zone grid.

    max_candidates : if set and smaller than the true combinatorial count,
        draws that many unique random combinations directly (see
        _sample_combos) rather than materializing the full space first.
        None = exhaustive.
    """
    ref_points = get_candidate_sensor_points(df, scenarios)
    n_pts      = len(ref_points)
    n_combos   = math.comb(n_pts, n_sensors)

    if verbose:
        print(f"Candidate sensor locations : {n_pts}")
        print(f"{n_pts} choose {n_sensors} = {n_combos:,} combinations")

    if max_candidates is not None and max_candidates < n_combos:
        all_combos = _sample_combos(n_pts, n_sensors, max_candidates, rng_seed=rng_seed)
        if verbose:
            print(f"Randomly sampling {len(all_combos):,} of {n_combos:,} combinations "
                  f"(direct k-subset sampling, not materializing the full space)")
    else:
        all_combos = list(itertools.combinations(range(n_pts), n_sensors))

    records = []
    for i, combo in enumerate(all_combos):
        if verbose and i % max(1, len(all_combos) // 20) == 0:
            print(f"  [{i}/{len(all_combos)}] evaluating layouts...")

        subset = ref_points.iloc[list(combo)].reset_index(drop=True)

        try:
            S = build_sensor_subset_matrix(df, scenarios, subset)
        except ValueError as e:
            if i < 3:
                print(f"  [WARNING] layout {combo} skipped (missing data): {e}")
            continue

        try:
            metrics = score_layout(S, A, pca, T,
                                   ridge_alpha=ridge_alpha,
                                   noise_sigma_T=noise_sigma_T)
        except Exception as e:
            if i < 3:
                print(f"  [WARNING] layout {combo} failed: {type(e).__name__}: {e}")
            continue

        record = {
            "combo_idx": combo,
            **{f"sensor{j+1}_x": subset.loc[j, "x"] for j in range(n_sensors)},
            **{f"sensor{j+1}_y": subset.loc[j, "y"] for j in range(n_sensors)},
            **{f"sensor{j+1}_z": subset.loc[j, "z"] for j in range(n_sensors)},
            **metrics,
        }
        records.append(record)

    if not records:
        raise RuntimeError(
            "No layouts were successfully evaluated. "
            "Check warning messages above for the underlying error."
        )

    result_df = pd.DataFrame(records).sort_values("field_rmse_cv").reset_index(drop=True)

    if verbose:
        print(f"\n   Evaluated {len(result_df):,} layouts")
        print("\nTop 10 by field_rmse_cv:")
        sensor_cols = [c for c in result_df.columns
                       if c.startswith("sensor") and c.endswith(("_x","_y","_z"))]
        print(result_df[["field_rmse_cv","condition_number","noise_rmse_floor"]
                        + sensor_cols].head(10).to_string(index=False))

    return result_df


# Greedy forward selection (fast alternative to exhaustive/random)

def search_layout_greedy(df: pd.DataFrame, T: np.ndarray, A: np.ndarray, pca,
                        scenarios: list, n_sensors: int = 2,
                        ridge_alpha: float = 0.1,
                        noise_sigma_T: float = 1e-9,
                        random_check_candidates: int | None = None,
                        rng_seed: int = 42,
                        return_history: bool = False,
                        verbose: bool = True):
    """
    Greedy forward selection: build the sensor layout one point at a time,
    each step choosing whichever remaining candidate most improves
    field_rmse_cv, holding the already-selected points fixed.

    Cost: O(n_pts * n_sensors) score_layout() evaluations, vs.
    O(C(n_pts, n_sensors)) for exhaustive/random search -- e.g. for
    n_pts=64, n_sensors=6: ~380 evaluations instead of ~74 million.

    Parameters
    ----------
    random_check_candidates : optional int. If set, ALSO evaluates this
        many random n_sensors-combinations (via the same fast sampler used
        by search_layouts) purely for comparison, and prints whether greedy
        beat the best random draw found -- a cheap empirical sanity check
        rather than a blind assumption that greedy is better on your data.
    return_history : if True, returns (result_df, history_df) where
        history_df has one row per greedy step, showing the running
        layout and score as sensors were added one at a time.

    Returns
    -------
    result_df : single-row DataFrame with the same column schema as
        search_layouts()'s output (sensor{j}_x/y/z, field_rmse_cv,
        condition_number, ...)
    """
    ref_points = get_candidate_sensor_points(df, scenarios)
    n_pts = len(ref_points)

    if n_sensors > n_pts:
        raise ValueError(f"n_sensors={n_sensors} exceeds n_pts={n_pts} candidate locations.")

    selected_idx: list = []
    remaining_idx = list(range(n_pts))
    history_rows = []

    if verbose:
        print(f"Greedy forward selection: {n_pts} candidates, building "
              f"{n_sensors}-sensor layout ({n_pts * n_sensors:,} evaluations "
              f"vs. {math.comb(n_pts, n_sensors):,} for exhaustive)")

    for step in range(n_sensors):
        best_score, best_idx, best_metrics = None, None, None

        for cand in remaining_idx:
            trial_idx = selected_idx + [cand]
            subset = ref_points.iloc[trial_idx].reset_index(drop=True)
            try:
                S = build_sensor_subset_matrix(df, scenarios, subset)
                metrics = score_layout(S, A, pca, T, ridge_alpha=ridge_alpha,
                                      noise_sigma_T=noise_sigma_T)
            except Exception:
                continue
            score = metrics["field_rmse_cv"]
            if best_score is None or score < best_score:
                best_score, best_idx, best_metrics = score, cand, metrics

        if best_idx is None:
            raise RuntimeError(
                f"Greedy search failed at step {step+1}/{n_sensors}: no candidate "
                f"could be evaluated (all raised exceptions or had missing data)."
            )

        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

        if verbose:
            pt = ref_points.iloc[best_idx]
            print(f"  step {step+1}/{n_sensors}: added (x={pt['x']:.2f}, y={pt['y']:.2f})  "
                  f"field_rmse_cv={best_score:.6f}")

        subset_now = ref_points.iloc[selected_idx].reset_index(drop=True)
        history_rows.append({
            "step": step + 1,
            **{f"sensor{j+1}_x": subset_now.loc[j, "x"] for j in range(len(selected_idx))},
            **{f"sensor{j+1}_y": subset_now.loc[j, "y"] for j in range(len(selected_idx))},
            **{f"sensor{j+1}_z": subset_now.loc[j, "z"] for j in range(len(selected_idx))},
            **best_metrics,
        })

    history_df = pd.DataFrame(history_rows)
    result_df = history_df.iloc[[-1]].reset_index(drop=True)

    if random_check_candidates:
        if verbose:
            print(f"\nSanity check: evaluating {random_check_candidates:,} random "
                  f"combinations for comparison...")
        random_result_df = search_layouts(
            df, T, A, pca, scenarios, n_sensors=n_sensors,
            max_candidates=random_check_candidates, ridge_alpha=ridge_alpha,
            noise_sigma_T=noise_sigma_T, rng_seed=rng_seed, verbose=False)
        best_random_rmse = float(random_result_df.iloc[0]["field_rmse_cv"])
        greedy_rmse = float(result_df.iloc[0]["field_rmse_cv"])
        verdict = ("  greedy beat random search" if greedy_rmse <= best_random_rmse
                  else "X random search found a BETTER layout than greedy on this run")
        print(f"  Greedy field_rmse_cv={greedy_rmse:.6f}  vs.  best of "
              f"{random_check_candidates:,} random={best_random_rmse:.6f}  -> {verdict}")

    if verbose:
        print(f"\n  Greedy layout complete: field_rmse_cv={result_df.iloc[0]['field_rmse_cv']:.6f}")

    if return_history:
        return result_df, history_df
    return result_df
