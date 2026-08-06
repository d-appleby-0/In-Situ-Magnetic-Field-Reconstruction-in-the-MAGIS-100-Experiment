"""
magis_pca_pipeline.py
=====================
PCA-based regression pipeline for magnetometer configuration selection.

Geometry (inches):
    tube_interior    r <= 2.75   free-space inside vacuum tube  → regression TARGETS
    tube_wall        2.75 < r <= 3.0   stainless wall           → discard
    sensor_zone      3.0 < r < 7.0    annular magnetometer gap  → regression INPUTS
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def _safe_inverse(pca, A: np.ndarray) -> np.ndarray:
    """inverse_transform that always works regardless of K or batch size."""
    A2d = np.atleast_2d(A) if A.ndim == 1 else A
    if A2d.ndim == 2 and A2d.shape[1] != pca.n_components_:
        A2d = A2d.reshape(-1, pca.n_components_)
    return pca.inverse_transform(np.atleast_2d(A2d).reshape(-1, pca.n_components_))



# ── Geometry constants (updated) ─────────────────────────────────────────────
R_TUBE_INNER  = 2.75   # inner radius: free-space vacuum tube interior
R_TUBE_OUTER  = 3.0    # outer radius: tube wall boundary
R_SHIELD      = 7.0    # inner face of shield material (approximately)


def classify_point(x: float, y: float) -> str:
    r2 = x * x + y * y
    if r2 <= R_TUBE_INNER ** 2:
        return "tube_interior"
    elif r2 <= R_TUBE_OUTER ** 2:
        return "tube_wall"
    elif r2 < R_SHIELD ** 2:
        return "sensor_zone"
    else:
        return "shield_material"


# Load and reclassify the CSV

def load_and_reclassify(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the parsed_fields CSV and re-apply the corrected geometry classifier.
    Drops shield_material and tube_wall rows (not used in regression).
    Detects and removes duplicate (coil_density, x, y, slice_id) entries.
    """
    df = pd.read_csv(csv_path)

    # Re-classify with updated radii
    df["region"] = df.apply(lambda row: classify_point(row["x"], row["y"]), axis=1)

    print("Region counts after reclassification:")
    print(df["region"].value_counts().to_string())
    print()

    # Keep only the two regions we care about
    df = df[df["region"].isin(["tube_interior", "sensor_zone"])].copy()

    # Check for and remove duplicate grid points
    key_cols = ["coil_density", "x", "y", "slice_id", "region"]
    n_before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"  WARNING: removed {n_dupes} duplicate (coil_density, x, y, slice_id) "
              f"rows. This usually means the same grid point appeared in multiple "
              f"source files. The first occurrence was kept.")
    return df


# STEP 1 — Extract tube field vectors, one per scenario

def build_tube_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    """
    Build the tube field matrix T.

    Each row = one (J, slice) scenario — i.e. scenario_id.
    This preserves variance across BOTH J and z, which is essential for PCA
    to find meaningful modes beyond K=1.

    Returns
    -------
    T           : (n_scenarios, n_tube_points) array  — one row per scenario
    scenarios   : list of scenario_id strings in row order
    tube_coords : list of (x, y, z) tuples in canonical column order
    """
    tube_df = df[df["region"] == "tube_interior"].copy()
    tube_df["x"] = tube_df["x"].round(6)
    tube_df["y"] = tube_df["y"].round(6)

    scenarios = sorted(df["scenario_id"].unique())

    # Canonical column order from the first scenario (all share same x,y grid)
    point_index = (
        tube_df[tube_df["scenario_id"] == scenarios[0]]
        [["x", "y", "z", "slice_id"]]
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )
    n_tube = len(point_index)
    print(f"Tube interior points per scenario : {n_tube}")
    print(f"Scenarios                         : {len(scenarios)}")

    rows = []
    for sid in scenarios:
        sub = (
            tube_df[tube_df["scenario_id"] == sid]
            .sort_values(["x", "y"])
        )
        if len(sub) != n_tube:
            print(f"  WARNING: {sid} has {len(sub)} tube points, expected {n_tube}")
        rows.append(sub["field_value"].values)

    T = np.vstack(rows)
    tube_coords = list(zip(point_index["x"], point_index["y"], point_index["z"]))
    return T, scenarios, tube_coords

# STEP 2 & 3 — PCA basis + project scenarios onto modes

def fit_pca(T: np.ndarray, variance_threshold: float = 0.99) -> tuple:
    """
    Run PCA on the tube field matrix T (n_scenarios x n_tube_points).

    Parameters
    ----------
    T                  : tube field matrix
    variance_threshold : keep enough modes to explain this fraction of variance

    Returns
    -------
    pca        : fitted sklearn PCA object
    K          : number of modes retained
    A          : (n_scenarios, K) coefficient matrix  a_k^(i) = <B^(i), phi_k>
    explained  : cumulative explained variance per mode
    """
    # Center the data (PCA will do this internally, exposed for clarity)
    pca_full = PCA()
    pca_full.fit(T)

    cumvar    = np.cumsum(pca_full.explained_variance_ratio_)
    K         = int(np.searchsorted(cumvar, variance_threshold)) + 1
    K         = min(K, T.shape[0] - 1)   # can't exceed n_scenarios - 1

    print(f"PCA: retaining K={K} modes (explains "
          f"{cumvar[K-1]*100:.2f}% of variance at threshold={variance_threshold*100:.0f}%)")

    n_show = min(len(pca_full.explained_variance_ratio_), 8)
    print(f"  Variance per mode (first {n_show}, retained K modes marked *):")
    for k in range(n_show):
        marker = "*" if k < K else " "
        pct    = pca_full.explained_variance_ratio_[k] * 100
        print(f"    {marker} PC{k+1}: {pct:.6f}%   (cumulative: {cumvar[k]*100:.6f}%)")
    if K == 1 and n_show > 1:
        pc2_pct = pca_full.explained_variance_ratio_[1] * 100
        if pc2_pct < 1e-4:
            print(f"  NOTE: PC2 explains {pc2_pct:.8f}% - this is a single-mode "
                  f"collapse, not a threshold artifact.")
        else:
            print(f"  NOTE: PC2 explains {pc2_pct:.6f}% (nonzero) but K=1 was still chosen, "
                  f"check variance_threshold={variance_threshold} against cumvar above.")

    pca = PCA(n_components=K)
    A   = pca.fit_transform(T)   # (n_scenarios, K)

    return pca, K, A, cumvar


# STEP 4 — Build sensor vectors

def build_sensor_matrix(df: pd.DataFrame, scenarios: list,
                        sensor_xy: list | None = None) -> tuple[np.ndarray, list]:
    """
    Build the sensor input matrix S (n_scenarios x n_sensor_points).

    Each row = one scenario (J, slice pair), matching build_tube_matrix.
    Each scenario has one slice, so (x, y) uniquely identifies a point
    within that scenario — no slice_id needed in the key.

    Parameters
    ----------
    scenarios : list of scenario_id strings from build_tube_matrix
    sensor_xy : optional list of (x, y) tuples to restrict sensor locations
    """
    sz = df[df["region"] == "sensor_zone"].copy()
    sz["x"] = sz["x"].round(6)
    sz["y"] = sz["y"].round(6)

    if sensor_xy is not None:
        xyset = set((round(x,6), round(y,6)) for x,y in sensor_xy)
        mask  = sz.apply(lambda r: (r["x"], r["y"]) in xyset, axis=1)
        sz    = sz[mask]

    # Build lookup: (scenario_id, x, y) -> field_value
    # Each scenario has exactly one slice so (scenario_id, x, y) is unique
    sz["_key"] = list(zip(sz["scenario_id"], sz["x"], sz["y"]))
    lookup = sz.set_index("_key")["field_value"]

    # Canonical column order from first scenario
    ref = (
        sz[sz["scenario_id"] == scenarios[0]]
        [["x", "y", "z"]]
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )
    print(f"Sensor zone points used           : {len(ref)}")

    xs = ref["x"].tolist()
    ys = ref["y"].tolist()

    rows = []
    for sid in scenarios:
        vals = []
        for x, y in zip(xs, ys):
            key = (sid, x, y)
            result = lookup.get(key)
            if result is None:
                raise ValueError(f"Missing sensor reading for {sid} at (x={x}, y={y})")
            vals.append(float(result))
        rows.append(vals)

    S = np.array(rows, dtype=float)
    sensor_locs = list(zip(ref["x"], ref["y"], ref["z"]))
    return S, sensor_locs

# STEP 5 — Ridge regression: S → A

def fit_regression(S: np.ndarray, A: np.ndarray,
                   alpha: float = 0.1, cv_folds: int = 10,
                   rng_seed: int = 42) -> tuple:
    """
    Fit Ridge regression A_hat = S @ W.T + b with k-fold CV predictions.

    Uses LOO when n_scenarios <= 30 (cheap and tight), k-fold otherwise.

    Returns
    -------
    model     : Ridge fitted on ALL data
    scaler    : StandardScaler fitted on S
    cv_preds  : (n_scenarios, K) out-of-fold predictions
    """
    scaler = StandardScaler()
    S_sc   = scaler.fit_transform(S)

    model  = Ridge(alpha=alpha)
    model.fit(S_sc, A)

    n        = S_sc.shape[0]
    splitter = LeaveOneOut() if n <= 30 else KFold(n_splits=cv_folds,
                                                    shuffle=True,
                                                    random_state=rng_seed)
    cv_preds = np.zeros_like(A)
    for train_idx, test_idx in splitter.split(S_sc):
        m = Ridge(alpha=alpha)
        m.fit(S_sc[train_idx], A[train_idx])
        cv_preds[test_idx] = np.atleast_2d(m.predict(S_sc[test_idx])).reshape(
            len(test_idx), -1)

    return model, scaler, cv_preds


# STEP 6 — Evaluate

def evaluate(pca, T: np.ndarray, A: np.ndarray, cv_preds: np.ndarray,
             S: np.ndarray, model, scaler,
             noise_levels: list = (0.0, 1e-4, 5e-4, 1e-3),
             n_noise_trials: int = 200,
             rng_seed: int = 42) -> dict:
    """
    Compute:
      - LOO RMSE on PCA coefficients
      - LOO RMSE on reconstructed tube |H| field
      - Noise robustness: RMSE vs sensor noise σ
      - Condition number of the sensor→coefficient mapping
    """
    rng = np.random.default_rng(rng_seed)
    results = {}

    # Coefficient RMSE (LOO)
    coeff_rmse = np.sqrt(mean_squared_error(A, cv_preds))
    results["coeff_rmse_cv"] = coeff_rmse

    # Tube field reconstruction RMSE (LOO)
    T_recon   = pca.inverse_transform(np.atleast_2d(cv_preds).reshape(-1, pca.n_components_))
    field_rmse = np.sqrt(mean_squared_error(T, T_recon))
    results["field_rmse_cv"] = field_rmse

    # Per-scenario worst-case tube error
    per_scenario_max = np.abs(T - T_recon).max(axis=1)
    results["worst_case_tube_error_mean"] = per_scenario_max.mean()
    results["worst_case_tube_error_max"]  = per_scenario_max.max()

    # Condition number of the sensor → coefficient MAPPING (not raw S)
    """
    model.coef_ has shape (K, n_sensors). Its singular values describe
    how sensitive the fitted mapping is to sensor noise/placement.
    """
    W     = np.atleast_2d(model.coef_)
    svs   = np.linalg.svd(W, compute_uv=False)
    condition_number_meaningful = W.shape[0] > 1
    if condition_number_meaningful:
        cond  = svs[0] / svs[-1] if svs[-1] > 1e-300 else np.inf
    else:
        svs_S = np.linalg.svd(S, compute_uv=False)
        cond = svs_S[0] / svs_S[-1] if svs_S[-1] > 1e-300 else np.inf
    results["condition_number"] = cond
    results["condition_number_meaningful"] = condition_number_meaningful
    results["singular_values"]  = svs

    # Noise robustness
    noise_results = {}
    for sigma in noise_levels:
        trial_rmses = []
        for _ in range(n_noise_trials):
            S_noisy  = S + rng.normal(0, sigma, size=S.shape)
            S_noisy_sc = scaler.transform(S_noisy)
            A_hat    = model.predict(S_noisy_sc)
            T_hat    = pca.inverse_transform(np.atleast_2d(A_hat).reshape(-1, pca.n_components_))
            trial_rmses.append(np.sqrt(mean_squared_error(T, T_hat)))
        noise_results[sigma] = {
            "mean_rmse" : np.mean(trial_rmses),
            "std_rmse"  : np.std(trial_rmses),
            "max_rmse"  : np.max(trial_rmses),
        }
    results["noise_robustness"] = noise_results

    return results


def print_results(results: dict) -> None:
    print("\n═══ Evaluation Results ═══")
    print(f"  CV coeff RMSE          : {results['coeff_rmse_cv']:.6f}")
    print(f"  CV field RMSE (|H|)    : {results['field_rmse_cv']:.6f}")
    print(f"  Worst-case tube error   : mean={results['worst_case_tube_error_mean']:.6f}"
          f"  max={results['worst_case_tube_error_max']:.6f}")
    if results.get("condition_number_meaningful", True):
        print(f"  Condition number        : {results['condition_number']:.2f}")
    else:
        print(f"  Condition number        : {results['condition_number']:.2f}  "
              f"(K=1 -- mapping-based number is degenerate/uninformative; "
              f"printed the raw sensor-matrix S condition number instead)")
    print("\n  Noise robustness (field RMSE vs sensor σ):")
    print(f"    {'σ':>10}  {'mean RMSE':>12}  {'std':>10}  {'max RMSE':>12}")
    for sigma, nr in results["noise_robustness"].items():
        print(f"    {sigma:10.2e}  {nr['mean_rmse']:12.6f}  "
              f"{nr['std_rmse']:10.6f}  {nr['max_rmse']:12.6f}")


# Top-level runner

def run_pipeline(csv_path: str | Path,
                 variance_threshold: float = 0.99,
                 ridge_alpha: float = 0.1,
                 sensor_xy: list | None = None,
                 noise_levels: tuple = (0.0, 1e-4, 5e-4, 1e-3),
                 save_outputs: bool = True) -> dict:
    """
    End-to-end pipeline from CSV → evaluation results.

    Parameters
    ----------
    csv_path           : path to parsed_fields.csv
    variance_threshold : PCA variance to retain (0.95–0.99)
    ridge_alpha        : Ridge regularisation strength
    sensor_xy          : optional list of (x,y) sensor positions to use
                         None = use all sensor_zone grid points
    noise_levels       : sensor noise σ values to test
    save_outputs       : write coefficient matrix and results to disk

    Returns
    -------
    dict with keys: df, T, S, pca, A, model, scaler, cv_preds, results
    """
    csv_path = Path(csv_path)
    print(f"Loading: {csv_path}\n")

    df        = load_and_reclassify(csv_path)
    T, scenarios, tube_coords = build_tube_matrix(df)
    print(f"Unique scenarios                  : {len(scenarios)}")
    print(f"  {scenarios[:5]}{'...' if len(scenarios)>5 else ''}\n")

    pca, K, A, cumvar = fit_pca(T, variance_threshold)
    S, sensor_locs    = build_sensor_matrix(df, scenarios, sensor_xy)

    print()
    model, scaler, cv_preds = fit_regression(S, A, ridge_alpha)
    print(f"Ridge regression fitted  (α={ridge_alpha})\n")

    results = evaluate(pca, T, A, cv_preds, S, model, scaler, noise_levels)
    print_results(results)

    payload = dict(df=df, T=T, S=S, scenarios=scenarios,
                   tube_coords=tube_coords, sensor_locs=sensor_locs,
                   pca=pca, K=K, A=A, cumvar=cumvar,
                   model=model, scaler=scaler,
                   cv_preds=cv_preds, results=results)

    if save_outputs:
        out_dir = csv_path.parent
        np.save(out_dir / "tube_matrix_T.npy",  T)
        np.save(out_dir / "sensor_matrix_S.npy", S)
        np.save(out_dir / "pca_coefficients_A.npy", A)
        pd.DataFrame(A, index=scenarios,
                     columns=[f"PC{k+1}" for k in range(K)]).to_csv(
            out_dir / "pca_coefficients.csv")
        print(f"\n   Saved T, S, A matrices and coefficient CSV to {out_dir}")

    return payload
