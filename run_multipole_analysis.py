from multipole_report import generate_full_report
from multipole_model_selection import run_model_selection
from magis_pca_pipeline import load_and_reclassify, build_tube_matrix, fit_pca
from sensor_layout_search import search_layouts

df = load_and_reclassify(r"C:\Users\dappleby\Desktop\Results_Files_no_long_H\parsed_fields.csv")
T, scenarios, tube_coords = build_tube_matrix(df)
pca, K, A, cumvar    = fit_pca(T, variance_threshold=0.99)

layout_df = search_layouts(df, T, A, pca, scenarios, n_sensors=2, noise_sigma_T=1e-9)
layout_df.to_csv(r"C:\Users\dappleby\Desktop\Results_Files_no_long_H\layout_search_n2.csv", index=False)

best = layout_df.iloc[0]
sensor_pos = [(best["sensor1_x"], best["sensor1_y"]),
              (best["sensor2_x"], best["sensor2_y"])]
pca_rmse   = float(best["field_rmse_cv"])
print(f"Best layout: {sensor_pos}  PCA RMSE={pca_rmse:.6f} A/m")   # from layout_search_n2.csv row 0

results = run_model_selection(
    df=df,
    sensor_positions=sensor_pos,
    n_max_range=range(1, 6),      # sweeps n_max 1 through 5
    pca_rmse=pca_rmse,            # the PCA benchmark from step 2
    noise_sigma_T=1e-9,           # your magnetometer noise floor in Tesla
    n_folds=10,
    save_dir=r"C:\Users\dappleby\Desktop\Results_Files_no_long_H",
)

recommended_n_max = results["recommended_n_max"]
print(f"Use n_max={recommended_n_max} in the full report")

generate_full_report(
    df=df,
    sensor_positions=sensor_pos,
    n_max=recommended_n_max,
    noise_sigma_T=1e-9,
    run_orientation_search=True,
    save_dir=r"C:\Users\dappleby\Desktop\Results_Files_no_long_H",
)
