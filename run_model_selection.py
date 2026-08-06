from multipole_model_selection import run_model_selection

results = run_model_selection(
    df=df,
    sensor_positions=sensor_pos,
    n_max_range=range(1, 6),      # sweeps n_max 1 through 5
    pca_rmse=pca_rmse,            # the PCA benchmark from step 2
    noise_sigma_T=1e-9,           # your magnetometer noise floor in Tesla
    n_folds=10,
    save_dir=r"C:\Users\dapplepy\Desktop\Results_Files_no_long_H",
)

recommended_n_max = results["recommended_n_max"]
print(f"Use n_max={recommended_n_max} in the full report")
