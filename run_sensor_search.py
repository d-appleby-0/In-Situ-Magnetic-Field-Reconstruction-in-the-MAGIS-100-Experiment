from magis_pca_pipeline import load_and_reclassify, build_tube_matrix, fit_pca
from sensor_layout_search import search_layouts

df = load_and_reclassify(r"C:\Users\dappleby\Desktop\Results_Files_no_long_H\parsed_fields.csv")
T, scenarios, tube_coords = build_tube_matrix(df)
pca, K, A, cumvar = fit_pca(T, variance_threshold=0.99)

results_n2 = search_layouts(df, T, A, pca, scenarios, n_sensors=2, noise_sigma_T=1e-9)
results_n3 = search_layouts(df, T, A, pca, scenarios, n_sensors=3, noise_sigma_T=1e-9,
                            max_candidates=8000)
