import keypoint_moseq as kpms

project_dir = "kpms_project"
keypoint_data_path = "training_data"
model_name = "2025_08_01-23_41_19"
num_ar_iters = 50

config = lambda: kpms.load_config(project_dir)

model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name, iteration=num_ar_iters)

model = kpms.update_hypparams(model, kappa=1e3)

# model = kpms.fit_model(
#     model,
#     data,
#     metadata,
#     project_dir,
#     model_name,
#     ar_only=False,
#     start_iter=current_iter,
#     num_iters=current_iter + 450,
# )[0]


kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
results = kpms.extract_results(model, metadata, project_dir, model_name)
kpms.save_results_as_csv(results, project_dir, model_name)

coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "sleap")

kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())
kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, keypoints_only=True, **config())
kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())
