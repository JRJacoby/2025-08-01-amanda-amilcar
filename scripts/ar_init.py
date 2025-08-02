import keypoint_moseq as kpms

project_dir = "kpms_project"
keypoint_data_path = "training_data"  # can be a file, a directory, or a list of files
config = lambda: kpms.load_config(project_dir)

coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "sleap")
data, metadata = kpms.format_data(coordinates, confidences, **config())

pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)
kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

model = kpms.init_model(data, pca=pca, **config())
model = kpms.update_hypparams(model, kappa=3e5)
num_ar_iters = 50

model, model_name = kpms.fit_model(model, data, metadata, project_dir, ar_only=True, num_iters=num_ar_iters)
