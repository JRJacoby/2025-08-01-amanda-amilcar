import keypoint_moseq as kpms

project_dir = "kpms_project"
model_name = "2025_08_01-23_41_19"
num_ar_iters = 50

model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name, iteration=num_ar_iters)

model = kpms.update_hypparams(model, kappa=1e3)

model = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    model_name,
    ar_only=False,
    start_iter=current_iter,
    num_iters=current_iter + 50,
)[0]
