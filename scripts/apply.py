from jax_moseq.utils import set_mixed_map_iters

set_mixed_map_iters(4)

import keypoint_moseq as kpms
import os

project_dir = "kpms_project"
model_name = "2025_08_01-23_41_19"
new_data = "all_data"
visualizations_dir = f"{project_dir}/{model_name}/apply_all_visualizations"
results_dir = f"{project_dir}/{model_name}/apply_all_results"

config = lambda: kpms.load_config(project_dir)

os.makedirs(visualizations_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

model = kpms.load_checkpoint(project_dir, model_name)[0]

coordinates, confidences, bodyparts = kpms.load_keypoints(new_data, "sleap")
data, metadata = kpms.format_data(coordinates, confidences, **config())
results = kpms.apply_model(model, data, metadata, project_dir, model_name, save_results=False, **config())
kpms.save_results_as_csv(results, project_dir, model_name, save_dir=results_dir)

kpms.generate_trajectory_plots(
    coordinates,
    results,
    project_dir,
    model_name,
    output_dir=visualizations_dir,
    **config(),
)
kpms.generate_grid_movies(
    results,
    project_dir,
    model_name,
    coordinates=coordinates,
    output_dir=visualizations_dir,
    **config(),
)
kpms.plot_similarity_dendrogram(
    coordinates,
    results,
    project_dir,
    model_name,
    output_dir=visualizations_dir,
    **config(),
)
