# Evaluate your experiments on the MVTec 3D-AD dataset

The evaluation scripts can be used to assess the performance of a method on the
MVTec 3D-AD dataset. Given a directory with anomaly images, the scripts compute
the area under the PRO curve for anomaly localization. Additionally, the area
under the ROC curve for anomaly classification is computed.

## Evaluating a single experiment.
It requires an anomaly map to be present for each test sample in our dataset in
`.tiff` format. Anomaly maps must contain real-valued anomaly scores and their
size must match the one of the original dataset images. Anomaly maps must all
share the same base directory and adhere to the following folder structure:
`<anomaly_maps_dir>/<object_name>/test/<defect_name>/<image_id>.tiff`

To evaluate a single experiment, the script `evaluate_experiment.py` can be
used. It requires the following user arguments:

- `dataset_base_dir`: Base directory that contains the MVTec 3D-AD dataset.
- `anomaly_maps_dir`: Base directory that contains the corresponding anomaly maps.

Optional parameters can be specified as follows:

- `output_dir`: Directory to store evaluation results as `.json` files.
- `pro_integration_limit`: Integration limit for the computation of the AU-PRO.
- `pro_num_thresholds`: Number of equidstant thresholds to use for computing the AU-PRO.
- `evaluated_objects`: List of dataset object categories for which the computation should be performed.

A possible example call to this script would be:
```
  python evaluate_experiment.py --dataset_base_dir 'path/to/dataset/' \
                                --anomaly_maps_dir 'path/to/anomaly_maps/' \
                                --output_dir 'metrics/'
                                --pro_integration_limit 0.3 \
                                --pro_num_thresholds 512
```

## Evaluate multiple experiments
If more than a single experiment should be evaluated at once, the script
`evaluate_multiple_experiments.py` can be used. All directories with anomaly
maps can be specified in a `config.json` file with the following structure:
```
{
    "exp_base_dir": "paths/to/all/experiments",
    "anomaly_maps_dirs": {
    "experiment_id_1": "eg/variation_model/voxel/anomaly_images",
    "experiment_id_1": "eg/ae/voxel/anomaly_images",
    "experiment_id_1": "eg/fanogan/voxel/anomaly_images",
    "...": "..."
    }
}
```
- `exp_base_dir`: Base directory that contains all experimental results for each evaluated method.
- `anomaly_maps_dirs`: Dictionary that contains an identifier for each evaluated experiment and the location of its anomaly maps relative to the `exp_base_dir`.

The evaluation can be run by calling `evaluate_multiple_experiments.py`,
providing the following user arguments:

- `dataset_base_dir`: Base directory that contains the MVTec 3D-AD dataset.
- `experiment_configs`: Path to the above `config.json` that contains all experimens to be evaluated.

Optional parameters can be specified as follows:

- `output_dir`: Directory to store evaluation results as `.json` file for each evaluated experiment.
- `pro_integration_limit`: Integration limit for the computation of the AU-PRO.
- `pro_num_thresholds`: Number of thresholds that serve as integration points for the AU-PRO.

A possible example call to this script would be:
```
  python evaluate_multiple_experiments.py --dataset_base_dir 'path/to/dataset/' \
                                         --experiment_configs 'configs.json' \
                                         --output_dir 'metrics/' \
                                         --pro_integration_limit 0.3 \
                                         --pro_num_thresholds 512
```

## Visualize the evaluation results.
After running `evaluate_multiple_experiments.py`, the script `print_metrics.py`
can be used to visualize all computed metrics in a table. It requires only a
single user argument:

- `metrics_folder`: The base directory that contains the computed metrics for each evaluated method.

This directory is usually identical to the output directory specified in
`evaluate_multiple_experiments.py`.

# License
The license agreement for our evaluation code is found in the accompanying
`LICENSE.txt` file.
