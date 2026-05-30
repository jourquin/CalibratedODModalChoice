# Run parameters

```
================================================================================
Run parameters
================================================================================
models: [1, 2, 3]
dataset_labels: {1: 'Europe_NUTS2', 2: 'Benelux_NUTS3', 3: 'Germany_NUTS3'}
groups: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
weight_type: raw
weight_alpha: None
dist_thresholds_km: [150.0, 300.0]
calibration_level: mode_od_group
include_rail_cost: False
generate_reports: False
save_row_predictions: False
max_iterations: 5000
calibration_max_iterations: 80
calibration_damping: 0.7
calibration_max_step: 2.0
calibration_max_abs_constant: 8.0
user: nodus
host: 127.0.0.1
database: stochastic
output_dir: output
model_name_prefix: 
store_parameters_in_table: True
parameter_table: None
parameter_table_prefix: modal_choice_params
holdout_fraction: 0.0
holdout_by: od
holdout_seed: 12345
```


