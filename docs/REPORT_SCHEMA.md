# Artifact schemas

All artifacts are PyTorch files. The bundled datasets use LZMA compression
(`.pt.xz`); the generators write new datasets as plain `{name}_00N.pt` with the next
free `N`. The tracked datasets are `_001`. Models, reports, and calibration artifacts
are plain `.pt`.

In the tables, `N` is the number of samples, `H` and `W` are spatial dimensions,
`R` is the number of regimes, and `P` is the number of λ values (nodes) per regime. The
pixel count is `n_pixels = H * W`. `r = GDF / n_pixels` is the normalized GDF, also
called capacity. Other terms are defined in [REFERENCE.md](REFERENCE.md#terms).

## Model checkpoint

`models/<model>/weights.pt`

| Key | Type | Meaning |
|---|---|---|
| `architecture` | `str` | `"SpatialGDFNet"` |
| `state_dict` | `dict` | PyTorch state dict |
| `in_phys_features` | `int` | Number of physical features, `2` |
| `channels` | `int` | Convolution channels, `32` |
| `num_blocks` | `int` | Residual blocks, `4` |
| `dilations` | `list[int]` | Dilation per block, `[1, 2, 4, 8]` |
| `training` | `dict` | Training metadata (see below) |

There is no feature-schema key. The feature order is fixed by
`features.extract_features`.

`training` is the metrics dictionary that `train` returns.

| Key | Type | Meaning |
|---|---|---|
| `seed` | `int \| None` | Training seed; `None` for an unseeded run |
| `data_file` | `str` | Training dataset path, repository-relative when inside the repository |
| `data_sha256` | `str` | SHA-256 of the training dataset file as read |
| `train_samples`, `val_samples` | `int` | Samples used per split |
| `max_samples` | `int \| None` | Per-split sample cap; `None` for full training |
| `epochs`, `batch_size` | `int` | Epochs and minibatch size |
| `lr`, `weight_decay`, `eta_min`, `grad_clip` | `float` | AdamW, cosine schedule, and clipping settings |
| `best_epoch` | `int` | 1-based epoch of the saved weights (lowest validation loss) |
| `best_val_loss` | `float` | Validation loss of the saved weights |
| `device` | `str` | Device type, for example `"cpu"`, `"cuda"`, or `"mps"` |
| `total_training_wall_time_seconds` | `float` | Wall time of the epoch loop |
| `time_per_epoch_ms` | `float` | Mean wall time per epoch |
| `created_at` | `str` | UTC timestamp, ISO 8601 |

## Per-sample dataset fields

The training splits and the stochastic test datasets store these per-sample fields.

| Key | Shape / type | Meaning |
|---|---|---|
| `observations_y` | `[N, 1, H, W]` float32 | Noisy observations |
| `x_clean` | `[N, 1, H, W]` float32 | Clean images |
| `dataset_names` | `list[str]` length `N` | Modality per sample |
| `img_indices` | `[N]` int32 | Source image index |
| `mc_gdf` | `[N]` float32 | Unclamped Monte Carlo GDF in pixels |
| `gdf_n` | `[N]` float32 | Unclamped normalized GDF `r = GDF / n_pixels` |
| `lambda_tv` | `[N]` float32 | TV regularization strength |
| `sigma` | `[N]` float32 | Noise standard deviation |
| `image_sizes` | `[N, 2]` int32 | Spatial size |
| `n_pixels` | `[N]` int32 | `H * W` |
| `sample_durations` | `[N]` float32 | Wall time per sample |
| `solve_times` | `[N]` float32 | Solver time per sample |
| `k_solvers` | `[N]` int32 | Solver budget `K = solver_k(L, η)` of each label |
| `gdf_n_se` | `[N]` float32 | 1-SE Monte Carlo uncertainty of `gdf_n` |

Generation stores the unclamped estimate. Training clamps both the train and the
validation targets to `[0, 1]`.

`provenance.solver_config` records `m_by_size`, the Monte Carlo probe count per image
size present in the dataset, and the solver-budget constants.

### `provenance.solver_config`

Both stochastic and sweeps datasets store the same keys:

| Key | Meaning |
|---|---|
| `algorithm` | `"Chambolle-Pock (PDHG 2011)"` |
| `stopping_rule` | `"calibrated fixed K, linear in log eta between tier values"` |
| `k_tiers` | `SOLVER_K_TIERS`, keyed by size |
| `nsr_tiers` | `DEFAULT_METROLOGY_NSR_TIERS` |
| `dominance_ratio` | `METROLOGY_DOMINANCE_RATIO` of the K calibration |
| `delta_k` | `DEFAULT_METROLOGY_DELTA_K` |
| `tau`, `sigma`, `theta` | Chambolle-Pock step sizes |
| `m_by_size` | Monte Carlo probe count per image size |
| `tau_mc` | `DEFAULT_TAU` |

## Training data

`data/generated/training_001.pt.xz`

Top-level keys: one dictionary per archive split of the workload (`train`, `val`), and
`provenance`.

Each split is a dictionary with the [per-sample dataset fields](#per-sample-dataset-fields).

`provenance` contains `n_samples`, `n_train`, `n_val`, `scales`, `seed`,
`solver_config`, `sampling_config`, `timing`, and `created_at`.

## Stochastic test datasets (single split)

`data/generated/test/{parity,scale_parity,val_parity}_001.pt.xz`

One flat dictionary with the [per-sample dataset fields](#per-sample-dataset-fields)
and `provenance`. The differences from the training splits:

- For mixed sizes (`scale_parity`), `observations_y` and `x_clean` are lists of
  `[1, H, W]` float32 tensors, one per sample.
- `provenance` contains `test_dataset_name`, `split`, `n_samples`, `scales`,
  `sampling_config`, `solver_config`, `timing`, and `created_at`.

## Sweeps test datasets

`data/generated/test/{sweeps,scale_sweeps}_001.pt.xz`

| Key | Shape / type | Meaning |
|---|---|---|
| `observations_y` | `[R, 1, H, W]` float32, or a list of `[1, 1, H, W]` for mixed sizes | One observation per regime |
| `x_clean` | `[R, 1, H, W]` float32, or a list of `[1, 1, H, W]` for mixed sizes | Clean image per regime |
| `gdf_n` | `[R, P]` float32 | Unclamped normalized GDF `r = GDF / n_pixels` per λ value |
| `gdf_n_se` | `[R, P]` float32 | 1-SE uncertainty of `gdf_n` per λ value |
| `lambdas` | `[R, P]` float32 | λ grid, log-spaced in λ / σ_MAD |
| `ssrs` | `[R, P]` float32 | Residual sum of squares |
| `solve_times` | `[R, P]` float32 | Solver time per λ value |
| `sigmas` | `[R]` float32 | True noise standard deviation σ per regime |
| `sigma_mads` | `[R]` float32 | Wavelet MAD noise estimate σ_MAD per regime |
| `n_pixels` | `[R]` int32 | `H * W` per regime |
| `mc_gdf_baseline` | `[R]` float32 | Baseline GDF in pixels, equal to `n_pixels` |
| `k_solvers` | `[R]` int32 | Solver budget `K = solver_k(L, η)` per regime; the same at every λ value |
| `provenance` | `dict` | Dataset metadata |

`lambdas[r]` equals the float32 `torch.logspace` over
log10(λ/σ_MAD) between `DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MIN` and
`DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MAX`, scaled by `sigma_mads[r]`, exactly.

`provenance` contains `test_dataset_name`, `sweep_points`, `n_points_per_regime`,
`n_regimes`, `scales`, `m_by_size`, `tau`, `shapes`, `regimes`, `solver_config`,
`timing`, and `created_at`. Each `regimes` entry stores `title`, `img_idx`, `nsr`, and
`shape`.

## Parity reports

`models/<model>/{parity,val_parity,scale_parity}.pt`

Every array field is a CPU `torch.Tensor`. Float fields are float32.

| Key | Type | Meaning |
|---|---|---|
| `suite_name` | `str` | Benchmark suite name |
| `model` | `str` | Model directory name |
| `evaluated_at` | `str` | ISO timestamp |
| `n_samples` | `int` | Number of evaluated samples |
| `summary_metrics` | `dict` | Aggregate metrics |
| `dataset_names` | `list[str]` | Modality per sample |
| `true_r` | `[N]` float32 | Monte Carlo `r` |
| `pred_r` | `[N]` float32 | Surrogate `r` |
| `sigmas` | `[N]` float32 | Noise standard deviation |
| `n_pixels` | `[N]` int32 | `H * W` |
| `sample_indices` | `[N]` int64 | `0 .. N-1`, in the order of the source datasets |
| `split` | `str` | `test` or `validation` |
| `test_data` | `str` or `list[str]` | Source `.pt` or `.pt.xz` paths |
| `source_datasets` | `list[dict]` | Source dataset names and paths |

`summary_metrics` contains:

- `r2`: coefficient of determination of `pred_r` against `true_r`
- `mae_pct`: `100 · mean|true_r - pred_r|`, in percentage points of `r`
- `rmse`: root mean squared error of `r`
- `max_error`: `100 · max|true_r - pred_r|`, in percentage points of `r`
- `by_scale_modality`: `{scale: {modality: {r2, mae_pct}}}`, with `scale` as `"64x64"`
- `by_modality_nsr`: `{modality: {nsr_range}}`, where `nsr_range` is `[min, max]` of
  `η = σ / std(x_clean)`
- `overall_nsr`: `{nsr_range}` over all samples
- `by_scale`: `{scale: {r2, mae_pct}}`, only in `scale_parity`

## Sweep reports

`models/<model>/{sweeps,scale_sweeps}.pt`

Top-level keys:

| Key | Type | Meaning |
|---|---|---|
| `suite_name` | `str` | Benchmark suite name |
| `model` | `str` | Model directory name |
| `evaluated_at` | `str` | ISO timestamp |
| `summary_metrics` | `dict` | Aggregate metrics |
| `regimes` | `list[dict]` | Per-regime records |
| `split` | `str` | Always `test` |
| `test_data` | `str` | Source `.pt` or `.pt.xz` path |
| `source_datasets` | `list[dict]` | Source dataset names and paths |
| `selection_excess` | `dict` | Selection-excess statistics |

`summary_metrics` contains:

- `mean_r2`: mean of the per-regime `r2`
- `mean_mae_pct`: mean of the per-regime `mae_pct` (percentage points of `r`)
- `n_regimes`

Array fields are CPU `torch.Tensor` objects. Float arrays are float32. `lambdas`,
`surrogate_gdf_norm`, and `anchor_gdf_norm` are float32 and bitwise equal to the values
stored in the test dataset. `summary_metrics` values, `sigma`, `sigma_mad`, `r2`, and
`mae_pct` are Python floats.

Each regime contains:

| Key | Shape / type | Meaning |
|---|---|---|
| `regime_index` | `int` | Regime number |
| `title` | `str` | Panel title |
| `modality` | `str` | Source modality |
| `sigma` | `float` | Noise standard deviation |
| `sigma_mad` | `float` | Wavelet MAD noise estimate σ_MAD |
| `r2`, `mae_pct` | `float` | Curve metrics |
| `n_pixels` | `int` | Image pixels |
| `lambdas` | `[P]` float32 | Regularization grid |
| `mc_gdf`, `mc_gdf_norm`, `mc_gdf_norm_err` | `[P]` float32 | Unclamped Monte Carlo GDF in pixels, `r`, and 1-SE of `r` |
| `surrogate_gdf_norm` | `[P]` float32 | Surrogate `r` curve |
| `anchor_gdf_norm` | `[P]` float32 | Tikhonov anchor |
| `mc_sure`, `mc_delta_sure`, `surrogate_sure` | `[P]` float32 | SURE curves |
| `mc_sure_mad`, `mc_delta_sure_mad`, `surrogate_sure_mad` | `[P]` float32 | MAD SURE curves |
| `mc_morozov_ratio`, `surrogate_morozov_ratio` | `[P]` float32 | MDP ratios with the true σ |
| `mc_delta_morozov` | `[P]` float32 | MDP ratio uncertainty with the true σ |
| `mc_morozov_ratio_mad`, `surrogate_morozov_ratio_mad` | `[P]` float32 | MDP ratios with σ_MAD |
| `mc_delta_morozov_mad` | `[P]` float32 | MDP ratio uncertainty with σ_MAD |
| `selected_lambda` | `dict[str, float]` | Selected lambda per rule: `mc_`/`surrogate_` × `sure`, `sure_mad`, `morozov`, `morozov_mad` |
| `selection_excess` | `dict[str, dict]` | Selection-excess cases for this regime, keyed `SURE` and `MDP`; each value is a case as below |

`mae_pct` is `100 · mean|mc_gdf_norm - surrogate_gdf_norm|`, computed as
`compute_regression_metrics(...)["mae"] * 100`.

### `selection_excess`

The selection excess compares the λ node that the surrogate selects with the node
that the Monte Carlo reference selects. Both are scored on the reference criterion.
SURE cases use the MAD SURE curve `mc_sure_mad`. MDP cases use `|mc_morozov_ratio -
SELECTION_MDP_TARGET|`. Derivations are in the accompanying publication.

Top-level fields:

- `statistic`: text description of the reported statistic `excess_z`
- `model`: text description of the error model
- `conditioning`: text description of what the statistic conditions on
- `mad_variance_factor`: `WAVELET_MAD_VARIANCE_FACTOR`
- `quadrature`: `absolute_tolerance`, `relative_tolerance`, `subdivision_limit`, and
  `refinement_factor` of the MDP variance integral, from the `SELECTION_QUADRATURE_*`
  constants
- `verification_absolute_errors`: absolute error of each analytical limit check of the
  folded-Gaussian model, keyed by check name
- `max_refinement_relative_excess_se_change`: maximum of
  `refinement_relative_excess_se_change` over all cases
- `summaries`: one summary per criterion and size, as below
- `by_resolution`: the same summaries, keyed `{str(size): {criterion: summary}}`
- `cases`: one case per regime and criterion, as below

Each case contains:

- `criterion`: `SURE` or `MDP`
- `size`: image side `L`
- `regime_index`: regime number
- `title`: panel title
- `surrogate_index`: λ node selected by the surrogate curve
- `reference_index`: λ node selected by the Monte Carlo curve
- `same_node`: `surrogate_index == reference_index`
- `excess`: reference criterion at the surrogate node minus reference criterion at the
  reference node; `0.0` when `same_node`
- `excess_se`: 1-SE uncertainty of `excess`
- `excess_z`: `excess / excess_se`; `0.0` when `excess` is zero, infinite when
  `excess > 0` and `excess_se` is zero
- `reference_criterion_at_surrogate_selection`: reference criterion at
  `surrogate_index`
- `reference_criterion_at_reference_selection`: reference criterion at
  `reference_index`
- `quadrature_error_estimate`: absolute error estimate of the MDP variance integral;
  `0.0` for SURE
- `refinement_relative_excess_se_change`: relative change of `excess_se` when the
  quadrature tolerances are divided by `SELECTION_QUADRATURE_REFINEMENT_FACTOR`; `0.0`
  for SURE
- `linear_excess_se`: first-order (linearized) `excess_se` of the MDP case; `0.0` for
  SURE
- `ratio_error_correlation`: correlation of the two MDP ratio errors; `None` for SURE
  and for `same_node`

Each summary contains:

- `criterion`: `SURE` or `MDP`
- `size`: image side `L`
- `case_count`: number of cases
- `same_node_matches`: number of cases with `same_node`
- `different_node_count`: `case_count - same_node_matches`
- `different_node_median_excess_z`: median `excess_z` over the different-node cases
- `different_node_max_excess_z`: maximum `excess_z` over the different-node cases

`different_node_median_excess_z` is computed with `statistics.median`. It is a Python
float or `None`. Both `different_node_*_excess_z` fields are `None` when
`different_node_count` is 0.

## Calibration artifacts

### `data/calibration/e_k_table.pt`

The precomputed flat-region solver error, written by `generate_flat_error_table`.
Bound to `DEFAULT_CP_SIGMA`, `DEFAULT_CP_TAU`, and `DEFAULT_CP_THETA` in `config.py`.

- `tables`: dict keyed by spatial size (int). Each value is a float32 tensor of
  length `k_max`; entry `K - 1` is e(K, L) for K = 1..`k_max`.
- `metadata`: `created_at`, `sizes`, `k_min`, `k_max`, `sigma`, `tau`, and `theta`.

The table is self-describing: `k_max` sets its length. `calibrate_k` raises
`ValueError` if the table is shorter than `K_CAL_MAX_K`. The loader rejects a table
whose `sigma`, `tau`, or `theta` differs from `DEFAULT_CP_SIGMA`,
`DEFAULT_CP_TAU`, or `DEFAULT_CP_THETA`. Regenerate it after changing those values,
`E_K_TABLE_K_MAX`, or the tabulated sizes.

### `data/calibration/k_calibration.pt`

Written by `calibrate_k`. Top-level keys:

- `records`
- `k_tiers`: dict keyed by size; each value holds the largest `k_star` per noise tier,
  in `nsr_tiers` order. `SOLVER_K_TIERS` in `config.py` is copied from it.
- `metadata`

Each record is one configuration (image and noise tier):

| Key | Meaning |
|---|---|
| `size` | Spatial resolution |
| `modality` | Source modality |
| `split` | Archive split of the image (`val`) |
| `image_idx` | Index within that split |
| `nsr` | Noise-to-signal ratio `η = sigma / std_x` |
| `std_x` | Population standard deviation of the clean image |
| `sigma` | Noise standard deviation |
| `sigma_mad` | Wavelet MAD noise estimate σ_MAD, used as λ |
| `k_star` | First checkpoint with `e(K, L) + \|E_τ\| <= METROLOGY_DOMINANCE_RATIO · SE_prod` |
| `gdf_n` | Unclamped float32 mean probe divergence at `k_star` |
| `s_probe` | Sample standard deviation (`ddof=1`) of the float32 probe divergences at `k_star` |
| `se_prod` | `s_probe / sqrt(DEFAULT_M_BY_SIZE[size])` |
| `e_k` | `e(k_star, size)` from `e_k_table.pt` |
| `e_tau` | Probe mean of the float32 minus the float64 divergences at `k_star` |
| `seconds` | Wall time of the configuration |

`metadata` contains `images`, `nsr_tiers`, `m_probes`, `tau`, `tau_ref`, `ratio`,
`delta_k`, `m_by_size`, `n_records`, `elapsed_seconds`, and `created_at`. Observation
noise and probe draws are unseeded, so each run draws them anew. The bundled file also
has `metadata["note"]` and NaN timings; see
[Bundled artifact provenance](REFERENCE.md#bundled-artifact-provenance).
