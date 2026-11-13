# Code reference

The package root `gdfnet_tvdenoise` exports five names: `predict_gdf`,
`SpatialGDFNet`, `extract_features`, `compute_sure`, and `compute_mdp`. Import all
other functions from their module. The [module map](#module-map) lists the modules,
and each entry below names its module in the import line. Artifact schemas are in
[REPORT_SCHEMA.md](REPORT_SCHEMA.md). The end-to-end pipeline is in the
[README](../README.md#pipeline).

## Module map

Modules are under `gdfnet_tvdenoise.`. The table lists the `__all__` names of each
module.

| Module | Public names |
|---|---|
| `config` | Constants only. See [Configuration](#configuration-configpy). |
| `inference` | `predict_gdf`, `predict`, `compute_sure`, `compute_mdp` |
| `features` | `extract_features`, `image_std` |
| `model` | `SpatialGDFNet`, `ModelNotFoundError`, `FiLMGenerator`, `SpatialResBlock` |
| `solver` | `tv_denoise`, `tv_denoise_iterates`, `solver_k` |
| `data` | `fetch_data`, `load_medmnist_images`, `prepare_archives`, `generate_stochastic`, `generate_sweeps`, `create_physics_operator`, `get_medmnist_split_size` |
| `calibration` | `generate_flat_error_table`, `calibrate_k` |
| `train` | `train` |
| `benchmarks` | `run_benchmark`, `plot_benchmark`, `evaluate_parity`, `evaluate_sweeps`, `SUITES`; `benchmarks.plots.plot_overview` |
| `helpers` | Serialization, Monte Carlo GDF at a fixed K (`probe_divergences`, `compute_mc_gdf`, `McGdfResult`), flat solver error, noise estimation, uncertainty, and selection-excess math. See its `__all__`. |
| `cli` | `build_parser`, `main` |

`compute_sure` and `compute_mdp` are defined in `helpers` and re-exported by
`inference` and the package root.

## Shape notation

- `B`: batch size, the number of images processed together.
- `H`, `W`: image height and width.
- `N`: number of samples in data and report tables. The pixel count is
  `n_pixels = H * W`.
- `R`, `P`: number of regimes and number of λ values per regime in sweeps data.

## Terms

- `n_pixels = H * W`: pixel count of one image.
- `L`: image side in pixels. Images are square, so `H = W = L` and `n_pixels = L²`.
- GDF: generalized degrees of freedom of the TV denoiser, in pixels (`mc_gdf`).
- `r`: normalized GDF, `r = GDF / n_pixels`, nominally in `[0, 1]`. The Monte Carlo
  estimate stored as `gdf_n` is unclamped, so probe noise can place it outside the
  range; training clamps the targets to `[0, 1]`. Reports and code also call `r` the
  capacity. Fields that store `r` are `gdf_n` (datasets, calibration), `true_r` and
  `pred_r` (parity reports), and `mc_gdf_norm` and `surrogate_gdf_norm` (sweep reports).
- `η`: noise-to-signal ratio (NSR), `η = σ / std(x_clean)`. The code computes `std`
  as the population standard deviation (ddof=0).
- `R`: in shape brackets such as `[R, P]`, the regime count.
- `K`: Chambolle-Pock iteration count.
- `e(K, L)`: flat-region solver error at `K` iterations and side `L`, from
  `e_k_table.pt`.
- `E_τ`: the finite-difference error of the float32 Monte Carlo divergence at step
  `DEFAULT_TAU`, measured in the K calibration against a float64 reference at
  `K_CAL_TAU_REF`.
- `SE_prod`: the production standard error `ŝ / sqrt(M_prod)`, with `ŝ` the sample SD of
  the probe divergences and `M_prod = DEFAULT_M_BY_SIZE[L]`.
- `K*`: the calibrated budget of one calibration configuration, the first checkpoint
  with `e(K, L) + |E_τ| <= METROLOGY_DOMINANCE_RATIO · SE_prod`.
- solver budget: every label runs `K = solver.solver_k(L, η)` iterations, linear in
  `log η` between the tier values `SOLVER_K_TIERS[L]`, the largest `K*` per size and
  noise tier.
- `σ_MAD`: wavelet MAD estimate of σ.

## Configuration (`config.py`)

`gdfnet_tvdenoise/config.py` is the single source of truth for constants, paths,
defaults, and workloads. Every module imports from it. CLI defaults read from it.
The package reads no config files and no environment variables. To change a
setting, edit `config.py`.

### Sections

| § | Section | Key names | Main readers |
|---|---|---|---|
| 0 | Device | `DEFAULT_DEVICE` | every function with a `device` argument, CLI `--device` |
| 1 | Problem domain | `DEFAULT_IMAGE_SIZE`, `CANONICAL_MEDMNIST_DATASETS`, `CANONICAL_MEDMNIST_SIZES`, `MEDMNIST_PLUS_DATASETS` | `data`, `calibration`, `benchmarks`, `cli` |
| 2 | Noise and regularization sampling | `DEFAULT_NSR_MIN`, `DEFAULT_NSR_MAX`, `DEFAULT_LAMBDA_MIN`, `DEFAULT_LAMBDA_MAX`, `DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MIN`, `DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MAX`, `CALIBRATED_DENOISE_LOGIT_X0`, `CALIBRATED_DENOISE_LOGIT_W`, `LOG_UNIFORM_SAMPLING`, `BALANCED_SCATTER_SAMPLING` | `data` (directly and through the workloads) |
| 3 | Solver | `DEFAULT_CP_TAU`, `DEFAULT_CP_SIGMA`, `DEFAULT_CP_THETA`, `SOLVER_K_TIERS` | `solver`, `helpers`, `data`, `calibration` |
| 4 | Monte Carlo GDF and metrology | `DEFAULT_M_BY_SIZE`, `DEFAULT_TAU`, `METROLOGY_DOMINANCE_RATIO`, `SELECTION_*`, `LITS_*` build constants, `WAVELET_MAD_*`, `DEFAULT_WAVELET_MAD_*` | `helpers`, `data`, `calibration`, `benchmarks` |
| 5 | Calibration process | `CALIBRATION_IMAGES`, `DEFAULT_METROLOGY_NSR_TIERS`, `DEFAULT_METROLOGY_DELTA_K`, `K_CAL_PROBES`, `K_CAL_TAU_REF`, `K_CAL_MAX_K`, `E_K_TABLE_K_MAX` | `calibration`, `solver`, `cli` |
| 6 | Features and architecture | `FEATURE_TIKHONOV_DENOM_FLOOR`, `PREDICT_CHUNK_SIZE`, `MDP_DENOMINATOR_FLOOR`, `MODEL_CHANNELS`, `MODEL_NUM_BLOCKS`, `MODEL_DILATIONS` | `features`, `inference`, `helpers`, `train` |
| 7 | Training defaults | `DEFAULT_TRAIN_*`, `TRAIN_BCE_EPS`, `TRAIN_TARGET_MIN`, `TRAIN_TARGET_MAX` | `train`, `cli` |
| 8 | Paths | `REPO_ROOT`, `*_DIR`, `*_PATH`, `DEFAULT_DATASETS`, `CHUNK_SIZE_BYTES`, `DEFAULT_MAX_RETRIES`, `GENERATION_CHECKPOINT_*` | `data`, `calibration`, `train`, `inference`, `helpers`, `benchmarks` |
| 9 | Benchmark regimes and workloads | `CANONICAL_SWEEP_NSR_TIERS`, `CANONICAL_SWEEP_IMAGES`, `CANONICAL_12PANEL_REGIMES`, `DEFAULT_*_WORKLOAD`, `BENCHMARK_SUITES` | `data`, `cli`, `benchmarks` |

### What you can change

Each constant belongs to one of four classes.

**Runtime only.** A change does not affect the bundled files or the values of new
artifacts.

- `DEFAULT_DEVICE`. Results on an accelerator can differ from CPU results in the last
  bits.
- `PREDICT_CHUNK_SIZE`, `CHUNK_SIZE_BYTES`, `DEFAULT_MAX_RETRIES`,
  `GENERATION_CHECKPOINT_SUFFIX`, `GENERATION_CHECKPOINT_INTERVAL_SECONDS`.
- Paths: `REPO_ROOT`, `DATA_DIR`, `ARCHIVE_DIR`, `LITS_RAW_DIR`, `LITS_SPLIT_PATH`,
  `GENERATED_DIR`, `GENERATED_TEST_DIR`, `CALIBRATION_DIR`, `MODELS_DIR`,
  `K_CALIBRATION_PATH`, `E_K_TABLE_PATH`. Move the bundled files when you move their path.

**Input selection.** A change selects other existing files.

- `DEFAULT_DATASETS`: the dataset files that `train` and each benchmark suite read.
- `PUBLISHED_MODELS`: the bundled model directories, one per training seed.
- `DEFAULT_MODEL`: the declared median of `PUBLISHED_MODELS` by validation MAE. It must be
  one of them.
- `DEFAULT_MODEL_PATH`: the checkpoint that `predict_gdf(model=None)` and CLI
  `infer` load. It is derived from `DEFAULT_MODEL`.
- `OVERVIEW_SUITE`, `OVERVIEW_REGIME_INDEX`, `OVERVIEW_PATH_NODE_OFFSET`,
  `OVERVIEW_FIGURE_PATH`: the regime, examples, and output file of `plot_overview`.

**New artifacts only.** The bundled files stay valid. New runs differ from them.

- Workloads: `DEFAULT_TRAINING_WORKLOAD`, `DEFAULT_PARITY_WORKLOAD`,
  `DEFAULT_SCALE_PARITY_WORKLOAD`, `DEFAULT_VAL_PARITY_WORKLOAD`,
  `DEFAULT_SWEEPS_WORKLOAD`, `DEFAULT_SCALE_SWEEPS_WORKLOAD`, and
  `DEFAULT_IMAGE_SIZE`.
- Sampling: `LOG_UNIFORM_SAMPLING`, `BALANCED_SCATTER_SAMPLING`,
  `CALIBRATED_DENOISE_LOGIT_X0`, `CALIBRATED_DENOISE_LOGIT_W`, `DEFAULT_NSR_MIN`,
  `DEFAULT_NSR_MAX`, `DEFAULT_LAMBDA_MIN`, `DEFAULT_LAMBDA_MAX`,
  `DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MIN`, `DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MAX`.
- Calibration process: `K_CAL_PROBES`, `K_CAL_TAU_REF`, `K_CAL_MAX_K`. Regenerate
  `e_k_table.pt` after a change to `E_K_TABLE_K_MAX`; it must cover `K_CAL_MAX_K`.
- Training: `DEFAULT_TRAIN_EPOCHS`, `DEFAULT_TRAIN_LR`, `DEFAULT_TRAIN_BATCH_SIZE`,
  `DEFAULT_TRAIN_WEIGHT_DECAY`, `DEFAULT_TRAIN_ETA_MIN`, `DEFAULT_TRAIN_GRAD_CLIP`,
  `DEFAULT_TRAIN_SEED`, `TRAIN_BCE_EPS`, `TRAIN_TARGET_MIN`, `TRAIN_TARGET_MAX`.
- Architecture: `MODEL_CHANNELS`, `MODEL_NUM_BLOCKS`, `MODEL_DILATIONS`. Only `train`
  reads them. `SpatialGDFNet.load` reads the architecture from the checkpoint.

**Bound to bundled artifacts.** The bundled files depend on these values. A change
requires you to regenerate the listed artifacts.

| Constants | Bound artifacts | Effect of a change |
|---|---|---|
| `DEFAULT_CP_TAU`, `DEFAULT_CP_SIGMA`, `DEFAULT_CP_THETA` | `e_k_table.pt`, and through it all calibration artifacts, datasets, and models | The `e_k_table.pt` loader raises `ValueError` until you run `calibrate --which table`. Every label changes. |
| `SOLVER_K_TIERS` | `k_calibration.pt`; `k_solvers` of every dataset | Copied from the `calibrate_k` output (see [K calibration](#k-calibration)). New datasets use the new budgets. |
| `DEFAULT_TAU`, `DEFAULT_M_BY_SIZE`, `METROLOGY_DOMINANCE_RATIO`, `DEFAULT_METROLOGY_NSR_TIERS`, `DEFAULT_METROLOGY_DELTA_K`, `CALIBRATION_IMAGES` | `k_calibration.pt`, `SOLVER_K_TIERS`, and every dataset | Change E_τ, SE_prod, the stop test, or the measured grid. Rerun `calibrate --which k`, copy `SOLVER_K_TIERS`, and regenerate the datasets. A resumed calibration fails fast against its checkpoint signature. |
| `CANONICAL_SWEEP_NSR_TIERS`, `CANONICAL_SWEEP_IMAGES`, `CANONICAL_12PANEL_REGIMES` | `sweeps_001`, `scale_sweeps_001`, sweep reports | Define the sweep regimes. |
| `LITS_HU_MIN`, `LITS_HU_MAX`, `LITS_CROP_MARGIN`, `LITS_MAX_AIR_FRACTION`, `LITS_AIR_HU_MAX`, `LITS_AIR_SHIFT_PX`, `LITS_AIR_SHIFT_STEP_PX`, `LITS_HF_REPO`, `LITS_HF_REVISION`, `LITS_SPLITS` | LiTS archives, `data/lits_split.pt`, and every LiTS index in the datasets, `CALIBRATION_IMAGES`, and `CANONICAL_SWEEP_IMAGES` | The archives differ from the archives of the bundled files. `prepare_archives` can fail against the manifest. |
| `CANONICAL_MEDMNIST_DATASETS`, `CANONICAL_MEDMNIST_SIZES`, `MEDMNIST_PLUS_DATASETS` | archives, `e_k_table.pt` sizes, all datasets | Define the modality and size grid of every bundled file. |
| `FEATURE_TIKHONOV_DENOM_FLOOR` | model checkpoints | Changes the feature values that the checkpoints were trained on. |
| `MDP_DENOMINATOR_FLOOR` | sweep reports | Changes the MDP values. |
| `SELECTION_*`, `DEFAULT_WAVELET_MAD_N_COEFFICIENTS`, `DEFAULT_WAVELET_MAD_SAMPLING_SEED` | sweep reports | Change σ_MAD, the MAD curves, and the `selection_excess` values. |

Do not edit `STANDARD_NORMAL_DENSITY_AT_ZERO`, `WAVELET_MAD_NORMAL_QUARTILE`, or
`WAVELET_MAD_VARIANCE_FACTOR`. They are mathematical constants. `BENCHMARK_SUITES`
must equal the keys of `benchmarks.SUITES`. The import raises `RuntimeError`
otherwise.

### Workload keys

`DEFAULT_DATASETS` lists the files that are read. Workloads define the files that are
generated. There are two kinds of dataset:

- **Stochastic** (`training`, `parity`, `scale_parity`, `val_parity`): every sample
  draws a modality, an image, a relative noise level, and a λ at random. The workloads
  are `DEFAULT_TRAINING_WORKLOAD`, `DEFAULT_PARITY_WORKLOAD`,
  `DEFAULT_SCALE_PARITY_WORKLOAD`, and `DEFAULT_VAL_PARITY_WORKLOAD`.
- **Sweeps** (`sweeps`, `scale_sweeps`): fixed regimes (image and noise level), each
  evaluated on a fixed grid of λ values. The workloads are `DEFAULT_SWEEPS_WORKLOAD`
  and `DEFAULT_SCALE_SWEEPS_WORKLOAD`.

Stochastic workload keys:

| Key | Type | Meaning |
|---|---|---|
| `name` | `str` | Output stem: the generator writes `{name}_00N.pt` |
| `output_dir` | `Path` | Output directory. Optional; without it, pass `output_dir` explicitly |
| `sizes` | `list[int]` | Image sizes |
| `sampling_config` | `dict` | `LOG_UNIFORM_SAMPLING` or `BALANCED_SCATTER_SAMPLING` |
| `splits` | `dict[str, int]` | Samples per archive split and per image size, in order, for example `{"train": 4000, "val": 700}` for `training` and `{"test": 100}` for `scale_parity` (100 at each of its two sizes, 200 total) |

Sweeps workload keys:

| Key | Type | Meaning |
|---|---|---|
| `name` | `str` | Output stem |
| `output_dir` | `Path` | Output directory, with the same rule as above |
| `scales` | `list[int]` | Image sizes. Each regime runs at each size. |
| `sweep_points` | `int` | λ values per regime |
| `regimes` | `list[dict]` | Regimes with `title`, `img_tag`, `img_idx` (default 0), `nsr` (default 0.50), and an optional `split` (default `test`) |

Sampling-config keys. Both configs draw η uniformly from `[nsr_min, nsr_max]` and set
`σ = η · std(x_clean)`.

| Config | `lambda_dist` | Keys | λ draw |
|---|---|---|---|
| `log_uniform` | `uniform` | `name`, `nsr_min`, `nsr_max`, `lambda_dist`, `lambda_min`, `lambda_max` | `log10(λ)` uniform in `[log10(lambda_min), log10(lambda_max)]` |
| `balanced_scatter` | `regime_logit` | `name`, `nsr_min`, `nsr_max`, `lambda_dist`, `u_min`, `u_max`, `denoise_x0`, `denoise_w` | `u` uniform in `[u_min, u_max]`, `log10(λ) = denoise_x0 + denoise_w · log10((1 - u) / u)` |

## Inference

### `predict_gdf(y, lambda_tv, model=None)`

Returns the normalized GDF `r = GDF / n_pixels` as a CPU tensor of shape `[1]`.

```python
from gdfnet_tvdenoise import predict_gdf

r = predict_gdf(y, 0.05)
r = predict_gdf(y, 0.05, "models/model_002/weights.pt")
r = predict_gdf(y, 0.05, model)
```

- `y` is a tensor or NumPy array of shape `[H, W]`, `[1, H, W]`, or `[1, 1, H, W]`.
  One image only. Other shapes raise `ValueError`.
- `lambda_tv` may be a scalar or a one-element tensor.
- `model` may be `None`, a `SpatialGDFNet`, a checkpoint path, or a model directory.
- `model=None` loads `models/model_003/weights.pt` (`config.DEFAULT_MODEL_PATH`).
- Feature extraction happens inside `predict_gdf`.
- The output is finite and in `[0, 1]`.

### `SpatialGDFNet.load(path, device=DEFAULT_DEVICE)`

Loads a published checkpoint onto `device`, which defaults to `DEFAULT_DEVICE` in `config.py`.

```python
from gdfnet_tvdenoise import SpatialGDFNet

model = SpatialGDFNet.load("models/model_003/weights.pt")
model = SpatialGDFNet.load("models/model_003")
```

- `path` may be a checkpoint file or a directory containing `weights.pt`.
- Loading is strict. The checkpoint must contain the keys listed in
  [REPORT_SCHEMA.md](REPORT_SCHEMA.md#model-checkpoint), and the state dict must match
  the architecture.
- The architecture comes from the checkpoint, not from the `MODEL_*` constants.
- The returned model is in evaluation mode.
- Raises `ModelNotFoundError` if the file does not exist.

### `extract_features(y, lambda_tv)`

Returns the two physical features used by the published checkpoints. It lives in
`gdfnet_tvdenoise.features` and is re-exported by the package.

```python
from gdfnet_tvdenoise import extract_features

features = extract_features(y, 0.05)
```

- `y` is a `torch.Tensor` of shape `[B, 1, H, W]`. Other shapes raise `ValueError`.
- `lambda_tv` is a scalar shared by the batch, or a `[B]` tensor. Other sizes raise
  `ValueError`.
- The output has shape `[B, 2]` and the dtype and device of `y`.

The feature order is fixed by `extract_features`:

1. `log_lam_raw`: `log10(lambda_tv)`
2. `tikhonov_capacity_normalized`: the trace of the Tikhonov smoother with the same
   `lambda_tv`, divided by `n_pixels`

This order defines the checkpoint input dimension.

### `predict(model, y, lambdas, chunk_size=16)`

Predicts the normalized GDF for a batch. It lives in `gdfnet_tvdenoise.inference`.

```python
import torch
from gdfnet_tvdenoise import SpatialGDFNet
from gdfnet_tvdenoise.inference import predict

y = torch.rand(4, 1, 64, 64)
model = SpatialGDFNet.load("models/model_003/weights.pt")
r = predict(model, y, torch.full((4,), 0.05))  # [4], CPU
```

- `y` has shape `[B, 1, H, W]`. `lambdas` is a scalar or a `[B]` tensor.
- Features and the network run on the model's device, in chunks of `chunk_size` images
  (`config.PREDICT_CHUNK_SIZE`).
- The keyword `feats` takes a precomputed `[B, 2]` feature tensor and skips
  `extract_features`.
- `predict_gdf` and both benchmark families call this function.

### `model.forward(y, features)`

Runs the surrogate directly. `SpatialGDFNet` lives in `gdfnet_tvdenoise.model`.

```python
features = extract_features(y, 0.05)  # [B, 2]
r = model(y, features)  # [B]
```

| `y` | `features` | Output |
|---|---|---|
| `[B, 1, H, W]` | `[B, 2]` | `[B]` |

The image has one channel because it is grayscale. `H` and `W` may vary because
the network is fully convolutional.

### `compute_sure(mse, gdf_n, sigma)`

Returns `mse - sigma**2 + 2 * sigma**2 * gdf_n`.

```python
from gdfnet_tvdenoise import compute_sure

sure = compute_sure(mse, float(r), sigma)
```

`mse` and `gdf_n` are scalars, NumPy arrays, or tensors. The result is a float32 tensor
(0-d for scalar inputs). Use `float(...)` to get a Python float. The function lives in
`helpers.py` and is re-exported by the package.

### `compute_mdp(mse, gdf_n, sigma)`

Returns `mse / ((1 - gdf_n) * sigma**2)`.

```python
from gdfnet_tvdenoise import compute_mdp

mdp = compute_mdp(mse, float(r), sigma)
```

- The denominator uses `1 - gdf_n >= config.MDP_DENOMINATOR_FLOOR` (1e-6).
- `mse` and `gdf_n` are scalars, NumPy arrays, or tensors. The result is a float32
  tensor. The function lives in `helpers.py` and is re-exported by the package.
- `sigma` must be finite and positive. Otherwise the function raises `ValueError`.

## Solver

### `tv_denoise(y, lambda_tv, k)`

Runs `k` fixed Chambolle-Pock iterations of 2D TV denoising with the identity
forward operator. The step sizes are `DEFAULT_CP_TAU`, `DEFAULT_CP_SIGMA`, and
`DEFAULT_CP_THETA`.

```python
from gdfnet_tvdenoise.solver import tv_denoise

x = tv_denoise(y, 0.05, 300)  # y: [B, 1, H, W]
```

`lambda_tv` must be positive. Otherwise the function raises `ValueError`.
`compute_mc_gdf` calls it for the Monte Carlo capacity labels.

### `tv_denoise_iterates(y, lambda_tv, k)`

Yields the estimate after each of `k` fixed Chambolle-Pock iterations, one
`[B, 1, H, W]` tensor per iteration. It is the same recursion and the same starting
state as `tv_denoise`, with the state carried across iterations, so one pass over the
generator is one continuous solve. `tv_denoise(y, lambda_tv, k)` returns its last
value. `calibrate_k` iterates it to read the estimate at each
`DEFAULT_METROLOGY_DELTA_K` checkpoint.

```python
from gdfnet_tvdenoise.solver import tv_denoise_iterates

for it, x in enumerate(tv_denoise_iterates(y, 0.05, 300), start=1):
	...
```

### `solver_k(size, nsr)`

Returns the calibrated solver budget `K(L, η)` of a label. K is linear in `log η`
between the tier values `SOLVER_K_TIERS[size]` at `DEFAULT_METROLOGY_NSR_TIERS`,
rounded up to a multiple of `DEFAULT_METROLOGY_DELTA_K`. It raises `ValueError` for a
size without tier values or for `nsr` outside the tier range.

```python
from gdfnet_tvdenoise.solver import solver_k

k = solver_k(64, 0.25)  # 1450
```

The budget does not depend on λ. It is calibrated at λ = σ_MAD.

### `compute_mc_gdf(y, lambda_tv, m_mc, k, tau=DEFAULT_TAU, rng=None)`

Computes the Monte Carlo normalized GDF at a fixed solver budget `k`. It lives in
`gdfnet_tvdenoise.helpers`. Dataset generation uses it for every capacity label, with
`k = solver_k(L, η)`.

```python
from gdfnet_tvdenoise.helpers import compute_mc_gdf

result = compute_mc_gdf(y, 0.05, m_mc=60, k=1450)
```

`y` has shape `[1, 1, H, W]` with `H == W`. `lambda_tv` must be positive. The function
draws `m_mc` Gaussian probes b and runs one joint Chambolle-Pock solve of `k`
iterations on the batch `[y; y + tau · b]`. The divergence of probe i is
`mean(b_i · (x_i − x_0) / tau)`, from `helpers.probe_divergences`. This is the
estimator of deepinv `mc_div`, evaluated on all probes at once; `calibrate_k` uses the
same function.

It returns a `McGdfResult` with:

| Field | Meaning |
|---|---|
| `gdf_n` | Mean probe divergence at K, unclamped; probe noise can place it outside `[0, 1]` |
| `gdf_n_se` | Measured `s / sqrt(m_mc)`; `0.0` if `m_mc == 1` |
| `x_hat` | Reconstruction of `y` at K, shape `[1, 1, H, W]` |
| `k_solver` | K |

`rng` is an optional `torch.Generator` for the probes; `None` uses the torch global RNG.
The function runs without gradients.

## Data

### `fetch_data(source="medmnist", ...)`

```python
from gdfnet_tvdenoise.data import fetch_data

fetch_data("medmnist")
fetch_data("lits")
fetch_data("all")
```

- `"medmnist"` downloads PathMNIST, ChestMNIST, and BloodMNIST at 64, 128, and
  224 px into `data/archives/{name}_{size}.npz`.
- `"lits"` downloads the pinned LiTS snapshot into `data/lits_raw/`.
- `"all"` runs both.
- `sizes`, `datasets`, `archive_dir`, `lits_raw_dir`, and `revision` are optional.
- The LiTS revision is pinned in `config.LITS_HF_REVISION`.

### `load_medmnist_images(dataset_name, ...)`

```python
from gdfnet_tvdenoise.data import load_medmnist_images

images = load_medmnist_images("pathmnist", split="test", indices=[0, 1], size=64)
```

Returns a float32 tensor normalized to [0, 1]. `indices` accepts an integer, a
sequence of integers, or a `torch.Tensor`. `count` samples without replacement
with `torch.randperm`, using the optional `rng: torch.Generator`. `rng=None`
uses the torch global RNG. `archive_dir` selects an explicit archive directory.

### `prepare_archives("lits", ...)`

Builds patient-disjoint, nested LiTS archives from the raw volumes in `data/lits_raw/`.
The build uses the authoritative split, so it uses no randomness.

```python
from gdfnet_tvdenoise.data import prepare_archives

prepare_archives("lits")
```

Writes `data/archives/lits_{size}.npz`. Each archive stores `{split}_images`,
`{split}_group_ids` (patient), and `{split}_slice_ids` (axial slice z).

The split manifest `data/lits_split.pt` (`config.LITS_SPLIT_PATH`) fixes the patient split
and the slice order. `slices[split]` is an int32 `[N, 2]` tensor of `(patient, z)` in archive
order. `counts[size][split]` is the number of slices in that size's archive. The `size`
archive holds the first `counts[size][split]` slices, so the larger archives are prefixes of
the 64 px archive. The manifest reproduces the archives that the committed datasets,
`CALIBRATION_IMAGES`, and `CANONICAL_SWEEP_IMAGES` index into.

`load_lits_split()` validates the manifest. It rejects repeated slices, patients in more than
one split, and out-of-range counts. Only the volumes of manifest patients are scanned. The
build fails fast if a manifest volume is missing (`FileNotFoundError`). It also fails if the
scan disagrees with the manifest at any size (`RuntimeError`): a listed slice is missing or
not eligible, or a split patient has an eligible slice that the manifest omits.
`dry_run=True` scans and checks without writing.

### Datasets and generation

The workloads in `config.py` define the generated datasets. Their keys are in
[Workload keys](#workload-keys).

`config.DEFAULT_DATASETS` maps `"training"` and each benchmark suite to a tuple of
bundled dataset files read by default. A suite may list several files, as
`scale_parity` does. These are the tracked `{name}_001.pt.xz` files. To use a newly
generated dataset, change the tuple there, or pass `data_path` (CLI `--data`) to
`train`. Loaders accept both `.pt.xz` and plain `.pt` files.

#### `generate_stochastic(workload, ...)`

```python
from gdfnet_tvdenoise.config import DEFAULT_PARITY_WORKLOAD, DEFAULT_TRAINING_WORKLOAD
from gdfnet_tvdenoise.data import generate_stochastic

path = generate_stochastic(DEFAULT_TRAINING_WORKLOAD)  # data/generated/training_002.pt
generate_stochastic(DEFAULT_PARITY_WORKLOAD, samples=10, dry_run=True)
```

- Writes a new `{name}_00N.pt` (`name` from the workload) with the next free `N`, in
  the workload's `output_dir` or in the explicit `output_dir`. A workload without
  `output_dir` fails unless `output_dir` is passed. Existing datasets are never
  overwritten.
- `samples` rescales the per-resolution sample count and keeps the split proportions.
- A workload with several splits (`training`) writes one sub-dict per split. A
  single-split workload writes one flat dict.
- An interrupted run leaves `{name}_00N.pt.partial` next to the target. Rerunning the
  same command resumes it automatically, and the partial is deleted after the final
  `.pt` is written. A different sample count, seed, sampling configuration, or archive
  directory fails fast. `resume=False` (CLI `--fresh`) discards the partial.

`seed` defaults to `None`, which draws fresh randomness on every run. Sampling
(modality, image, NSR, λ) uses a `torch.Generator` seeded with `seed` when one is
given, and the torch global RNG otherwise. Pass an integer `seed` to fix the
sampling. The sampled NSR, σ, and λ are float32 values. The stored `sigma` and
`lambda_tv` are exactly the values the solver received. Stored `gdf_n` and `mc_gdf`
are the unclamped Monte Carlo estimate. Training clamps the targets to `[0, 1]`. A
seeded run is not bit-exact: observation noise and the Monte Carlo probes in `gdf_n`
always use the torch global RNG, so labels differ by about their `gdf_n_se`. The
bundled datasets were generated unseeded.

#### `generate_sweeps(workload, ...)`

```python
from gdfnet_tvdenoise.config import DEFAULT_SCALE_SWEEPS_WORKLOAD, DEFAULT_SWEEPS_WORKLOAD
from gdfnet_tvdenoise.data import generate_sweeps

path = generate_sweeps(DEFAULT_SCALE_SWEEPS_WORKLOAD)
generate_sweeps(DEFAULT_SWEEPS_WORKLOAD, sweep_points=5, dry_run=True)
```

- Writes a new `{name}_00N.pt` with the next free `N`, using the same
  `output_dir` rules as `generate_stochastic`.
- `sweep_points` overrides the number of λ values per regime.
- An interrupted run resumes from the next regime boundary on the same command. The
  checkpoint rules match `generate_stochastic`.

Observation noise always uses the torch global RNG. `generate_sweeps` takes no seed
and the provenance records none. The η tiers still set σ = η · std(x) for the noise draw.
After the draw, the λ grid is a float32 `torch.logspace` over log10(λ/σ_MAD), scaled by
the float32 `σ_MAD = estimate_noise_wavelet_mad(y).sigma`. The dataset stores both
`sigmas` (true σ) and `sigma_mads`; `lambdas[r]` equals the grid scaled by
`sigma_mads[r]` exactly. The stored `lambdas` are exactly the float32 values passed to
the solver.

## Calibration

The calibration has three steps. Run them in this order:

1. `generate_flat_error_table` writes `e_k_table.pt`.
2. `calibrate_k` reads the archives and `e_k_table.pt` and writes `k_calibration.pt`.
   It prints the `SOLVER_K_TIERS` block.
3. Manual: copy the printed block into `config.py`. `calibrate_k` says if the printed
   values differ from the current ones.

### `generate_flat_error_table(...)`

Precomputes the flat-region solver error e(K, L) for K = 1..`k_max` at each
canonical size and writes `data/calibration/e_k_table.pt`. `k_max` defaults to
`E_K_TABLE_K_MAX`.

```python
from gdfnet_tvdenoise.calibration import generate_flat_error_table

generate_flat_error_table()
```

Use the CLI with `python -m gdfnet_tvdenoise calibrate --which table`. Regenerate
the artifact after changing `E_K_TABLE_K_MAX`, `DEFAULT_CP_SIGMA`, `DEFAULT_CP_TAU`,
`DEFAULT_CP_THETA`, or the tabulated sizes. The table stores its extent in
`metadata["k_max"]` and is loaded self-describing. `calibrate_k` raises `ValueError`
if the table is shorter than `K_CAL_MAX_K`. `helpers.flat_solver_error_table(size)`
returns the loaded table; K indexes entry K - 1.

### K calibration

`calibrate_k(...)` measures `K*` on the calibration images and writes
`data/calibration/k_calibration.pt`.

```python
from gdfnet_tvdenoise.calibration import calibrate_k

payload = calibrate_k()
```

Calibration settings live in the Calibration Process section of
`gdfnet_tvdenoise/config.py`: `CALIBRATION_IMAGES`, `DEFAULT_METROLOGY_NSR_TIERS`,
`DEFAULT_METROLOGY_DELTA_K`, `K_CAL_PROBES`, `K_CAL_TAU_REF`, and `K_CAL_MAX_K`. The
stop test also reads `DEFAULT_TAU`, `DEFAULT_M_BY_SIZE`, and
`METROLOGY_DOMINANCE_RATIO`.

`CALIBRATION_IMAGES` is a hand-picked set of archive images, each identified by
`(split, index)` and keyed by `(size, modality)`. There are 6 images per modality and
size, 72 in total. The function loads each image from its own split. It has no
automatic image selection. Pass `images=` to override the image dict, for example to
run a subset; the CLI has no flag for `images`.

For each image and NSR tier, the code sets `σ = η · std(x)`, draws
`y = create_noisy_observation(x, σ)`, measures `σ_MAD = estimate_noise_wavelet_mad(y)`,
and uses `λ = σ_MAD` with `K_CAL_PROBES` fresh Gaussian probes. Two solves share the
probes and step in lockstep:

- float32 on `device` (default `DEFAULT_DEVICE`) at step `DEFAULT_TAU`, as the labels;
- float64 on the CPU at step `K_CAL_TAU_REF`, the reference.

At every checkpoint, `E_τ` is the probe mean of the float32 minus the float64
divergences, and `SE_prod = ŝ / sqrt(DEFAULT_M_BY_SIZE[L])` from the float32
divergences. The run stops at the first checkpoint with
`e(K, L) + |E_τ| <= METROLOGY_DOMINANCE_RATIO · SE_prod`; that K is `K*`. A
configuration that reaches `K_CAL_MAX_K` raises `RuntimeError`. The tier values are the
largest `K*` over the images at each size and tier.

Observation noise and probe draws are unseeded, so every run draws them anew. A
resumable checkpoint is written next to the output after every configuration.
`fresh=True` discards it. The two solves do not overlap in time, so a step costs the
float32 step plus the float64 step. The full run takes about 6.5 h with MPS.

## Training

### `train(seed=11, ...)`

```python
from gdfnet_tvdenoise.train import train

path, metrics = train(seed=11)
```

`seed` defaults to 11 (CLI `--seed 11`). It seeds the torch global RNG and the
shuffle order of the training loader. `seed=None` leaves both unseeded.

The returned metrics are also stored under the `training` key of `weights.pt`: the
seed, the dataset path and SHA-256, the hyperparameters, the best epoch and
validation loss, the device type, the wall time, and the UTC creation time. See
[Model checkpoint](REPORT_SCHEMA.md#model-checkpoint).

Training uses AdamW with a cosine schedule, gradient clipping, a soft binary
cross-entropy loss on `gdf_n`, `SpatialGDFNet`, and the two features of
`extract_features`. The loss clamps the targets to `[0, 1]` for both the train and
the validation split. Numeric defaults are in `gdfnet_tvdenoise/config.py`. Training
on an accelerator is not bit-reproducible, also with a fixed `seed`.

Output:

- Default: the next free `models/model_00N/weights.pt`.
- Explicit `output`: the requested file.
- If the requested file exists, `_001`, `_002`, and so on are appended to the
  file stem.
- Writes are atomic.
- Returns `(absolute_path, metrics)`.

Use `max_samples` for smoke runs. Full training uses all training and validation
samples.

## Benchmarks

### `run_benchmark(model_path, suite_name="all", device=DEFAULT_DEVICE, plot=True)`

Runs one suite or all five suites and writes reports next to the model.
`DEFAULT_DEVICE` is the current PyTorch accelerator (for example CUDA or MPS), else CPU.

```python
from gdfnet_tvdenoise.benchmarks import run_benchmark

paths = run_benchmark("models/model_003", "parity")
paths = run_benchmark("models/model_003", "all")
```

Suite names:

- `parity`
- `val_parity`
- `sweeps`
- `scale_parity`
- `scale_sweeps`

Returns `dict[str, Path]`. A report is written to
`<model_path>/<suite_name>.pt`. Rerunning a suite overwrites the same file.
With `plot=True` (the default), each suite except `val_parity` also writes
`<model_path>/<suite_name>.png`. `val_parity` has no plot.

### `plot_benchmark(model_path, suite_name="all", output_path=None)`

Renders saved reports without re-evaluation.

```python
from gdfnet_tvdenoise.benchmarks import plot_benchmark

plot_benchmark("models/model_003", "parity")  # Path
plot_benchmark("models/model_003")  # list[Path], one per plottable suite
```

The plottable suites are `parity`, `sweeps`, `scale_parity`, and `scale_sweeps`.

Plotting reads the stored report arrays as CPU `torch.Tensor` values. A report that
stores NumPy arrays fails with `TypeError`.

### `plot_overview(model_path=DEFAULT_MODEL_PATH, output_path=OVERVIEW_FIGURE_PATH, ...)`

Renders the README figure `docs/overview.png` from a saved sweeps report and its dataset.

```python
from gdfnet_tvdenoise.benchmarks.plots import plot_overview

plot_overview()  # Path to docs/overview.png
```

The left part shows the clean image, the observation, and TV reconstructions at the
Monte Carlo SURE selection λ* and `OVERVIEW_PATH_NODE_OFFSET` grid nodes below and above
it, each labeled with its Monte Carlo GDF/N. The right part shows the Monte Carlo and
surrogate GDF and SURE curves of the same regime with both selections. The defaults show
regime `OVERVIEW_REGIME_INDEX` of `scale_sweeps` (Abdominal CT, 224 px, σ/std(x) = 1.00)
for the default model. The three TV solves run on `device` with the regime's stored K.
It raises `ValueError` if the report λ grid does not match its dataset, or if an offset
node falls outside the grid.

## Serialization

```python
from gdfnet_tvdenoise.helpers import load_data_file, load_pt, load_pt_xz, save_pt

payload = load_pt("models/model_003/parity.pt")
payload = load_pt_xz("data/generated/test/parity_001.pt.xz")
payload = load_data_file("data/generated/test/parity_001.pt")
save_pt(payload, "output.pt")
```

`save_pt` and `save_pt_xz` write atomically. `load_pt_xz` and `save_pt_xz` read and
write the compressed datasets; `load_data_file` chooses by suffix.
`next_free_path(directory, stem, suffix)` returns the first unused
`{stem}_00N{suffix}`; `train` uses it. The generators reuse the next free dataset
index across both `.pt` and `.pt.xz` files. The module also provides `atomic_write`,
`display_path`, and `resolve_model_paths`.

## Command line

Run `python -m gdfnet_tvdenoise --help` for the full parser.

| Subcommand | Purpose | Key arguments |
|---|---|---|
| `infer` | Predict GDF for one image and lambda | `--lambda-tv`, `--image`, `--model`, `--sigma`, `--mse` |
| `generate` | Fetch sources, prepare archives, generate data | `--source`, `--prepare-lits`, `--stochastic NAME\|all`, `--sweeps NAME\|all`, `--samples`, `--sweep-points`, `--seed` (stochastic only; default unseeded), `--archive-dir`, `--output-dir`, `--dry-run`, `--fresh` |
| `calibrate` | Tabulate the solver error or calibrate the solver budget K | `--which` (`table`, `k`; default `k`), `--nsr-tiers`, `--archive-dir`, `--output`, `--fresh` |
| `train` | Train a surrogate | `--seed` (default 11), `--epochs`, `--batch-size`, `--lr`, `--data`, `--output`, `--max-samples` |
| `benchmark` | Run benchmark suites and plot them | `--model-path`, `--suite` (default `all`), `--no-plot` |
| `plot` | Render saved reports | `--model-path`, `--suite` (default `all`), `--output` |

`generate`, `calibrate`, `train`, and `benchmark` also take `--device`. The default is
`DEFAULT_DEVICE`.

`--nsr-tiers`, `--archive-dir`, and `--fresh` apply to `k`. `--output` applies to
both modes.

The `calibrate` help text says "Calibrate the solver budget K."

`--archive-dir` overrides `config.ARCHIVE_DIR` for every `generate` mode and for
`calibrate --which k`.

```bash
python -m gdfnet_tvdenoise infer --lambda-tv 0.05
python -m gdfnet_tvdenoise benchmark --model-path models/model_003 --suite parity
python -m gdfnet_tvdenoise plot --model-path models/model_003 --suite parity
```

## Bundled artifact provenance

The current code generated the bundled datasets and calibration, and trained the
bundled models. A rerun reproduces the conclusions, not the files bit for bit: the
sample draws, observation noise, Monte Carlo probes, and calibration noise are
unseeded, and training on an accelerator is not bit-reproducible.

- **K calibration.** `k_calibration.pt` was assembled from a two-pass run of the same
  measurement: the float32 solves ran first, with stored probe divergences at every
  checkpoint, and the float64 reference ran afterwards on the same y and probes. The
  images, tiers, probes, steps, and stop test equal those of `calibrate_k`, so K* is the
  K the lockstep loop finds for the same draws. Its `seconds` and `elapsed_seconds` are
  NaN, and `metadata["note"]` records the origin.
- **Model metadata.** `training.data_file` names `training_001.pt.xz`, and
  `training.data_sha256` is its hash. Training read an uncompressed copy with the same
  contents.

## Exceptions

- `ModelNotFoundError`: checkpoint file does not exist.
- `FileNotFoundError`: data or report file does not exist.
- `ValueError`: invalid shape, value, or selection index.
