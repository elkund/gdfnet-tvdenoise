"""Single source of truth for shared constants, paths, and workload defaults.

See the Configuration section of docs/REFERENCE.md for which constants are safe to change.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

# ===========================================================================
# 0. Device
# ===========================================================================

#: Device used by every function that takes a `device` argument. Replace it with
#: `torch.device("cpu")`, `torch.device("cuda")`, or similar to force a device.
DEFAULT_DEVICE: torch.device = torch.accelerator.current_accelerator() or torch.device("cpu")

# ===========================================================================
# 1. Problem Domain
# ===========================================================================

#: Default spatial image dimension in pixels (square H = W = 64).
DEFAULT_IMAGE_SIZE: int = 64

#: Canonical biomedical modalities.
CANONICAL_MEDMNIST_DATASETS: tuple[str, ...] = (
	"lits",
	"pathmnist",
	"chestmnist",
	"bloodmnist",
)

#: Supported 2D image resolutions.
CANONICAL_MEDMNIST_SIZES: tuple[int, ...] = (64, 128, 224)

#: MedMNIST+ modalities that are downloaded from Zenodo. LiTS is fetched separately.
MEDMNIST_PLUS_DATASETS: tuple[str, ...] = ("pathmnist", "chestmnist", "bloodmnist")

# ===========================================================================
# 2. Noise and Regularization Sampling
# ===========================================================================

#: Noise domain: dimensionless noise-to-contrast ratio eta = sigma / std(x_clean).
DEFAULT_NSR_MIN: float = 0.05
DEFAULT_NSR_MAX: float = 1.00

#: Regularization parameter sampling bounds.
DEFAULT_LAMBDA_MIN: float = 10**-4.0
DEFAULT_LAMBDA_MAX: float = 10**1.0

#: Canonical dimensionless sweep range in log10(lambda / sigma_MAD).
DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MIN: float = -2.0
DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MAX: float = 1.0

#: Calibrated empirical capacity sigmoid fit on 64 px denoising sweeps.
CALIBRATED_DENOISE_LOGIT_X0: float = -1.5206
CALIBRATED_DENOISE_LOGIT_W: float = 0.8435

#: Training-dataset sampling: log-uniform lambda.
LOG_UNIFORM_SAMPLING: dict[str, float | str] = {
	"name": "log_uniform",
	"nsr_min": DEFAULT_NSR_MIN,
	"nsr_max": DEFAULT_NSR_MAX,
	"lambda_dist": "uniform",
	"lambda_min": DEFAULT_LAMBDA_MIN,
	"lambda_max": DEFAULT_LAMBDA_MAX,
}

#: Parity-dataset sampling: lambda from the logistic capacity quantile u.
BALANCED_SCATTER_SAMPLING: dict[str, float | str] = {
	"name": "balanced_scatter",
	"nsr_min": DEFAULT_NSR_MIN,
	"nsr_max": DEFAULT_NSR_MAX,
	"lambda_dist": "regime_logit",
	"u_min": 0.01,
	"u_max": 0.99,
	"denoise_x0": CALIBRATED_DENOISE_LOGIT_X0,
	"denoise_w": CALIBRATED_DENOISE_LOGIT_W,
}

# ===========================================================================
# 3. Solver Engine and Convergence Criteria
# ===========================================================================

#: Canonical single-loop Chambolle-Pock hyperparameters.
DEFAULT_CP_TAU: float = 0.31
DEFAULT_CP_SIGMA: float = 0.31
DEFAULT_CP_THETA: float = 1.0

#: Solver budget K per size at the noise tiers `DEFAULT_METROLOGY_NSR_TIERS`, in tier order.
#: Each entry is the largest calibrated K* over the calibration images at that size and tier
#: (`calibration.calibrate_k`). `solver.solver_k` interpolates linearly in log eta.
SOLVER_K_TIERS: dict[int, tuple[int, ...]] = {
	64: (1150, 1550, 2100),
	128: (1200, 2100, 3000),
	224: (1300, 2550, 3200),
}

# ===========================================================================
# 4. Monte Carlo GDF and Metrology
# ===========================================================================

#: Production Monte Carlo probe counts M_prod per size.
DEFAULT_M_BY_SIZE: dict[int, int] = {
	64: 60,
	128: 15,
	224: 5,
}

#: Finite-difference step of the Monte Carlo divergence for normalized 2D images.
DEFAULT_TAU: float = 1e-5

#: Metrological dominance ratio r of the K calibration: e(K) + |E_tau| <= r * SE_prod.
METROLOGY_DOMINANCE_RATIO: float = 0.30

#: Selection-excess model constants.
SELECTION_MDP_TARGET: float = 1.0
SELECTION_QUADRATURE_ABS_TOL: float = 1e-11
SELECTION_QUADRATURE_REL_TOL: float = 1e-10
SELECTION_QUADRATURE_LIMIT: int = 250
SELECTION_QUADRATURE_REFINEMENT_FACTOR: float = 10.0
SELECTION_VERIFICATION_ABS_TOL: float = 1e-9
SELECTION_VERIFICATION_CORRELATION: float = 0.6
SELECTION_VERIFICATION_FAR_MEAN: float = 8.0
SELECTION_VERIFICATION_REL_TOL: float = 1e-9

#: Standard normal density at zero, 1 / sqrt(2 pi).
STANDARD_NORMAL_DENSITY_AT_ZERO: float = 1.0 / math.sqrt(2.0 * math.pi)

#: Standard abdominal CT soft-tissue Hounsfield Unit window for LiTS archives.
LITS_HU_MIN: float = -160.0
LITS_HU_MAX: float = 240.0

#: LiTS source repository (Medical Segmentation Decathlon Task03 Liver mirror).
LITS_HF_REPO: str = "MedOtter/msd-liver"
#: Immutable Hugging Face revision used to build the distributed LiTS archives.
LITS_HF_REVISION: str = "6a5aad7ca262f2420d1bd4c2517b37b265db53eb"

#: LiTS archive build constants.
LITS_CROP_MARGIN: float = 1.05
LITS_MAX_AIR_FRACTION: float = 0.05
LITS_AIR_HU_MAX: float = -200.0
LITS_AIR_SHIFT_PX: int = 30
LITS_AIR_SHIFT_STEP_PX: int = 10

#: LiTS archive partitions, in archive order.
LITS_SPLITS: tuple[str, ...] = ("train", "val", "test")

#: Upper quartile z = Phi^-1(3/4) of the standard normal.
WAVELET_MAD_NORMAL_QUARTILE: float = 0.6744897501960817
#: Asymptotic Gaussian variance factor of the Haar-HH Wavelet MAD noise estimate.
WAVELET_MAD_VARIANCE_FACTOR: float = (
	4.0
	* math.pi
	* math.exp(WAVELET_MAD_NORMAL_QUARTILE**2)
	/ (8.0 * WAVELET_MAD_NORMAL_QUARTILE**2)
)
#: Fixed Haar-HH sample budget and independent sampling seed for noise estimation.
DEFAULT_WAVELET_MAD_N_COEFFICIENTS: int = 1024
DEFAULT_WAVELET_MAD_SAMPLING_SEED: int = 0

# ===========================================================================
# 5. Calibration Process
# ===========================================================================

#: Calibration images: hand-picked archive images, keyed by (size, modality).
#: Each image is a (split, index) pair.
CALIBRATION_IMAGES: dict[tuple[int, str], list[tuple[str, int]]] = {
	(64, "lits"): [
		("val", 529),
		("val", 749),
		("val", 347),
		("val", 311),
		("val", 374),
		("val", 45),
	],
	(64, "pathmnist"): [
		("val", 4639),
		("val", 8474),
		("val", 2393),
		("val", 6667),
		("val", 3789),
		("val", 15),
	],
	(64, "chestmnist"): [
		("val", 7854),
		("val", 4192),
		("val", 4643),
		("val", 9510),
		("val", 2764),
		("val", 8402),
	],
	(64, "bloodmnist"): [
		("val", 1131),
		("val", 56),
		("val", 714),
		("val", 564),
		("val", 1685),
		("val", 1302),
	],
	(128, "lits"): [
		("val", 219),
		("val", 327),
		("val", 394),
		("val", 519),
		("val", 349),
		("val", 234),
	],
	(128, "pathmnist"): [
		("val", 8865),
		("val", 26),
		("val", 4959),
		("val", 7102),
		("val", 9462),
		("val", 4316),
	],
	(128, "chestmnist"): [
		("val", 1928),
		("val", 11162),
		("val", 7243),
		("val", 9902),
		("val", 4889),
		("val", 9041),
	],
	(128, "bloodmnist"): [
		("val", 302),
		("val", 585),
		("val", 1230),
		("val", 544),
		("val", 524),
		("val", 196),
	],
	(224, "lits"): [
		("val", 216),
		("val", 228),
		("val", 79),
		("val", 146),
		("val", 203),
		("val", 181),
	],
	(224, "pathmnist"): [
		("val", 4604),
		("val", 3181),
		("val", 7254),
		("val", 6207),
		("val", 8171),
		("val", 5),
	],
	(224, "chestmnist"): [
		("val", 9867),
		("val", 728),
		("val", 320),
		("val", 9230),
		("val", 2376),
		("val", 8436),
	],
	(224, "bloodmnist"): [
		("val", 996),
		("val", 1200),
		("val", 622),
		("val", 1554),
		("val", 346),
		("val", 11),
	],
}

#: Metrology calibration noise tiers in eta = sigma / std(x_clean).
DEFAULT_METROLOGY_NSR_TIERS: tuple[float, ...] = (0.05, 0.50, 1.00)

#: Solver checkpoint interval of the K calibration and the Monte Carlo divergence.
DEFAULT_METROLOGY_DELTA_K: int = 50

#: Monte Carlo probes per K calibration configuration, at every size.
K_CAL_PROBES: int = 60

#: Finite-difference step of the float64 CPU reference divergence in the K calibration.
K_CAL_TAU_REF: float = 1e-8

#: Safety limit of the K calibration solve. A configuration that reaches it raises.
K_CAL_MAX_K: int = 20000

#: Extent of the flat solver-error table.
E_K_TABLE_K_MAX: int = 40000

# ===========================================================================
# 6. Feature Representation and Architecture
# ===========================================================================

#: Denominator floor of the Tikhonov trace in the second feature.
FEATURE_TIKHONOV_DENOM_FLOOR: float = 1e-12

#: Maximum batch size per surrogate network call.
PREDICT_CHUNK_SIZE: int = 16

#: Denominator floor of 1 - r in the ratio-based Morozov criterion and its uncertainty.
MDP_DENOMINATOR_FLOOR: float = 1e-6

#: Model architecture: convolution channels, residual blocks, and dilation per block.
MODEL_CHANNELS: int = 32
MODEL_NUM_BLOCKS: int = 4
MODEL_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)

# ===========================================================================
# 7. Training Defaults
# ===========================================================================

DEFAULT_TRAIN_EPOCHS: int = 250
DEFAULT_TRAIN_LR: float = 1e-3
DEFAULT_TRAIN_BATCH_SIZE: int = 16
DEFAULT_TRAIN_WEIGHT_DECAY: float = 1e-4
DEFAULT_TRAIN_ETA_MIN: float = 1e-6
DEFAULT_TRAIN_GRAD_CLIP: float = 1.0
DEFAULT_TRAIN_SEED: int | None = 11

#: Probability clamp margin of the soft BCE loss: predictions are clipped to [eps, 1 - eps].
TRAIN_BCE_EPS: float = 1e-7

#: Target clamp of the soft BCE loss. Labels are stored unclamped; training clips them.
TRAIN_TARGET_MIN: float = 0.0
TRAIN_TARGET_MAX: float = 1.0

# ===========================================================================
# 8. Paths
# ===========================================================================

#: Repository root.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = REPO_ROOT / "data"
ARCHIVE_DIR: Path = DATA_DIR / "archives"
LITS_RAW_DIR: Path = DATA_DIR / "lits_raw"
#: Authoritative LiTS split manifest. For each split, it holds the (patient, z) slices in archive
#: order and the slice count per archive size.
LITS_SPLIT_PATH: Path = DATA_DIR / "lits_split.pt"
GENERATED_DIR: Path = DATA_DIR / "generated"
GENERATED_TEST_DIR: Path = GENERATED_DIR / "test"
CALIBRATION_DIR: Path = DATA_DIR / "calibration"
MODELS_DIR: Path = REPO_ROOT / "models"

#: Data-fetch transport: read/hash buffer size and maximum download attempts.
CHUNK_SIZE_BYTES: int = 1024 * 1024
DEFAULT_MAX_RETRIES: int = 5

#: Data-generation checkpoint state: file suffix and periodic save interval.
GENERATION_CHECKPOINT_SUFFIX: str = ".partial"
GENERATION_CHECKPOINT_INTERVAL_SECONDS: float = 300.0

#: Default dataset files read by `train` ("training") and by the benchmarks (the others).
#: A suite may aggregate several source datasets. These are the bundled files. Generated
#: datasets are not selected here; each workload declares its own `output_dir`.
DEFAULT_DATASETS: dict[str, tuple[Path, ...]] = {
	"training": (GENERATED_DIR / "training_001.pt.xz",),
	"parity": (GENERATED_TEST_DIR / "parity_001.pt.xz",),
	"sweeps": (GENERATED_TEST_DIR / "sweeps_001.pt.xz",),
	"scale_parity": (
		GENERATED_TEST_DIR / "parity_001.pt.xz",
		GENERATED_TEST_DIR / "scale_parity_001.pt.xz",
	),
	"scale_sweeps": (GENERATED_TEST_DIR / "scale_sweeps_001.pt.xz",),
	"val_parity": (GENERATED_TEST_DIR / "val_parity_001.pt.xz",),
}

#: Calibration artifacts.
K_CALIBRATION_PATH: Path = CALIBRATION_DIR / "k_calibration.pt"
E_K_TABLE_PATH: Path = CALIBRATION_DIR / "e_k_table.pt"

#: Published models: three training seeds of one configuration, as `models/` directory names.
#: `DEFAULT_MODEL` is the declared median of the three by validation MAE.
PUBLISHED_MODELS: tuple[str, ...] = ("model_001", "model_002", "model_003")
DEFAULT_MODEL: str = "model_003"
if DEFAULT_MODEL not in PUBLISHED_MODELS:
	raise ValueError(f"DEFAULT_MODEL {DEFAULT_MODEL!r} is not in PUBLISHED_MODELS")
DEFAULT_MODEL_PATH: Path = MODELS_DIR / DEFAULT_MODEL / "weights.pt"

#: README overview figure: the sweeps report and regime shown, the offset in grid nodes of
#: the under- and over-regularized examples from the SURE selection, and the output file.
OVERVIEW_SUITE: str = "scale_sweeps"
OVERVIEW_REGIME_INDEX: int = 20
OVERVIEW_PATH_NODE_OFFSET: int = 4
OVERVIEW_FIGURE_PATH: Path = REPO_ROOT / "docs" / "overview.png"

# ===========================================================================
# 9. Benchmark Regimes and Canonical Workloads
# ===========================================================================

#: Canonical noise tiers for benchmark sweeps in eta = sigma / std(x_clean).
CANONICAL_SWEEP_NSR_TIERS: dict[str, float] = {
	"Mild": 0.25,
	"Moderate": 0.50,
	"Severe": 1.00,
}

#: Canonical benchmark images: (img_tag, img_idx, description, short label).
CANONICAL_SWEEP_IMAGES: tuple[tuple[str, int, str, str], ...] = (
	("lits", 546, "Abdominal CT", "LiTS"),
	("pathmnist", 0, "Colon Pathology", "PathMNIST"),
	("chestmnist", 0, "Chest X-Ray", "ChestMNIST"),
	("bloodmnist", 24, "Blood Cell Microscopy", "BloodMNIST"),
)

#: Canonical 12-panel pure-denoising benchmark regimes.
CANONICAL_12PANEL_REGIMES: list[dict[str, object]] = [
	{
		"title": (
			f"Panel {t_idx * len(CANONICAL_SWEEP_IMAGES) + s_idx + 1}: Denoise — {desc} "
			f"({short}, {tier} σ/std(x) = {nsr:.2f})"
		),
		"img_tag": tag,
		"img_idx": idx,
		"nsr": nsr,
	}
	for t_idx, (tier, nsr) in enumerate(CANONICAL_SWEEP_NSR_TIERS.items())
	for s_idx, (tag, idx, desc, short) in enumerate(CANONICAL_SWEEP_IMAGES)
]

# Workloads for `data.generate_stochastic` and `data.generate_sweeps`. `name` sets the
# output file `{name}_00N.pt` and `output_dir` sets where it is written. A workload
# without `output_dir` is not directly generatable and fails unless a directory is
# passed explicitly. Stochastic workloads draw images at random from the archive splits
# in `splits`, with the given number of samples per split, in order. Sweeps workloads
# run a fixed λ grid on fixed regimes.

#: Stochastic training dataset workload.
DEFAULT_TRAINING_WORKLOAD: dict[str, object] = {
	"name": "training",
	"output_dir": GENERATED_DIR,
	"sizes": [DEFAULT_IMAGE_SIZE],
	"sampling_config": LOG_UNIFORM_SAMPLING,
	"splits": {"train": 4000, "val": 700},
}

#: Stochastic 64 px parity workload.
DEFAULT_PARITY_WORKLOAD: dict[str, object] = {
	"name": "parity",
	"output_dir": GENERATED_TEST_DIR,
	"sizes": [DEFAULT_IMAGE_SIZE],
	"sampling_config": BALANCED_SCATTER_SAMPLING,
	"splits": {"test": 100},
}

#: Stochastic multi-scale parity workload. Each split count is per resolution: 100
#: gives 100 samples at each of the two sizes (200 total).
DEFAULT_SCALE_PARITY_WORKLOAD: dict[str, object] = {
	"name": "scale_parity",
	"output_dir": GENERATED_TEST_DIR,
	"sizes": [128, 224],
	"sampling_config": BALANCED_SCATTER_SAMPLING,
	"splits": {"test": 100},
}

#: Stochastic validation parity workload for model selection.
DEFAULT_VAL_PARITY_WORKLOAD: dict[str, object] = {
	"name": "val_parity",
	"output_dir": GENERATED_TEST_DIR,
	"sizes": [DEFAULT_IMAGE_SIZE],
	"sampling_config": BALANCED_SCATTER_SAMPLING,
	"splits": {"val": 200},
}

#: Sweeps workload.
DEFAULT_SWEEPS_WORKLOAD: dict[str, object] = {
	"name": "sweeps",
	"output_dir": GENERATED_TEST_DIR,
	"scales": [DEFAULT_IMAGE_SIZE],
	"sweep_points": 25,
	"regimes": CANONICAL_12PANEL_REGIMES,
}

#: Multi-scale sweeps workload.
DEFAULT_SCALE_SWEEPS_WORKLOAD: dict[str, object] = {
	"name": "scale_sweeps",
	"output_dir": GENERATED_TEST_DIR,
	"scales": [128, 224],
	"sweep_points": 25,
	"regimes": CANONICAL_12PANEL_REGIMES,
}

#: Benchmark suites in deterministic order.
BENCHMARK_SUITES: tuple[str, ...] = (
	"parity",
	"val_parity",
	"sweeps",
	"scale_parity",
	"scale_sweeps",
)
