"""Shared analytical, device, and Monte Carlo GDF/SURE utilities."""

from __future__ import annotations

import functools
import lzma
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

from gdfnet_tvdenoise.config import (
	DEFAULT_CP_SIGMA,
	DEFAULT_CP_TAU,
	DEFAULT_CP_THETA,
	DEFAULT_TAU,
	DEFAULT_WAVELET_MAD_N_COEFFICIENTS,
	DEFAULT_WAVELET_MAD_SAMPLING_SEED,
	E_K_TABLE_PATH,
	MDP_DENOMINATOR_FLOOR,
	REPO_ROOT,
	SELECTION_MDP_TARGET,
	SELECTION_QUADRATURE_ABS_TOL,
	SELECTION_QUADRATURE_LIMIT,
	SELECTION_QUADRATURE_REFINEMENT_FACTOR,
	SELECTION_QUADRATURE_REL_TOL,
	SELECTION_VERIFICATION_ABS_TOL,
	SELECTION_VERIFICATION_CORRELATION,
	SELECTION_VERIFICATION_FAR_MEAN,
	SELECTION_VERIFICATION_REL_TOL,
	STANDARD_NORMAL_DENSITY_AT_ZERO,
	WAVELET_MAD_NORMAL_QUARTILE,
	WAVELET_MAD_VARIANCE_FACTOR,
)
from gdfnet_tvdenoise.solver import tv_denoise_iterates

# ===========================================================================
# Device, IO, and path utilities
# ===========================================================================


def release_device_cache() -> None:
	"""Release cached MPS memory. Other devices need no action."""
	if torch.backends.mps.is_available():
		torch.mps.empty_cache()


@contextmanager
def atomic_write(path: str | Path) -> Iterator[Path]:
	"""Yield a temporary sibling path and move it onto `path` when the block succeeds."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name(path.name + ".tmp")
	yield tmp
	tmp.replace(path)


def save_pt(payload: Any, path: str | Path) -> Path:
	"""Atomically save a plain PyTorch `.pt` payload."""
	with atomic_write(path) as tmp:
		torch.save(payload, tmp)
	return Path(path)


def save_pt_xz(payload: Any, path: str | Path) -> Path:
	"""Atomically save an LZMA-compressed PyTorch `.pt.xz` payload."""
	with atomic_write(path) as tmp, lzma.open(tmp, "wb") as handle:
		torch.save(payload, handle)
	return Path(path)


def next_free_path(directory: Path, stem: str, suffix: str = "") -> Path:
	"""Return the first `directory / f"{stem}_{N:03d}{suffix}"` that does not exist, N >= 1."""
	index = 1
	while (directory / f"{stem}_{index:03d}{suffix}").exists():
		index += 1
	return directory / f"{stem}_{index:03d}{suffix}"


def load_pt(path: str | Path) -> Any:
	"""Load a plain PyTorch `.pt` payload."""
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"PyTorch file not found: {path}")
	return torch.load(path, map_location="cpu")


def load_pt_xz(path: str | Path) -> Any:
	"""Load a compressed PyTorch `.pt.xz` payload."""
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Compressed data file not found: {path}")
	with lzma.open(path, "rb") as handle:
		return torch.load(handle, map_location="cpu")


def load_data_file(path: str | Path) -> Any:
	"""Load a plain `.pt` or LZMA-compressed `.pt.xz` dataset payload."""
	path = Path(path)
	if path.suffix == ".xz":
		return load_pt_xz(path)
	return load_pt(path)


def display_path(path: Path) -> str:
	"""Return a repository-relative path when possible."""
	try:
		return str(path.resolve().relative_to(REPO_ROOT))
	except ValueError:
		return str(path)


def resolve_model_paths(model_path: str | Path) -> tuple[Path, Path]:
	"""Return (weights file, model directory) for a model directory or a weights file."""
	path = Path(model_path)
	if path.is_dir():
		return path / "weights.pt", path
	return path, path.parent


# ===========================================================================
# Metrics and selection criteria
# ===========================================================================


def compute_regression_metrics(
	y_true: torch.Tensor,
	y_pred: torch.Tensor,
) -> dict[str, float]:
	"""Compute R^2, MAE, and RMSE for regression evaluation.

	Args:
		y_true: Ground truth target values.
		y_pred: Predicted values.

	Returns:
		Dictionary containing 'r2', 'mae', and 'rmse' metrics as Python floats.
	"""
	true_t = torch.as_tensor(y_true, dtype=torch.float32).flatten()
	pred_t = torch.as_tensor(y_pred, dtype=torch.float32).flatten()
	residual = true_t - pred_t
	ss_res = torch.sum(residual**2).item()
	ss_tot = torch.sum((true_t - true_t.mean()) ** 2).item()
	r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
	return {
		"r2": r2,
		"mae": torch.mean(torch.abs(residual)).item(),
		"rmse": torch.sqrt(torch.mean(residual**2)).item(),
	}


def compute_sure(
	mse: torch.Tensor | float,
	gdf_n: torch.Tensor | float,
	sigma: float,
) -> torch.Tensor:
	"""Compute Stein's Unbiased Risk Estimate (per pixel).

	Formula: SURE = MSE - sigma^2 + 2 * sigma^2 * GDF_n.

	Args:
		mse: Mean squared residual ||y - A x_hat||^2 / N in measurement space.
		gdf_n: Normalized Generalized Degrees of Freedom in [0, 1].
		sigma: White Gaussian noise standard deviation.

	Returns:
		float32 SURE loss values of matching shape. Python float inputs give a 0-d tensor.
	"""
	mse_t = torch.as_tensor(mse, dtype=torch.float32)
	gdf_t = torch.as_tensor(gdf_n, dtype=torch.float32)
	return mse_t - (sigma**2) + 2.0 * (sigma**2) * gdf_t


def compute_mdp(
	mse: torch.Tensor | float,
	gdf_n: torch.Tensor | float,
	sigma: float,
) -> torch.Tensor:
	"""Compute the ratio-based Morozov discrepancy criterion.

	Formula: MDP = mse / ((1 - r) * sigma^2), with the denominator floor
	`1 - r >= config.MDP_DENOMINATOR_FLOOR`. The floor keeps the criterion finite at
	r = 1. A nonpositive sigma fails fast.

	Args:
		mse: Mean squared residual in measurement space.
		gdf_n: Normalized Generalized Degrees of Freedom in [0, 1].
		sigma: White Gaussian noise standard deviation.

	Returns:
		float32 ratio values of matching shape. Python float inputs give a 0-d tensor.
	"""
	if not math.isfinite(sigma) or sigma <= 0.0:
		raise ValueError(f"sigma must be finite and positive; got {sigma}.")
	mse_t = torch.as_tensor(mse, dtype=torch.float32)
	gdf_t = torch.as_tensor(gdf_n, dtype=torch.float32)
	return mse_t / torch.clamp(1.0 - gdf_t, min=MDP_DENOMINATOR_FLOOR) / sigma**2


# ===========================================================================
# Criterion uncertainty and Wavelet MAD noise estimation (internal benchmark support)
# ===========================================================================


def compute_sure_uncertainty(
	sigma: float,
	gdf_se: torch.Tensor,
) -> torch.Tensor:
	"""Propagate the 1-SE Monte Carlo capacity uncertainty to known-sigma SURE risk.

	Formula: Delta SURE = 2 * sigma^2 * SE(r).

	Args:
		sigma: White Gaussian noise standard deviation.
		gdf_se: 1-SE Monte Carlo uncertainty of the normalized capacity r.

	Returns:
		1-SE uncertainty in SURE risk of matching shape and dtype.
	"""
	return 2.0 * (sigma**2) * gdf_se


def compute_mdp_known_sigma_uncertainty(
	morozov_ratio: torch.Tensor,
	gdf_n: torch.Tensor,
	gdf_se: torch.Tensor,
) -> torch.Tensor:
	"""Propagate the 1-SE capacity uncertainty to the known-sigma Morozov ratio.

	Formula: Delta ratio = ratio * SE(r) / max(1 - r, MDP_DENOMINATOR_FLOOR).

	Args:
		morozov_ratio: Known-sigma Morozov ratio (residual variance over sigma^2).
		gdf_n: Normalized capacity r.
		gdf_se: 1-SE Monte Carlo uncertainty of the normalized capacity r.

	Returns:
		1-SE uncertainty in the Morozov ratio of matching shape and dtype.
	"""
	return morozov_ratio * gdf_se / torch.clamp(1.0 - gdf_n, min=MDP_DENOMINATOR_FLOOR)


@dataclass(frozen=True)
class WaveletMADNoiseEstimate:
	"""Noise estimate and the actual number of sampled Haar-HH coefficients."""

	sigma: float
	n_coefficients: int


def estimate_noise_wavelet_mad(y: torch.Tensor) -> WaveletMADNoiseEstimate:
	"""Estimate white Gaussian noise standard deviation via Haar wavelet diagonal MAD.

	Uses Donoho & Johnstone (1994) robust median absolute deviation on the finest diagonal
	(HH) wavelet subband.

	Args:
		y: One finite grayscale observation, shaped [H, W], [1, H, W], or [1, 1, H, W].

	Returns:
		Noise estimate from the configured fixed coefficient budget and its sample count.
	"""
	if not isinstance(y, torch.Tensor):
		raise TypeError("y must be a torch.Tensor.")
	y_t = y.detach()
	if y_t.is_complex():
		raise TypeError("y must contain real-valued pixels.")
	y_t = y_t.to(dtype=torch.float32)

	if y_t.ndim == 2:
		y_t = y_t.unsqueeze(0).unsqueeze(0)
	elif y_t.ndim == 3 and y_t.shape[0] == 1:
		y_t = y_t.unsqueeze(0)
	elif y_t.ndim != 4 or tuple(y_t.shape[:2]) != (1, 1):
		raise ValueError(f"Expected one grayscale image; got shape {tuple(y_t.shape)}.")
	if not torch.isfinite(y_t).all():
		raise ValueError("y must contain only finite pixels.")
	n_available = (y_t.shape[-2] // 2) * (y_t.shape[-1] // 2)
	if n_available < DEFAULT_WAVELET_MAD_N_COEFFICIENTS:
		raise ValueError(
			f"Need {DEFAULT_WAVELET_MAD_N_COEFFICIENTS} Haar-HH coefficients; "
			f"image provides {n_available}."
		)

	kernel = (
		torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=y_t.dtype, device=y_t.device).view(
			1, 1, 2, 2
		)
		* 0.5
	)
	hh = F.conv2d(y_t, kernel, stride=2).flatten()
	if hh.numel() > DEFAULT_WAVELET_MAD_N_COEFFICIENTS:
		generator = torch.Generator(device="cpu").manual_seed(DEFAULT_WAVELET_MAD_SAMPLING_SEED)
		indices = torch.randperm(hh.numel(), generator=generator, device="cpu")
		hh = hh[indices[:DEFAULT_WAVELET_MAD_N_COEFFICIENTS].to(hh.device)]
	med = torch.median(torch.abs(hh)).item()
	return WaveletMADNoiseEstimate(
		sigma=float(med / WAVELET_MAD_NORMAL_QUARTILE), n_coefficients=hh.numel()
	)


def compute_mad_sigma_sq_variance(sigma_mad: float, n_coefficients: int) -> float:
	"""Return the asymptotic variance of the Haar-HH Wavelet MAD noise variance estimate."""
	if type(n_coefficients) is not int or n_coefficients < 1:
		raise ValueError(f"n_coefficients must be a positive integer; got {n_coefficients!r}.")
	if not math.isfinite(sigma_mad) or sigma_mad < 0.0:
		raise ValueError(f"sigma_mad must be finite and nonnegative; got {sigma_mad}.")
	return WAVELET_MAD_VARIANCE_FACTOR * (float(sigma_mad) ** 4) / float(n_coefficients)


def compute_sure_mad_uncertainty(
	sigma_mad: float,
	gdf_n: torch.Tensor,
	gdf_se: torch.Tensor,
	n_coefficients: int,
) -> torch.Tensor:
	"""Propagate 1-SE capacity and Wavelet MAD variance uncertainty to SURE_MAD."""
	var_sigma_sq = compute_mad_sigma_sq_variance(sigma_mad, n_coefficients)
	gdf_term = (2.0 * (float(sigma_mad) ** 2) * gdf_se) ** 2
	noise_term = ((2.0 * gdf_n - 1.0) ** 2) * var_sigma_sq
	return torch.sqrt(gdf_term + noise_term)


def compute_mdp_uncertainty(
	morozov_ratio: torch.Tensor,
	gdf_n: torch.Tensor,
	gdf_se: torch.Tensor,
	sigma_mad: float,
	n_coefficients: int,
) -> torch.Tensor:
	"""Propagate 1-SE capacity and Wavelet MAD variance uncertainty to the Morozov ratio."""
	if not math.isfinite(sigma_mad) or sigma_mad <= 0.0:
		raise ValueError(f"sigma_mad must be finite and positive; got {sigma_mad}.")
	var_ratio_sigma = (
		compute_mad_sigma_sq_variance(sigma_mad, n_coefficients) / float(sigma_mad) ** 4
	)
	denom = torch.clamp(1.0 - gdf_n, min=MDP_DENOMINATOR_FLOOR)
	gdf_rel_term = (gdf_se / denom) ** 2
	return morozov_ratio * torch.sqrt(gdf_rel_term + var_ratio_sigma)


# ===========================================================================
# Monte Carlo GDF and flat-solver error accounting
# ===========================================================================


class McGdfResult(NamedTuple):
	"""Monte Carlo GDF estimate at a fixed solver budget K."""

	gdf_n: float
	gdf_n_se: float
	x_hat: torch.Tensor
	k_solver: int


def probe_divergences(b: torch.Tensor, x: torch.Tensor, tau: float) -> torch.Tensor:
	"""Return the divergence of each probe, mean(b_i * (x_i - x_0) / tau).

	This is the estimator of deepinv `mc_div`, evaluated on all probes at once.

	Args:
		b: Probes [M, 1, H, W].
		x: Joint reconstruction [M + 1, 1, H, W] of the batch [y; y + tau * b]; row 0
			is the reconstruction of `y`.
		tau: Finite-difference step of the batch.

	Returns:
		Probe divergences [M], in the dtype and on the device of `x`.
	"""
	m = b.shape[0]
	return (b * (x[1:] - x[0:1]) / tau).reshape(m, -1).mean(1)


@torch.no_grad()
def compute_mc_gdf(
	y: torch.Tensor,
	lambda_tv: float,
	m_mc: int,
	k: int,
	tau: float = DEFAULT_TAU,
	rng: torch.Generator | None = None,
) -> McGdfResult:
	"""Estimate the normalized Monte Carlo GDF at a fixed solver budget K.

	One joint Chambolle-Pock batch [y; y + tau * b] runs for K iterations. The divergence
	of probe i is mean(b_i * (x_i - x_0) / tau), and the standard error is the ddof=1
	sample SD of the probe divergences divided by sqrt(m_mc). The production K comes from
	`solver.solver_k`.

	Args:
		y: Noisy observation tensor of shape [1, 1, H, W] with H == W.
		lambda_tv: Total Variation regularization weight, strictly positive.
		m_mc: Number of Monte Carlo probe realizations, at least 1.
		k: Solver iteration count K, at least 1.
		tau: Finite-difference step, strictly positive.
		rng: Optional common-random-number probe generator; `None` uses the torch global RNG.

	Returns:
		The estimate, its measured standard error, the reconstruction of `y` at K, and K.
		`gdf_n` is the unclamped probe mean; Monte Carlo noise can place it outside `[0, 1]`.

	Raises:
		ValueError: If the shapes, `lambda_tv`, `m_mc`, `k`, or `tau` are invalid.
	"""
	if not isinstance(y, torch.Tensor):
		raise TypeError("y must be a torch.Tensor.")
	if y.ndim != 4 or tuple(y.shape[:2]) != (1, 1):
		raise ValueError(f"Expected y of shape [1, 1, H, W]; got {tuple(y.shape)}.")
	if y.shape[2] != y.shape[3]:
		raise ValueError(f"Expected a square image; got shape {tuple(y.shape)}.")
	if not math.isfinite(lambda_tv) or lambda_tv <= 0.0:
		raise ValueError(f"lambda_tv must be finite and positive; got {lambda_tv}.")
	if type(m_mc) is not int or m_mc < 1:
		raise ValueError(f"m_mc must be a positive integer; got {m_mc!r}.")
	if type(k) is not int or k < 1:
		raise ValueError(f"k must be a positive integer; got {k!r}.")
	if not math.isfinite(tau) or tau <= 0.0:
		raise ValueError(f"tau must be finite and positive; got {tau}.")

	b = torch.empty((m_mc, *y.shape[1:]), device=y.device, dtype=y.dtype).normal_(generator=rng)
	y_all = torch.cat([y, y.repeat(m_mc, 1, 1, 1) + b * tau])
	x = y_all
	for iterate in tv_denoise_iterates(y_all, lambda_tv, k):
		x = iterate
	div = probe_divergences(b, x, tau)
	return McGdfResult(
		gdf_n=float(div.mean()),
		gdf_n_se=float(div.std()) / math.sqrt(m_mc) if m_mc > 1 else 0.0,
		x_hat=x[0:1],
		k_solver=k,
	)


def compute_flat_solver_error(
	k: int | Sequence[int] | torch.Tensor,
	size: int,
	sig: float = DEFAULT_CP_SIGMA,
	tau: float = DEFAULT_CP_TAU,
	theta: float = DEFAULT_CP_THETA,
) -> float | torch.Tensor:
	"""Compute the exact unclipped Chambolle-Pock solver error e(K, L) in the flat region.

	The recurrence runs in float64 because the error spans several decades and changes
	sign. The returned tensor is float32.

	Args:
		k: Solver iteration count K (scalar or sequence).
		size: Spatial image resolution L.
		sig: Chambolle-Pock dual step size sigma.
		tau: Chambolle-Pock primal step size tau.
		theta: Chambolle-Pock over-relaxation parameter.

	Returns:
		Capacity truncation error in normalized capacity units.
	"""
	j = torch.arange(size, dtype=torch.float64)[:, None]
	m = torch.arange(size, dtype=torch.float64)[None, :]
	mu = (2.0 - 2.0 * torch.cos(math.pi * j / size)) + (2.0 - 2.0 * torch.cos(math.pi * m / size))
	n_pixels = size * size

	k_tensor = torch.as_tensor(k)
	k_list = [int(k_tensor)] if k_tensor.ndim == 0 else [int(value) for value in k_tensor]
	max_k = max(k_list)
	k_set = set(k_list)

	x = torch.ones_like(mu)
	z = torch.ones_like(mu)
	u = torch.zeros_like(mu)
	q = torch.zeros_like(mu)

	recorded: dict[int, float] = {}
	for step in range(1, max_k + 1):
		u = (u + sig * (z - 1.0)) / (1.0 + sig)
		q = q + sig * mu * z
		x_prev = x
		x = x - tau * u - tau * q
		z = x + theta * (x - x_prev)
		if step in k_set:
			recorded[step] = float(x[mu > 0].sum()) / n_pixels

	if isinstance(k, int):
		return recorded[k_list[0]]
	result = torch.tensor([recorded[value] for value in k_list], dtype=torch.float32)
	return result if k_tensor.ndim > 0 else result[0]


@functools.cache
def _flat_solver_error_tables() -> dict[int, torch.Tensor]:
	"""Load, validate, and cache the stored flat solver-error tables."""
	payload = load_pt(E_K_TABLE_PATH)
	metadata = payload["metadata"]
	k_max = int(metadata["k_max"])
	for name, expected in (
		("sigma", DEFAULT_CP_SIGMA),
		("tau", DEFAULT_CP_TAU),
		("theta", DEFAULT_CP_THETA),
	):
		if float(metadata[name]) != float(expected):
			raise ValueError(
				f"Stored e_K table has {name} {metadata[name]} != {expected}; "
				"run generate_flat_error_table()."
			)
	tables = {int(size): table for size, table in payload["tables"].items()}
	for size, table in tables.items():
		if table.dtype != torch.float32 or table.shape != (k_max,):
			raise ValueError(f"Stored e_K table for size {size} has the wrong dtype or shape.")
	return tables


def clear_flat_solver_error_table_cache() -> None:
	"""Drop the cached tables so the next lookup reloads the stored artifact."""
	_flat_solver_error_tables.cache_clear()


def flat_solver_error_table(size: int) -> torch.Tensor:
	"""Return the stored flat-region solver error e(K, L) for K = 1..`k_max`.

	The K axis is contiguous, so an iteration count K indexes entry K - 1. The
	returned tensor is shared; treat it as read-only.
	"""
	tables = _flat_solver_error_tables()
	if size not in tables:
		raise ValueError(
			f"No stored e_K table for size {size}; run generate_flat_error_table(sizes=...)."
		)
	return tables[size]


# ===========================================================================
# Selection-excess math (internal benchmark support)
# ===========================================================================


class SelectionExcess(NamedTuple):
	"""Reference-criterion excess at the surrogate-selected candidate."""

	excess: float
	excess_se: float
	excess_z: float
	quadrature_error: float
	correlation: float | None
	linear_se: float
	refinement_relative_change: float


def folded_moments(mean: float, sd: float) -> tuple[float, float]:
	"""Return the mean and variance of the absolute value of a Gaussian variable."""
	if not math.isfinite(mean) or not math.isfinite(sd) or sd < 0.0:
		raise ValueError("Gaussian mean and SD must be finite, with nonnegative SD.")
	if sd == 0.0:
		return abs(mean), 0.0
	absolute_mean = abs(mean)
	standard_mean = absolute_mean / sd
	density = STANDARD_NORMAL_DENSITY_AT_ZERO * math.exp(-0.5 * standard_mean**2)
	correction = 2.0 * (sd * density - absolute_mean * float(ndtr(-standard_mean)))
	folded_mean = absolute_mean + correction
	folded_var = sd**2 - correction * (2.0 * absolute_mean + correction)
	if folded_var < 0.0:
		raise ArithmeticError(f"Negative folded variance: {folded_var}.")
	return folded_mean, folded_var


def folded_difference_variance(
	mean_a: float,
	mean_b: float,
	sd_a: float,
	sd_b: float,
	correlation: float,
	*,
	abs_tol: float = SELECTION_QUADRATURE_ABS_TOL,
	rel_tol: float = SELECTION_QUADRATURE_REL_TOL,
) -> tuple[float, float]:
	"""Integrate Var(|A| - |B|) for a correlated Gaussian pair."""
	if not -1.0 <= correlation <= 1.0:
		raise ValueError(f"Invalid correlation: {correlation}.")
	mean_abs_a, var_abs_a = folded_moments(mean_a, sd_a)
	mean_abs_b, var_abs_b = folded_moments(mean_b, sd_b)
	if sd_a == 0.0:
		return var_abs_b, 0.0
	if sd_b == 0.0:
		return var_abs_a, 0.0
	if correlation == 0.0:
		return var_abs_a + var_abs_b, 0.0
	if correlation == 1.0 and mean_a == mean_b and sd_a == sd_b:
		return 0.0, 0.0
	if correlation == -1.0 and mean_a == -mean_b and sd_a == sd_b:
		return 0.0, 0.0

	scale = max(sd_a, sd_b)
	mu_a, mu_b = mean_a / scale, mean_b / scale
	s_a, s_b = sd_a / scale, sd_b / scale
	mean_difference = (mean_abs_a - mean_abs_b) / scale
	conditional_sd = s_b * math.sqrt((1.0 - correlation) * (1.0 + correlation))

	def integrand(probability: float) -> float:
		standard_a = float(ndtri(probability))
		value_a = mu_a + s_a * standard_a
		conditional_mean_b = mu_b + correlation * s_b * standard_a
		conditional_abs_mean, conditional_abs_var = folded_moments(
			conditional_mean_b, conditional_sd
		)
		conditional_difference = abs(value_a) - conditional_abs_mean
		return conditional_abs_var + (conditional_difference - mean_difference) ** 2

	points = {0.0, 0.5, 1.0, float(ndtr(-mu_a / s_a))}
	points.add(float(ndtr(-mu_b / (correlation * s_b))))
	ordered = sorted(points)
	variance = 0.0
	error = 0.0
	for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
		value, estimate = quad(
			integrand,
			lower,
			upper,
			epsabs=abs_tol,
			epsrel=rel_tol,
			limit=SELECTION_QUADRATURE_LIMIT,
		)
		variance += value
		error += estimate
	if variance < 0.0 or not math.isfinite(variance):
		raise ArithmeticError(f"Invalid integrated variance: {variance}.")
	return variance * scale**2, error * scale**2


def verify_folded_limits() -> dict[str, float]:
	"""Check the folded model against its analytical limits at report time.

	Returns:
		Absolute error of each limit check, keyed by check name.

	Raises:
		AssertionError: If any check misses its analytical value.
	"""
	folded_variance = 1.0 - 2.0 / math.pi
	mean, variance = folded_moments(0.0, 1.0)
	assert math.isclose(mean, math.sqrt(2.0 / math.pi), abs_tol=SELECTION_VERIFICATION_ABS_TOL)
	assert math.isclose(variance, folded_variance, abs_tol=SELECTION_VERIFICATION_ABS_TOL)
	rho = SELECTION_VERIFICATION_CORRELATION
	checks = {
		"independent_at_unity": ((0.0, 0.0, 1.0, 1.0, 0.0), 2.0 * folded_variance),
		"correlated_at_unity": (
			(0.0, 0.0, 1.0, 1.0, rho),
			2.0 - 4.0 / math.pi * (math.sqrt(1.0 - rho**2) + rho * math.asin(rho)),
		),
		"identical": ((0.0, 0.0, 1.0, 1.0, 1.0), 0.0),
		"symmetric_perfect_anticorrelation": ((0.0, 0.0, 1.0, 1.0, -1.0), 0.0),
		"one_deterministic": ((0.0, 0.0, 0.0, 1.0, 0.0), folded_variance),
		"same_side_linear_limit": (
			(SELECTION_VERIFICATION_FAR_MEAN, SELECTION_VERIFICATION_FAR_MEAN, 1.0, 1.0, rho),
			2.0 * (1.0 - rho),
		),
		"opposite_side_linear_limit": (
			(
				SELECTION_VERIFICATION_FAR_MEAN,
				-SELECTION_VERIFICATION_FAR_MEAN,
				1.0,
				1.0,
				rho,
			),
			2.0 * (1.0 + rho),
		),
	}
	errors: dict[str, float] = {}
	for label, (arguments, expected) in checks.items():
		actual, _ = folded_difference_variance(*arguments)
		errors[label] = abs(actual - expected)
		assert math.isclose(
			actual,
			expected,
			rel_tol=SELECTION_VERIFICATION_REL_TOL,
			abs_tol=SELECTION_VERIFICATION_ABS_TOL,
		), (label, actual, expected)
	return errors


def standardized_excess(excess: float, excess_se: float) -> float:
	"""Return D / SE[D], with zero for D = 0 and infinity for D > 0 with SE[D] = 0."""
	if excess == 0.0:
		return 0.0
	if excess_se <= 0.0:
		return math.inf
	return excess / excess_se


def validate_selection_inputs(
	values: tuple[torch.Tensor, ...],
	index_surrogate: int,
	index_reference: int,
) -> tuple[torch.Tensor, ...]:
	"""Require finite 1D float32 tensors of equal length and two in-range node indices."""
	size = values[0].numel()
	for array in values:
		if array.dtype != torch.float32:
			raise TypeError(f"Selection-excess inputs must be float32; got {array.dtype}.")
		if array.ndim != 1 or array.numel() != size:
			raise ValueError("Selection-excess inputs must be 1D tensors of equal length.")
		if not bool(torch.isfinite(array).all()):
			raise ValueError("Selection-excess inputs must be finite.")
	for index in (index_surrogate, index_reference):
		if not 0 <= index < size:
			raise ValueError(f"Selection index {index} is outside the {size}-node grid.")
	return values


def compute_sure_selection_excess(
	sure_mad: torch.Tensor,
	gdf_n: torch.Tensor,
	gdf_se: torch.Tensor,
	sigma_mad: float,
	n_coefficients: int,
	index_surrogate: int,
	index_reference: int,
) -> SelectionExcess:
	"""Propagate the uncertainty of the SURE_MAD excess between two candidates."""
	if not math.isfinite(sigma_mad) or sigma_mad <= 0.0:
		raise ValueError(f"sigma_mad must be finite and positive; got {sigma_mad}.")
	validate_selection_inputs((sure_mad, gdf_n, gdf_se), index_surrogate, index_reference)
	if bool((gdf_se < 0.0).any()):
		raise ValueError("gdf_se must be nonnegative.")
	if index_surrogate == index_reference:
		return SelectionExcess(0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0)
	excess = float(sure_mad[index_surrogate]) - float(sure_mad[index_reference])
	if excess < 0.0:
		raise ValueError(
			f"Negative SURE excess {excess}: the reference index does not minimize the criterion."
		)
	noise_variance = compute_mad_sigma_sq_variance(sigma_mad, n_coefficients)
	capacity_difference = float(gdf_n[index_surrogate]) - float(gdf_n[index_reference])
	se_surrogate = float(gdf_se[index_surrogate])
	se_reference = float(gdf_se[index_reference])
	variance = (2.0 * sigma_mad**2) ** 2 * (
		se_surrogate**2 + se_reference**2
	) + 4.0 * capacity_difference**2 * noise_variance
	excess_se = math.sqrt(variance)
	return SelectionExcess(
		excess=excess,
		excess_se=excess_se,
		excess_z=standardized_excess(excess, excess_se),
		quadrature_error=0.0,
		correlation=None,
		linear_se=0.0,
		refinement_relative_change=0.0,
	)


def compute_mdp_selection_excess(
	morozov_ratio: torch.Tensor,
	gdf_n: torch.Tensor,
	gdf_se: torch.Tensor,
	sigma_mad: float,
	n_coefficients: int,
	index_surrogate: int,
	index_reference: int,
	*,
	target: float = SELECTION_MDP_TARGET,
) -> SelectionExcess:
	"""Propagate the uncertainty of the MDP excess between two candidates."""
	if not math.isfinite(sigma_mad) or sigma_mad <= 0.0:
		raise ValueError(f"sigma_mad must be finite and positive; got {sigma_mad}.")
	if not math.isfinite(target) or target <= 0.0:
		raise ValueError(f"target must be finite and positive; got {target}.")
	validate_selection_inputs((morozov_ratio, gdf_n, gdf_se), index_surrogate, index_reference)
	if bool((gdf_se < 0.0).any()):
		raise ValueError("gdf_se must be nonnegative.")
	if index_surrogate == index_reference:
		return SelectionExcess(0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0)
	for index in (index_surrogate, index_reference):
		capacity = float(gdf_n[index])
		if not 0.0 <= capacity < 1.0:
			raise ValueError(
				f"Normalized capacity r = {capacity!r} at node {index} must lie in [0, 1) "
				"for the first-order ratio propagation."
			)

	ratio_sd = compute_mdp_uncertainty(morozov_ratio, gdf_n, gdf_se, sigma_mad, n_coefficients)
	sd_a = float(ratio_sd[index_surrogate])
	sd_b = float(ratio_sd[index_reference])
	ratio_a = float(morozov_ratio[index_surrogate])
	ratio_b = float(morozov_ratio[index_reference])
	relative_noise_variance = WAVELET_MAD_VARIANCE_FACTOR / float(n_coefficients)
	covariance = ratio_a * ratio_b * relative_noise_variance
	correlation = covariance / (sd_a * sd_b) if sd_a * sd_b > 0.0 else 0.0
	deviations = (ratio_a - target, ratio_b - target)
	variance, quadrature_error = folded_difference_variance(
		deviations[0], deviations[1], sd_a, sd_b, correlation
	)
	refined_variance, _ = folded_difference_variance(
		deviations[0],
		deviations[1],
		sd_a,
		sd_b,
		correlation,
		abs_tol=SELECTION_QUADRATURE_ABS_TOL / SELECTION_QUADRATURE_REFINEMENT_FACTOR,
		rel_tol=SELECTION_QUADRATURE_REL_TOL / SELECTION_QUADRATURE_REFINEMENT_FACTOR,
	)
	excess_se = math.sqrt(variance)
	refinement_change = (
		abs(excess_se - math.sqrt(refined_variance)) / excess_se if excess_se > 0.0 else 0.0
	)
	if refinement_change > SELECTION_VERIFICATION_REL_TOL:
		raise ArithmeticError(
			f"Quadrature refinement moves the MDP excess SE by {refinement_change:.3g} relative."
		)
	swapped_variance, _ = folded_difference_variance(
		deviations[1], deviations[0], sd_b, sd_a, correlation
	)
	if not math.isclose(
		variance,
		swapped_variance,
		rel_tol=SELECTION_VERIFICATION_REL_TOL,
		abs_tol=SELECTION_VERIFICATION_ABS_TOL,
	):
		raise ArithmeticError(f"Node-swapped variance {swapped_variance} differs from {variance}.")

	excess = abs(deviations[0]) - abs(deviations[1])
	if excess < 0.0:
		raise ValueError(
			f"Negative MDP excess {excess}: the reference index does not minimize the criterion."
		)
	sign_a = 0.0 if deviations[0] == 0.0 else math.copysign(1.0, deviations[0])
	sign_b = 0.0 if deviations[1] == 0.0 else math.copysign(1.0, deviations[1])
	linear_se = math.sqrt(sd_a**2 + sd_b**2 - 2.0 * sign_a * sign_b * covariance)
	return SelectionExcess(
		excess=excess,
		excess_se=excess_se,
		excess_z=standardized_excess(excess, excess_se),
		quadrature_error=quadrature_error,
		correlation=correlation,
		linear_se=linear_se,
		refinement_relative_change=refinement_change,
	)


__all__ = [
	"McGdfResult",
	"SelectionExcess",
	"WaveletMADNoiseEstimate",
	"atomic_write",
	"clear_flat_solver_error_table_cache",
	"compute_flat_solver_error",
	"compute_mad_sigma_sq_variance",
	"compute_mc_gdf",
	"compute_mdp_known_sigma_uncertainty",
	"compute_mdp_selection_excess",
	"compute_mdp_uncertainty",
	"compute_regression_metrics",
	"compute_sure",
	"compute_sure_mad_uncertainty",
	"compute_sure_selection_excess",
	"compute_sure_uncertainty",
	"display_path",
	"estimate_noise_wavelet_mad",
	"flat_solver_error_table",
	"folded_difference_variance",
	"folded_moments",
	"load_data_file",
	"load_pt",
	"load_pt_xz",
	"next_free_path",
	"probe_divergences",
	"resolve_model_paths",
	"save_pt",
	"save_pt_xz",
	"standardized_excess",
	"validate_selection_inputs",
	"verify_folded_limits",
]
