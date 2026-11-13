"""Solver budget calibration: the flat-region e_K table and the K calibration."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from gdfnet_tvdenoise.config import (
	CALIBRATION_IMAGES,
	CANONICAL_MEDMNIST_SIZES,
	DEFAULT_CP_SIGMA,
	DEFAULT_CP_TAU,
	DEFAULT_CP_THETA,
	DEFAULT_DEVICE,
	DEFAULT_M_BY_SIZE,
	DEFAULT_METROLOGY_DELTA_K,
	DEFAULT_METROLOGY_NSR_TIERS,
	DEFAULT_TAU,
	E_K_TABLE_K_MAX,
	E_K_TABLE_PATH,
	GENERATION_CHECKPOINT_SUFFIX,
	K_CAL_MAX_K,
	K_CAL_PROBES,
	K_CAL_TAU_REF,
	K_CALIBRATION_PATH,
	METROLOGY_DOMINANCE_RATIO,
	SOLVER_K_TIERS,
)
from gdfnet_tvdenoise.data import create_noisy_observation, load_medmnist_images
from gdfnet_tvdenoise.features import image_std
from gdfnet_tvdenoise.helpers import (
	clear_flat_solver_error_table_cache,
	compute_flat_solver_error,
	estimate_noise_wavelet_mad,
	flat_solver_error_table,
	load_pt,
	probe_divergences,
	save_pt,
)
from gdfnet_tvdenoise.solver import tv_denoise_iterates


def generate_flat_error_table(
	*,
	sizes: Sequence[int] = CANONICAL_MEDMNIST_SIZES,
	k_max: int | None = None,
	output: str | Path | None = None,
) -> dict[str, Any]:
	"""Precompute the flat-region solver error e(K, L) for K = 1..`k_max`.

	Writes `data/calibration/e_k_table.pt`, a table per spatial size. The K calibration
	loads this table instead of running the recurrence.

	Args:
		sizes: Spatial resolutions to tabulate.
		k_max: Last tabulated iteration count. Defaults to `E_K_TABLE_K_MAX`.
		output: Optional explicit output path.

	Returns:
		The written payload.
	"""
	if k_max is None:
		k_max = E_K_TABLE_K_MAX
	k = torch.arange(1, k_max + 1)
	payload = {
		"tables": {int(size): compute_flat_solver_error(k, int(size)) for size in sizes},
		"metadata": {
			"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
			"sizes": [int(size) for size in sizes],
			"k_min": 1,
			"k_max": k_max,
			"sigma": float(DEFAULT_CP_SIGMA),
			"tau": float(DEFAULT_CP_TAU),
			"theta": float(DEFAULT_CP_THETA),
		},
	}
	path = Path(output) if output is not None else E_K_TABLE_PATH
	save_pt(payload, path)
	clear_flat_solver_error_table_cache()
	print(f"Saved flat solver-error table: {path.resolve()}")
	return payload


# ===========================================================================
# K calibration
# ===========================================================================


@torch.no_grad()
def _calibrate_configuration(
	y: torch.Tensor,
	sigma_mad: float,
	m_probes: int,
	m_prod: int,
	e_k_table: torch.Tensor,
	device: torch.device,
) -> dict[str, Any]:
	"""Run one configuration to its K* with the float32 solve and the float64 reference.

	Both solves share the probes and step in lockstep. The float32 batch [y; y + tau b]
	runs on `device` at `DEFAULT_TAU`, as the labels do. The float64 batch runs on the
	CPU at `K_CAL_TAU_REF`. At every checkpoint, E_tau is the probe mean of the float32
	minus the float64 divergences, and the run stops at the first K with
	e(K) + |E_tau| <= r * SE_prod, with SE_prod = s_hat / sqrt(M_prod).

	Raises:
		RuntimeError: If no checkpoint up to `K_CAL_MAX_K` passes.
	"""
	b = torch.randn((m_probes, *y.shape[1:]), dtype=torch.float32)
	b_dev = b.to(device)
	y_dev = y.to(device)
	batch32 = torch.cat([y_dev, y_dev.repeat(m_probes, 1, 1, 1) + b_dev * DEFAULT_TAU])
	b64 = b.double()
	y64 = y.detach().cpu().double()
	batch64 = torch.cat([y64, y64.repeat(m_probes, 1, 1, 1) + b64 * K_CAL_TAU_REF])
	solves = zip(
		tv_denoise_iterates(batch32, sigma_mad, K_CAL_MAX_K),
		tv_denoise_iterates(batch64, sigma_mad, K_CAL_MAX_K),
		strict=True,
	)
	for k, (x32, x64) in enumerate(solves, start=1):
		if k % DEFAULT_METROLOGY_DELTA_K != 0:
			continue
		div32 = probe_divergences(b_dev, x32, DEFAULT_TAU).cpu().double()
		div64 = probe_divergences(b64, x64, K_CAL_TAU_REF)
		s_probe = float(div32.std())
		se_prod = s_probe / math.sqrt(m_prod)
		e_tau = float((div32 - div64).mean())
		e_k = float(e_k_table[k - 1])
		if e_k + abs(e_tau) <= METROLOGY_DOMINANCE_RATIO * se_prod:
			return {
				"k_star": k,
				"gdf_n": float(div32.mean()),
				"s_probe": s_probe,
				"se_prod": se_prod,
				"e_k": e_k,
				"e_tau": e_tau,
			}
	raise RuntimeError(f"No checkpoint K <= {K_CAL_MAX_K} passed the calibration test.")


def calibrate_k(
	*,
	images: dict[tuple[int, str], list[tuple[str, int]]] = CALIBRATION_IMAGES,
	nsr_tiers: Sequence[float] = DEFAULT_METROLOGY_NSR_TIERS,
	device: torch.device = DEFAULT_DEVICE,
	output: str | Path | None = None,
	archive_dir: str | Path | None = None,
	fresh: bool = False,
) -> dict[str, Any]:
	"""Measure the calibrated solver budget K* on the calibration images.

	Each image is measured at every noise tier, with sigma = eta * std(x_clean), y from
	`create_noisy_observation` (unseeded), lambda = sigma_MAD measured on y, and
	`K_CAL_PROBES` fresh probes. K* is the first checkpoint with
	e(K) + |E_tau| <= `METROLOGY_DOMINANCE_RATIO` * SE_prod; see
	`_calibrate_configuration`. The tier values of `SOLVER_K_TIERS` are the largest K*
	over the images at each (size, tier).

	The run is resumable. A checkpoint beside the output is saved after every
	configuration. `fresh=True` discards an existing checkpoint.

	Args:
		images: Image dict keyed by `(size, modality)`. Defaults to `CALIBRATION_IMAGES`.
		nsr_tiers: Noise-to-signal ratio tiers.
		device: Device of the float32 solve. The float64 reference runs on the CPU.
		output: Optional explicit output path. Defaults to `K_CALIBRATION_PATH`.
		archive_dir: Optional explicit archive directory.
		fresh: Whether to discard an existing checkpoint.

	Returns:
		The calibration payload: records, tier values per size, and metadata.
	"""
	output_path = Path(output) if output is not None else K_CALIBRATION_PATH
	checkpoint_path = output_path.with_name(output_path.name + GENERATION_CHECKPOINT_SUFFIX)
	signature = {
		"images": {
			f"{int(size)}|{modality}": [[split, int(index)] for split, index in image_list]
			for (size, modality), image_list in images.items()
		},
		"nsr_tiers": [float(tier) for tier in nsr_tiers],
		"m_probes": K_CAL_PROBES,
		"tau": DEFAULT_TAU,
		"tau_ref": K_CAL_TAU_REF,
		"ratio": METROLOGY_DOMINANCE_RATIO,
		"delta_k": DEFAULT_METROLOGY_DELTA_K,
		"m_by_size": {int(size): int(m) for size, m in DEFAULT_M_BY_SIZE.items()},
	}
	if fresh:
		checkpoint_path.unlink(missing_ok=True)
	records: list[dict[str, Any]] = []
	elapsed_before = 0.0
	if checkpoint_path.exists():
		state = load_pt(checkpoint_path)
		if state.get("signature") != signature:
			raise ValueError(
				f"Checkpoint {checkpoint_path} was written for a different calibration "
				"configuration; delete it or use fresh=True to start over."
			)
		records = list(state["records"])
		elapsed_before = float(state["elapsed_seconds"])
		print(f"[resume] K calibration: {len(records)} records from {checkpoint_path}")
	done = {(r["size"], r["modality"], r["split"], r["image_idx"], r["nsr"]) for r in records}
	n_total = sum(len(image_list) for image_list in images.values()) * len(nsr_tiers)
	job_start = time.perf_counter()

	for (size, modality), image_list in images.items():
		size = int(size)
		e_k_table = flat_solver_error_table(size)
		if e_k_table.numel() < K_CAL_MAX_K:
			raise ValueError(f"The e_K table for size {size} is shorter than K_CAL_MAX_K.")
		m_prod = DEFAULT_M_BY_SIZE[size]
		for split, image_idx in image_list:
			x_clean = load_medmnist_images(
				modality, split=split, indices=image_idx, size=size, archive_dir=archive_dir
			).to(device)
			std_x = image_std(x_clean)
			for nsr in nsr_tiers:
				key = (size, str(modality), str(split), int(image_idx), float(nsr))
				if key in done:
					continue
				start = time.perf_counter()
				sigma = float(nsr) * std_x
				y = create_noisy_observation(x_clean, sigma)
				sigma_mad = float(estimate_noise_wavelet_mad(y).sigma)
				if sigma_mad <= 0.0:
					raise ValueError(f"Wavelet MAD noise estimate is not positive for {key}.")
				result = _calibrate_configuration(
					y, sigma_mad, K_CAL_PROBES, m_prod, e_k_table, device
				)
				records.append(
					{
						"size": size,
						"modality": str(modality),
						"split": str(split),
						"image_idx": int(image_idx),
						"nsr": float(nsr),
						"std_x": std_x,
						"sigma": sigma,
						"sigma_mad": sigma_mad,
						**result,
						"seconds": time.perf_counter() - start,
					}
				)
				save_pt(
					{
						"signature": signature,
						"records": records,
						"elapsed_seconds": elapsed_before + (time.perf_counter() - job_start),
					},
					checkpoint_path,
				)
				print(
					f"[{len(records)}/{n_total}] {size} {modality} {split} {image_idx} "
					f"eta={nsr}: K*={result['k_star']} "
					f"|E_tau|/SE={abs(result['e_tau']) / result['se_prod']:.3f} "
					f"({records[-1]['seconds']:.0f} s)",
					flush=True,
				)

	k_tiers: dict[int, tuple[int, ...]] = {}
	for size in dict.fromkeys(int(size) for size, _ in images):
		k_tiers[size] = tuple(
			max(r["k_star"] for r in records if r["size"] == size and r["nsr"] == float(nsr))
			for nsr in nsr_tiers
		)
	payload = {
		"records": records,
		"k_tiers": k_tiers,
		"metadata": {
			**signature,
			"n_records": len(records),
			"elapsed_seconds": elapsed_before + (time.perf_counter() - job_start),
			"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
		},
	}
	save_pt(payload, output_path)
	checkpoint_path.unlink(missing_ok=True)
	print(f"Saved K calibration: {output_path.resolve()}")
	print("SOLVER_K_TIERS: dict[int, tuple[int, ...]] = {")
	for size, ks in k_tiers.items():
		print(f"\t{size}: {ks},")
	print("}")
	if k_tiers != {size: SOLVER_K_TIERS.get(size) for size in k_tiers}:
		print("These tier values differ from SOLVER_K_TIERS in config.py.")
	return payload


__all__ = [
	"calibrate_k",
	"generate_flat_error_table",
]
