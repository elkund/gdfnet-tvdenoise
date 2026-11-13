"""Parity benchmark for capacity scatter datasets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from math import isqrt
from pathlib import Path
from typing import Any

import torch

from gdfnet_tvdenoise.config import (
	CANONICAL_MEDMNIST_DATASETS,
	CANONICAL_MEDMNIST_SIZES,
	DEFAULT_DATASETS,
	DEFAULT_DEVICE,
)
from gdfnet_tvdenoise.features import image_std
from gdfnet_tvdenoise.helpers import (
	compute_regression_metrics,
	display_path,
	load_data_file,
	release_device_cache,
	resolve_model_paths,
	save_pt,
)
from gdfnet_tvdenoise.inference import predict
from gdfnet_tvdenoise.model import SpatialGDFNet


def _per_sample_noise_ratio(clean_images: list[torch.Tensor], sigmas: torch.Tensor) -> torch.Tensor:
	"""Recover per-sample eta = sigma / std(x_clean), with the population std the generator uses."""
	stds = torch.tensor([image_std(image) for image in clean_images])
	if (
		stds.shape != sigmas.shape
		or not bool(torch.isfinite(stds).all())
		or bool((stds <= 0.0).any())
	):
		raise ValueError("Parity noise ratio is not computable from the loaded clean images.")
	return sigmas / stds


def _modality_metrics(
	true_r: torch.Tensor,
	pred_r: torch.Tensor,
	n_pixels: torch.Tensor,
	labels: list[str],
	nsr: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
	"""Compute each modality's R2/MAE per scale and empirical eta ranges per row."""
	if len(labels) != true_r.shape[0] or nsr.shape != true_r.shape:
		raise ValueError("Modality labels and noise ratios must align with the prediction vectors.")
	by_scale: dict[str, Any] = {}
	for pixels in torch.unique(n_pixels):
		side = isqrt(int(pixels))
		if side * side != int(pixels) or side not in CANONICAL_MEDMNIST_SIZES:
			raise ValueError(f"Unsupported parity pixel count: {int(pixels)}.")
		by_modality: dict[str, Any] = {}
		for modality in CANONICAL_MEDMNIST_DATASETS:
			mask = (n_pixels == pixels) & torch.tensor([name == modality for name in labels])
			if not bool(mask.any()):
				raise ValueError(f"No parity samples for {modality} at {side} pixels.")
			metrics = compute_regression_metrics(true_r[mask], pred_r[mask])
			by_modality[modality] = {
				"r2": metrics["r2"],
				"mae_pct": metrics["mae"] * 100,
			}
		by_scale[f"{side}x{side}"] = by_modality
	by_modality_nsr: dict[str, Any] = {}
	for modality in CANONICAL_MEDMNIST_DATASETS:
		mask = torch.tensor([name == modality for name in labels])
		if not bool(mask.any()):
			raise ValueError(f"No parity samples for {modality}.")
		by_modality_nsr[modality] = {
			"nsr_range": [float(nsr[mask].min()), float(nsr[mask].max())],
		}
	overall_nsr = {"nsr_range": [float(nsr.min()), float(nsr.max())]}
	return by_scale, by_modality_nsr, overall_nsr


def _predict(
	model: SpatialGDFNet,
	observations: Sequence[torch.Tensor],
	lambdas: torch.Tensor,
) -> torch.Tensor:
	"""Run surrogate inference sample by sample over one or more datasets."""
	predictions = [
		predict(model, y.unsqueeze(0) if y.ndim == 3 else y, lambdas[index])
		for index, y in enumerate(observations)
	]
	release_device_cache()
	return torch.cat(predictions)


def _source_dataset_name(path: Path) -> str:
	"""Return the `name` in a `{name}_00N.pt` or `{name}_00N.pt.xz` dataset filename."""
	name = path.name
	if name.endswith(".pt.xz"):
		name = name.removesuffix(".pt.xz")
	elif name.endswith(".pt"):
		name = name.removesuffix(".pt")
	stem, separator, index = name.rpartition("_")
	if separator and index.isdigit():
		return stem
	return name


def evaluate_parity(
	model_path: str | Path,
	suite_name: str = "parity",
	device: torch.device = DEFAULT_DEVICE,
) -> tuple[dict[str, Any], Path]:
	"""Evaluate a surrogate model on capacity parity datasets with one report schema.

	Every parity report uses the same keys. `split` is `validation` for `val_parity`
	and `test` for the other suites. `scale_parity` aggregates the parity and
	multi-scale scatter data by default.

	Args:
		model_path: Model directory or weights file.
		suite_name: Report suite name ('parity', 'scale_parity', or 'val_parity').
		device: PyTorch device.

	Returns:
		Tuple of (report payload, report path).
	"""
	weights, model_dir = resolve_model_paths(model_path)
	model = SpatialGDFNet.load(weights, device=device)

	if suite_name not in DEFAULT_DATASETS:
		raise ValueError(
			f"Unknown parity suite {suite_name!r}. Available: {list(DEFAULT_DATASETS)}"
		)

	all_observations: list[torch.Tensor] = []
	all_true_r: list[torch.Tensor] = []
	all_npix: list[torch.Tensor] = []
	all_sigmas: list[torch.Tensor] = []
	all_lams: list[torch.Tensor] = []
	all_names: list[str] = []
	all_clean: list[torch.Tensor] = []
	source_datasets: list[dict[str, str]] = []
	for path in DEFAULT_DATASETS[suite_name]:
		arrays = load_data_file(path)
		all_observations.extend(arrays["observations_y"])
		all_true_r.append(arrays["gdf_n"])
		all_npix.append(arrays["n_pixels"])
		all_sigmas.append(arrays["sigma"])
		all_lams.append(arrays["lambda_tv"])
		all_names.extend(arrays["dataset_names"])
		all_clean.extend(arrays["x_clean"])
		source_datasets.append(
			{"dataset_name": _source_dataset_name(path), "test_data": display_path(path)}
		)

	true_r = torch.cat(all_true_r)
	n_pixels = torch.cat(all_npix)
	sigmas = torch.cat(all_sigmas)
	lams = torch.cat(all_lams)
	pred_r = _predict(model, all_observations, lams)

	metrics = compute_regression_metrics(true_r, pred_r)
	r2 = float(metrics["r2"])
	mae_pct = metrics["mae"] * 100
	rmse = float(metrics["rmse"])
	max_err = torch.max(torch.abs(true_r - pred_r)).item() * 100
	summary_metrics: dict[str, Any] = {
		"r2": r2,
		"mae_pct": mae_pct,
		"rmse": rmse,
		"max_error": max_err,
	}
	unique_scales = sorted({int(value) for value in n_pixels})
	if len(unique_scales) > 1:
		by_scale: dict[str, Any] = {}
		for npix in unique_scales:
			mask = n_pixels == npix
			scale_metrics = compute_regression_metrics(true_r[mask], pred_r[mask])
			side = isqrt(npix)
			by_scale[f"{side}x{side}"] = {
				"r2": float(scale_metrics["r2"]),
				"mae_pct": scale_metrics["mae"] * 100,
			}
		summary_metrics["by_scale"] = by_scale

	labels = all_names
	nsr = _per_sample_noise_ratio(all_clean, sigmas)
	by_scale_modality, by_modality_nsr, overall_nsr = _modality_metrics(
		true_r, pred_r, n_pixels, labels, nsr
	)
	summary_metrics["by_scale_modality"] = by_scale_modality
	summary_metrics["by_modality_nsr"] = by_modality_nsr
	summary_metrics["overall_nsr"] = overall_nsr

	test_data = [entry["test_data"] for entry in source_datasets]
	payload = {
		"suite_name": suite_name,
		"model": model_dir.name,
		"evaluated_at": datetime.now(UTC).isoformat(),
		"n_samples": len(true_r),
		"summary_metrics": summary_metrics,
		"dataset_names": labels,
		"true_r": true_r,
		"pred_r": pred_r,
		"sigmas": sigmas,
		"n_pixels": n_pixels,
		"sample_indices": torch.arange(len(true_r)),
		"split": "validation" if suite_name == "val_parity" else "test",
		"test_data": test_data[0] if len(test_data) == 1 else test_data,
		"source_datasets": source_datasets,
	}
	report_path = model_dir / f"{suite_name}.pt"
	save_pt(payload, report_path)
	print(f"Completed {suite_name}: model={model_dir.name} | R2={r2:.4f} | MAE={mae_pct:.2f}%")
	return payload, report_path


__all__ = ["evaluate_parity"]
