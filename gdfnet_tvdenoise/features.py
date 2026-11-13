"""Physical feature extraction for the 2-feature identity-denoising path."""

from __future__ import annotations

import math

import torch

from gdfnet_tvdenoise.config import FEATURE_TIKHONOV_DENOM_FLOOR


def image_std(img: torch.Tensor) -> float:
	"""Return the population (ddof=0) standard deviation over all pixels of an image."""
	return float(img.std(correction=0).item())


def extract_features(y: torch.Tensor, lambda_tv: float | torch.Tensor) -> torch.Tensor:
	"""Compute the physical feature tensor `[log10(lambda), Tikhonov capacity / N]`.

	The columns are `log_lam_raw` and `tikhonov_capacity_normalized`. The Tikhonov capacity is the trace of
	the linear Tikhonov smoother (I + lambda * L)^-1 for the discrete Laplacian L.

	Args:
		y: Observation batch of shape [B, 1, H, W].
		lambda_tv: TV regularization parameter. A scalar is shared by the batch. A
			vector has shape [B].

	Returns:
		Feature tensor of shape [B, 2] in the dtype and device of `y`.
	"""
	if y.ndim != 4 or y.shape[1] != 1:
		raise ValueError(f"Expected observations of shape [B, 1, H, W]; got {tuple(y.shape)}.")
	b, _, h, w = y.shape
	lam = torch.as_tensor(lambda_tv, dtype=y.dtype, device=y.device).reshape(-1)
	if lam.numel() not in (1, b):
		raise ValueError(f"Expected a scalar or {b} lambda values; got {lam.numel()}.")
	lam = lam.expand(b)

	log_lam = torch.log10(lam)

	u = torch.arange(h, dtype=y.dtype, device=y.device)
	v = torch.arange(w, dtype=y.dtype, device=y.device)
	grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
	lap_power = (
		4.0 * torch.sin(math.pi * grid_u / h) ** 2 + 4.0 * torch.sin(math.pi * grid_v / w) ** 2
	)
	denom = 1.0 + lam.view(b, 1, 1) * lap_power.unsqueeze(0)
	trace = torch.where(denom > FEATURE_TIKHONOV_DENOM_FLOOR, 1.0 / denom, torch.zeros_like(denom))
	n_pixels = float(h * w)
	capacity = torch.clamp(torch.sum(trace, dim=(-2, -1)), 0.0, n_pixels)
	return torch.stack([log_lam, torch.clamp(capacity / n_pixels, 0.0, 1.0)], dim=-1)


__all__ = ["extract_features", "image_std"]
