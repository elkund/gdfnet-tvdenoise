"""Inference helpers for image-level GDF prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gdfnet_tvdenoise.config import DEFAULT_MODEL_PATH, PREDICT_CHUNK_SIZE
from gdfnet_tvdenoise.features import extract_features
from gdfnet_tvdenoise.helpers import compute_mdp, compute_sure
from gdfnet_tvdenoise.model import SpatialGDFNet


def _as_image_tensor(y: torch.Tensor | np.ndarray) -> torch.Tensor:
	"""Return a single 2D image as a strict [1, 1, H, W] float32 tensor."""
	y_t = torch.as_tensor(y, dtype=torch.float32)
	if y_t.ndim == 2:
		return y_t.unsqueeze(0).unsqueeze(0)
	if y_t.ndim == 3 and y_t.shape[0] == 1:
		return y_t.unsqueeze(0)
	if y_t.ndim == 4 and y_t.shape[0] == 1 and y_t.shape[1] == 1:
		return y_t
	raise ValueError(f"Expected one grayscale image; got shape {tuple(y_t.shape)}.")


def predict(
	model: SpatialGDFNet,
	y: torch.Tensor,
	lambdas: float | torch.Tensor,
	chunk_size: int = PREDICT_CHUNK_SIZE,
	*,
	feats: torch.Tensor | None = None,
) -> torch.Tensor:
	"""Predict the normalized GDF for a batch of observations.

	Features and the network run on the model's device. The network runs in chunks.

	Args:
		model: Surrogate network.
		y: Observation batch of shape [B, 1, H, W].
		lambdas: TV regularization parameter, a scalar or a [B] vector.
		chunk_size: Maximum batch size per network call.
		feats: Precomputed feature tensor of shape [B, 2] in float32. When omitted,
			the features are computed from `y` and `lambdas`.

	Returns:
		CPU tensor of shape [B].
	"""
	y = y.to(next(model.parameters()).device, torch.float32)
	chunks: list[torch.Tensor] = []
	with torch.inference_mode():
		feats = extract_features(y, lambdas) if feats is None else feats.to(y.device, y.dtype)
		for start in range(0, len(y), chunk_size):
			stop = start + chunk_size
			chunks.append(model(y[start:stop], feats[start:stop]).flatten())
	return torch.cat(chunks).cpu()


def predict_gdf(
	y: torch.Tensor | np.ndarray,
	lambda_tv: float | torch.Tensor,
	model: SpatialGDFNet | str | Path | None = None,
) -> torch.Tensor:
	"""Predict the normalized GDF r = GDF / N for one image and TV strength.

	Args:
		y: Observation image, shaped [H, W], [1, H, W], or [1, 1, H, W].
		lambda_tv: TV regularization parameter.
		model: Optional model instance or checkpoint path. Defaults to
			`config.DEFAULT_MODEL_PATH`.

	Returns:
		Normalized GDF tensor of shape [1], finite and in [0, 1].
	"""
	net = (
		model
		if isinstance(model, SpatialGDFNet)
		else SpatialGDFNet.load(DEFAULT_MODEL_PATH if model is None else model)
	)
	net.eval()
	return predict(net, _as_image_tensor(y), torch.as_tensor(lambda_tv, dtype=torch.float32))


__all__ = [
	"compute_mdp",
	"compute_sure",
	"predict",
	"predict_gdf",
]
