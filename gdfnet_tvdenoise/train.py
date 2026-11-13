"""Surrogate training for the published 2-feature identity-denoising setup."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from gdfnet_tvdenoise.config import (
	DEFAULT_DATASETS,
	DEFAULT_DEVICE,
	DEFAULT_TRAIN_BATCH_SIZE,
	DEFAULT_TRAIN_EPOCHS,
	DEFAULT_TRAIN_ETA_MIN,
	DEFAULT_TRAIN_GRAD_CLIP,
	DEFAULT_TRAIN_LR,
	DEFAULT_TRAIN_SEED,
	DEFAULT_TRAIN_WEIGHT_DECAY,
	MODEL_CHANNELS,
	MODEL_DILATIONS,
	MODEL_NUM_BLOCKS,
	MODELS_DIR,
	TRAIN_BCE_EPS,
	TRAIN_TARGET_MAX,
	TRAIN_TARGET_MIN,
)
from gdfnet_tvdenoise.features import extract_features
from gdfnet_tvdenoise.helpers import display_path, load_data_file, next_free_path
from gdfnet_tvdenoise.model import SpatialGDFNet


def _extract_split_features(
	split_data: dict[str, Any],
	max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Extract image tensors, physical features, and clamped targets for one split."""
	y_values = split_data["observations_y"]
	lam_values = split_data["lambda_tv"]
	target = split_data["gdf_n"].clamp(TRAIN_TARGET_MIN, TRAIN_TARGET_MAX)
	n = len(y_values)
	if max_samples is not None:
		n = min(n, int(max_samples))
	if n < 1:
		raise ValueError("Training split contains no samples.")

	images: list[torch.Tensor] = []
	lams: list[float] = []
	for index in range(n):
		y_t = y_values[index]
		if y_t.ndim == 3:
			y_t = y_t.squeeze(0)
		images.append(y_t.view(1, y_t.shape[-2], y_t.shape[-1]))
		lams.append(float(lam_values[index]))

	image_batch = torch.stack(images, dim=0)
	lam_batch = torch.tensor(lams, dtype=torch.float32)
	return image_batch, extract_features(image_batch, lam_batch), target[:n]


def _resolve_collision_free_path(output: str | Path | None) -> Path:
	"""Resolve the requested output file, appending `_001`, `_002` on collision."""
	if output is None:
		return next_free_path(MODELS_DIR, "model") / "weights.pt"
	path = Path(output)
	if path.is_dir() or path.suffix == "":
		path = path / "weights.pt"
	if not path.exists():
		return path
	stem = path.stem
	suffix = path.suffix
	index = 1
	while True:
		candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
		if not candidate.exists():
			return candidate
		index += 1


def train(
	seed: int | None = DEFAULT_TRAIN_SEED,
	*,
	epochs: int = DEFAULT_TRAIN_EPOCHS,
	batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
	lr: float = DEFAULT_TRAIN_LR,
	weight_decay: float = DEFAULT_TRAIN_WEIGHT_DECAY,
	eta_min: float = DEFAULT_TRAIN_ETA_MIN,
	grad_clip: float = DEFAULT_TRAIN_GRAD_CLIP,
	device: torch.device = DEFAULT_DEVICE,
	output: str | Path | None = None,
	data_path: str | Path | None = None,
	max_samples: int | None = None,
) -> tuple[Path, dict[str, Any]]:
	"""Train a SpatialGDFNet surrogate on the canonical training dataset.

	Defaults reproduce the published setup: 250 epochs, batch 16, AdamW with
	weight decay 1e-4, cosine schedule with eta_min 1e-6, gradient clipping 1.0,
	soft BCE loss, the two canonical features, and 32 channels.

	Args:
		seed: Optional random seed. None leaves initialization unseeded.
		epochs: Number of training epochs.
		batch_size: Minibatch size.
		lr: Learning rate.
		weight_decay: AdamW weight decay.
		eta_min: Minimum cosine learning rate.
		grad_clip: Gradient-norm clipping threshold.
		device: Execution device.
		output: Optional explicit output file or directory. Existing files receive
			`_001`, `_002`, ... suffixes and are never overwritten.
		data_path: Optional explicit training data path.
		max_samples: Optional per-split sample cap for smoke runs.

	Returns:
		Tuple of (absolute checkpoint path, metrics dictionary).
	"""
	if seed is not None:
		torch.manual_seed(seed)
	data_file = Path(data_path) if data_path is not None else DEFAULT_DATASETS["training"][0]
	payload = load_data_file(data_file)
	with data_file.open("rb") as handle:
		data_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
	if "train" not in payload or "val" not in payload:
		raise ValueError(f"Training data must contain 'train' and 'val' splits: {data_file}")

	img_train, x_train, y_train = _extract_split_features(payload["train"], max_samples)
	img_val, x_val, y_val = _extract_split_features(payload["val"], max_samples)
	in_features = x_train.shape[1]

	train_ds = TensorDataset(img_train, x_train, y_train)
	val_ds = TensorDataset(img_val, x_val, y_val)
	order_generator = None
	if seed is not None:
		order_generator = torch.Generator(device="cpu")
		order_generator.manual_seed(seed)
	train_loader = DataLoader(
		train_ds, batch_size=batch_size, shuffle=True, generator=order_generator
	)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

	model = SpatialGDFNet(
		in_phys_features=in_features,
		channels=MODEL_CHANNELS,
		num_blocks=MODEL_NUM_BLOCKS,
		dilations=MODEL_DILATIONS,
	)
	model.fit_normalization(x_train)
	model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

	best_val_loss = float("inf")
	best_epoch = 0
	best_weights: dict[str, torch.Tensor] | None = None
	start_time = time.perf_counter()

	for epoch in range(epochs):
		model.train()
		train_loss = 0.0
		for batch_img, batch_x, batch_y in train_loader:
			optimizer.zero_grad()
			batch_img = batch_img.to(device)
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			pred = model(batch_img, batch_x)
			loss = F.binary_cross_entropy(pred.clamp(TRAIN_BCE_EPS, 1 - TRAIN_BCE_EPS), batch_y)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			optimizer.step()
			train_loss += loss.item() * len(batch_y)
		train_loss /= len(train_ds)
		scheduler.step()

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for batch_img, batch_x, batch_y in val_loader:
				batch_img = batch_img.to(device)
				batch_x = batch_x.to(device)
				batch_y = batch_y.to(device)
				pred = model(batch_img, batch_x)
				val_loss += F.binary_cross_entropy(
					pred.clamp(TRAIN_BCE_EPS, 1 - TRAIN_BCE_EPS), batch_y
				).item() * len(batch_y)
		val_loss = val_loss / len(val_ds)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch + 1
			best_weights = {key: value.cpu().clone() for key, value in model.state_dict().items()}

		if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
			print(
				f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {train_loss:.5f} | "
				f"Val Loss: {val_loss:.5f}",
				flush=True,
			)

	total_seconds = float(time.perf_counter() - start_time)
	if best_weights is not None:
		model.load_state_dict({key: value.to(device) for key, value in best_weights.items()})

	metrics = {
		"seed": seed,
		"data_file": display_path(data_file),
		"data_sha256": data_sha256,
		"train_samples": len(train_ds),
		"val_samples": len(val_ds),
		"max_samples": max_samples,
		"epochs": epochs,
		"batch_size": batch_size,
		"lr": lr,
		"weight_decay": weight_decay,
		"eta_min": eta_min,
		"grad_clip": grad_clip,
		"best_epoch": best_epoch,
		"best_val_loss": best_val_loss,
		"device": device.type,
		"total_training_wall_time_seconds": total_seconds,
		"time_per_epoch_ms": (total_seconds / epochs) * 1000.0 if epochs > 0 else 0.0,
		"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
	}
	output_path = _resolve_collision_free_path(output)
	model.save(output_path, training=metrics)
	print(f"Saved checkpoint: {output_path.resolve()}")
	return output_path.resolve(), metrics


__all__ = ["train"]
