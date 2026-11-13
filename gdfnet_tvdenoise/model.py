"""Published neural GDF surrogate architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from gdfnet_tvdenoise.config import DEFAULT_DEVICE
from gdfnet_tvdenoise.helpers import resolve_model_paths, save_pt


class ModelNotFoundError(FileNotFoundError):
	"""Raised when a specified model checkpoint or weights file is not found."""


class SpatialResBlock(nn.Module):
	"""Dilated residual convolutional block with Feature-wise Linear Modulation (FiLM)."""

	def __init__(self, channels: int, dilation: int = 1) -> None:
		"""Initialize the dilated residual block.

		Args:
			channels: Channel dimension.
			dilation: Dilation rate for the first 3x3 convolution.
		"""
		super().__init__()
		self.conv1 = nn.Conv2d(
			channels, channels, kernel_size=3, padding=dilation, dilation=dilation
		)
		self.act1 = nn.SiLU()
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
		self.act2 = nn.SiLU()

	def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
		"""Apply a dilated convolution and FiLM modulation.

		Args:
			x: Input feature map of shape (B, C, H, W).
			gamma: Multiplicative scale modulation of shape (B, C).
			beta: Additive shift modulation of shape (B, C).

		Returns:
			Residual output of shape (B, C, H, W).
		"""
		res = x
		h = self.conv1(x)
		h = h * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
		h = self.act1(h)
		h = self.conv2(h)
		return self.act2(h + res)


class FiLMGenerator(nn.Module):
	"""Multi-layer perceptron generating per-block FiLM modulation parameters."""

	def __init__(self, in_features: int, channels: int, num_blocks: int) -> None:
		"""Initialize the FiLM modulation generator.

		Args:
			in_features: Number of physical input features.
			channels: Feature channel dimension per block.
			num_blocks: Number of residual blocks to modulate.
		"""
		super().__init__()
		self.channels = channels
		self.num_blocks = num_blocks
		self.net = nn.Sequential(
			nn.Linear(in_features, 64),
			nn.SiLU(),
			nn.Linear(64, 2 * channels * num_blocks),
		)

	def forward(self, x_phys: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
		"""Generate (gamma, beta) pairs for each residual block.

		Args:
			x_phys: Normalized physical features of shape (B, in_features).

		Returns:
			List of (gamma, beta) tensors, each of shape (B, channels).
		"""
		params = self.net(x_phys).view(-1, self.num_blocks, 2, self.channels)
		return [(params[:, i, 0, :], params[:, i, 1, :]) for i in range(self.num_blocks)]


class SpatialGDFNet(nn.Module):
	"""Local sensitivity map surrogate predicting dimensionless capacity fraction r in [0, 1].

	Computes a full 2D per-pixel local sensitivity map S(u, v) in [0, 1] through a multi-scale
	dilated convolutional network with FiLM conditioning from physical covariates. The global
	capacity fraction is the spatial trace integral r = (1 / N) * sum_{u, v} S(u, v).
	"""

	def __init__(
		self,
		in_phys_features: int,
		channels: int,
		num_blocks: int,
		dilations: tuple[int, ...],
	) -> None:
		"""Initialize SpatialGDFNet.

		Args:
			in_phys_features: Number of physical scalar features.
			channels: Number of intermediate convolutional channels.
			num_blocks: Number of multi-scale dilated residual blocks.
			dilations: Dilation rate per block. Its length must equal `num_blocks`.
		"""
		super().__init__()
		if len(dilations) != num_blocks:
			raise ValueError(f"Expected {num_blocks} dilations; got {len(dilations)}.")
		self.in_phys_features = in_phys_features
		self.channels = channels
		self.num_blocks = num_blocks
		self.dilations = tuple(dilations)

		self.register_buffer("phys_mean", torch.zeros(in_phys_features, dtype=torch.float32))
		self.register_buffer("phys_std", torch.ones(in_phys_features, dtype=torch.float32))

		self.input_conv = nn.Sequential(
			nn.Conv2d(1 + in_phys_features, channels, kernel_size=3, padding=1),
			nn.SiLU(),
		)
		self.film_gen = FiLMGenerator(in_phys_features, channels, num_blocks)
		self.blocks = nn.ModuleList(
			[SpatialResBlock(channels, dilation=dilation) for dilation in self.dilations]
		)
		self.output_conv = nn.Conv2d(channels, 1, kernel_size=1)

	def fit_normalization(self, x_phys: torch.Tensor) -> None:
		"""Set physical feature normalization statistics from training features."""
		x_t = x_phys.to(torch.float32)
		self.phys_mean.copy_(torch.mean(x_t, dim=0).squeeze())
		self.phys_std.copy_(torch.clamp(torch.std(x_t, dim=0).squeeze(), min=1e-8))

	def forward(self, y: torch.Tensor, x_phys: torch.Tensor) -> torch.Tensor:
		"""Predict dimensionless capacity fraction by spatial trace integration.

		Args:
			y: Observation tensor of shape (B, 1, H, W).
			x_phys: Physical scalar features of shape (B, in_phys_features).

		Returns:
			Predicted dimensionless fraction of shape (B,).
		"""
		x_phys_norm = (x_phys - self.phys_mean) / self.phys_std

		b, _, h, w = y.shape
		phys_spatial = x_phys_norm.view(b, self.in_phys_features, 1, 1).expand(
			b, self.in_phys_features, h, w
		)
		feat = torch.cat([y, phys_spatial], dim=1)

		feat = self.input_conv(feat)
		film_params = self.film_gen(x_phys_norm)
		for block, (gamma, beta) in zip(self.blocks, film_params, strict=True):
			feat = block(feat, gamma, beta)

		s_map = torch.sigmoid(self.output_conv(feat))
		return s_map.mean(dim=(-2, -1)).squeeze(-1)

	def num_forward_parameters(self) -> int:
		"""Count learned parameters."""
		return sum(p.numel() for p in self.parameters())

	def save(self, path: str | Path, training: dict[str, Any]) -> None:
		"""Save model checkpoint and training metadata to disk."""
		save_pt(
			{
				"architecture": self.__class__.__name__,
				"state_dict": self.state_dict(),
				"in_phys_features": self.in_phys_features,
				"channels": self.channels,
				"num_blocks": self.num_blocks,
				"dilations": list(self.dilations),
				"training": training,
			},
			path,
		)

	@classmethod
	def load(cls, path: str | Path, device: torch.device = DEFAULT_DEVICE) -> SpatialGDFNet:
		"""Load a trained SpatialGDFNet checkpoint from disk."""
		path, _ = resolve_model_paths(path)
		if not path.exists():
			raise ModelNotFoundError(f"Checkpoint file not found: {path}")

		ckpt = torch.load(path, map_location=device)
		inst = cls(
			in_phys_features=ckpt["in_phys_features"],
			channels=ckpt["channels"],
			num_blocks=ckpt["num_blocks"],
			dilations=tuple(ckpt["dilations"]),
		)
		inst.to(device)
		inst.load_state_dict(ckpt["state_dict"], strict=True)
		inst.eval()
		return inst


__all__ = [
	"FiLMGenerator",
	"ModelNotFoundError",
	"SpatialGDFNet",
	"SpatialResBlock",
]
