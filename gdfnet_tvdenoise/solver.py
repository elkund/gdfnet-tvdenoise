"""Fixed-budget Chambolle-Pock TV solver."""

from __future__ import annotations

import math
from collections.abc import Iterator

import deepinv as dinv
import torch

from gdfnet_tvdenoise.config import (
	DEFAULT_CP_SIGMA,
	DEFAULT_CP_TAU,
	DEFAULT_CP_THETA,
	DEFAULT_METROLOGY_DELTA_K,
	DEFAULT_METROLOGY_NSR_TIERS,
	SOLVER_K_TIERS,
)

__all__ = [
	"solver_k",
	"tv_denoise",
	"tv_denoise_iterates",
]

#: Stateless DeepInverse TV operators (nabla, nabla_adjoint, prox_sigma_g_conj).
_TV = dinv.models.tv.TVDenoiser()


def solver_k(size: int, nsr: float) -> int:
	"""Return the calibrated solver budget K(L, eta) of a label.

	K is linear in log eta between the tier values `SOLVER_K_TIERS[size]` at the noise
	tiers `DEFAULT_METROLOGY_NSR_TIERS`, rounded up to a multiple of
	`DEFAULT_METROLOGY_DELTA_K`.

	Args:
		size: Spatial image resolution L.
		nsr: Noise-to-signal ratio eta = sigma / std(x_clean), inside the tier range.

	Returns:
		The iteration count K.

	Raises:
		ValueError: If `size` has no calibrated budget, the tier table does not match the
			noise tiers, or `nsr` lies outside the tier range.
	"""
	if size not in SOLVER_K_TIERS:
		raise ValueError(f"No calibrated solver budget for size {size}.")
	ks = SOLVER_K_TIERS[size]
	tiers = DEFAULT_METROLOGY_NSR_TIERS
	if len(ks) != len(tiers) or len(tiers) < 2:
		raise ValueError(f"SOLVER_K_TIERS[{size}] does not match DEFAULT_METROLOGY_NSR_TIERS.")
	if not tiers[0] <= nsr <= tiers[-1]:
		raise ValueError(f"nsr {nsr} lies outside the calibrated range [{tiers[0]}, {tiers[-1]}].")
	log_eta = math.log(nsr)
	j = next(i for i in range(len(tiers) - 1) if log_eta <= math.log(tiers[i + 1]))
	lo, hi = math.log(tiers[j]), math.log(tiers[j + 1])
	k = ks[j] + (ks[j + 1] - ks[j]) * (log_eta - lo) / (hi - lo)
	# The 1e-9 guard keeps a tier value on the grid despite floating-point rounding.
	return math.ceil(k / DEFAULT_METROLOGY_DELTA_K - 1e-9) * DEFAULT_METROLOGY_DELTA_K


def tv_denoise_iterates(
	y: torch.Tensor,
	lambda_tv: float,
	k: int,
) -> Iterator[torch.Tensor]:
	"""Yield the estimate after each of k fixed Chambolle-Pock iterations.

	This is the recursion of brief eq. 9: it starts from `x = z = y` and `u = p = 0`,
	and uses `DEFAULT_CP_TAU`, `DEFAULT_CP_SIGMA`, and `DEFAULT_CP_THETA`. The state is
	carried across iterations, so one run of the generator is one continuous solve.

	Args:
		y: Noisy observation tensor [B, 1, H, W].
		lambda_tv: Total Variation regularization weight.
		k: Fixed iteration budget.

	Yields:
		The reconstruction after each iteration, [B, 1, H, W].
	"""
	x = y.clone()
	z = x.clone()
	u = torch.zeros_like(y)
	p = torch.zeros((*y.shape, 2), device=y.device, dtype=y.dtype)
	for _ in range(k):
		u = (u + DEFAULT_CP_SIGMA * z - DEFAULT_CP_SIGMA * y) / (1.0 + DEFAULT_CP_SIGMA)
		if lambda_tv > 0.0:
			p = _TV.prox_sigma_g_conj(p + DEFAULT_CP_SIGMA * _TV.nabla(z), lambda_tv)
		x_prev = x
		x = x - DEFAULT_CP_TAU * u - DEFAULT_CP_TAU * _TV.nabla_adjoint(p)
		z = x + DEFAULT_CP_THETA * (x - x_prev)
		yield x


def tv_denoise(y: torch.Tensor, lambda_tv: float, k: int) -> torch.Tensor:
	"""Run K fixed Chambolle-Pock iterations of 2D TV denoising (identity forward operator).

	Args:
		y: Noisy observation tensor [B, 1, H, W].
		lambda_tv: Total Variation regularization weight, strictly positive.
		k: Fixed iteration budget.

	Returns:
		Reconstructed image tensor [B, 1, H, W].
	"""
	if lambda_tv <= 0.0:
		raise ValueError(f"lambda_tv must be positive; got {lambda_tv}.")
	x = y.clone()
	for iterate in tv_denoise_iterates(y, lambda_tv, k):
		x = iterate
	return x
