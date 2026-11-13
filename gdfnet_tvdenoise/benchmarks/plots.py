"""Plotting routines for saved benchmark reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

from gdfnet_tvdenoise.config import (
	DEFAULT_DEVICE,
	DEFAULT_IMAGE_SIZE,
	DEFAULT_MODEL_PATH,
	OVERVIEW_FIGURE_PATH,
	OVERVIEW_PATH_NODE_OFFSET,
	OVERVIEW_REGIME_INDEX,
	OVERVIEW_SUITE,
	REPO_ROOT,
)
from gdfnet_tvdenoise.helpers import load_data_file, load_pt, resolve_model_paths
from gdfnet_tvdenoise.solver import tv_denoise

SCALE_STYLES: dict[int, dict[str, Any]] = {
	4096: {"marker": "o", "color": "#1f77b4", "label": "64×64"},
	16384: {"marker": "s", "color": "#ff7f0e", "label": "128×128"},
	50176: {"marker": "D", "color": "#2ca02c", "label": "224×224"},
}


class _VerticalLineLegendHandler(HandlerBase):
	"""Legend handler that renders a vertical line sample."""

	def create_artists(
		self,
		legend: Any,
		orig_handle: Line2D,
		xdescent: float,
		ydescent: float,
		width: float,
		height: float,
		fontsize: float,
		trans: Any,
	) -> list[Line2D]:
		"""Create the legend artist."""
		line_x = xdescent + width / 2
		return [
			Line2D(
				[line_x, line_x],
				[ydescent, ydescent + height],
				color=orig_handle.get_color(),
				linestyle=orig_handle.get_linestyle(),
				linewidth=orig_handle.get_linewidth(),
				transform=trans,
			)
		]


def _load_report(model_path: str | Path, suite_name: str) -> tuple[Path, dict[str, Any]]:
	"""Return the model directory and the saved report payload."""
	_, model_dir = resolve_model_paths(model_path)
	path = model_dir / f"{suite_name}.pt"
	if not path.exists():
		raise FileNotFoundError(f"Benchmark report not found: {path}")
	return model_dir, load_pt(path)


def _linear_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
	"""Return the least-squares slope and intercept of a degree-1 fit of y on x."""
	x_mean = x.mean()
	y_mean = y.mean()
	denom = ((x - x_mean) ** 2).sum()
	slope = ((x - x_mean) * (y - y_mean)).sum() / denom
	return float(slope), float(y_mean - slope * x_mean)


def _resolve_output_path(
	model_dir: Path,
	suite_name: str,
	output_path: str | Path | None,
) -> Path:
	"""Resolve the destination path for a rendered benchmark figure."""
	dest = Path(output_path) if output_path is not None else model_dir / f"{suite_name}.png"
	dest.parent.mkdir(parents=True, exist_ok=True)
	return dest


def plot_parity(
	model_path: str | Path,
	suite_name: str = "parity",
	output_path: str | Path | None = None,
) -> Path:
	"""Render a two-panel parity scatter and noise-correlation diagnostic.

	Args:
		model_path: Model directory or weights file.
		suite_name: Report suite name.
		output_path: Optional explicit PNG output path.

	Returns:
		Path to the rendered PNG.
	"""
	model_dir, payload = _load_report(model_path, suite_name)
	metrics = payload["summary_metrics"]
	t_arr = payload["true_r"]
	p_arr = payload["pred_r"]
	sig_arr = payload["sigmas"]
	npix_arr = payload["n_pixels"]

	residuals = t_arr - p_arr
	r2 = metrics["r2"]
	mae_pct = metrics["mae_pct"]
	rmse = metrics["rmse"]

	fig, (ax_parity, ax_noise) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
	unique_scales = sorted({int(value) for value in npix_arr})
	if len(unique_scales) > 1:
		for npix in unique_scales:
			mask = npix_arr == npix
			cfg = SCALE_STYLES.get(
				npix, {"marker": "o", "color": "#9467bd", "label": f"{math.isqrt(npix)}px"}
			)
			ax_parity.scatter(
				t_arr[mask],
				p_arr[mask],
				marker=cfg["marker"],
				color=cfg["color"],
				label=cfg["label"],
				alpha=0.75,
				s=35,
			)
	else:
		scatter = ax_parity.scatter(t_arr, p_arr, c=sig_arr, cmap="viridis", alpha=0.75, s=35)
		colorbar = fig.colorbar(scatter, ax=ax_parity, pad=0.02)
		colorbar.set_label("Noise Level σ", fontsize=9)

	ax_parity.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ideal 1:1 Parity")
	ax_parity.set_xlabel("Monte Carlo Ground Truth r = GDF / N", fontsize=10)
	ax_parity.set_ylabel("Neural-GDF Predicted r̂", fontsize=10)
	ax_parity.set_title("Parity Scatter (r in [0, 1])", fontsize=11, fontweight="bold")
	ax_parity.grid(True, linestyle=":", alpha=0.6)
	ax_parity.legend(loc="upper left", framealpha=0.9)
	ax_parity.text(
		0.95,
		0.05,
		f"R² = {r2:.4f}\nMAE = {mae_pct:.2f}%\nRMSE = {rmse * 100.0:.2f}%",
		transform=ax_parity.transAxes,
		verticalalignment="bottom",
		horizontalalignment="right",
		bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"),
		fontsize=9,
	)

	ax_noise.scatter(sig_arr, residuals, color="#1f77b4", alpha=0.70, s=35)
	ax_noise.axhline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.7)
	if len(sig_arr) > 1 and float(sig_arr.std(correction=0)) > 1e-6:
		slope, intercept = _linear_fit(sig_arr, residuals)
		x_t = torch.linspace(float(sig_arr.min()), float(sig_arr.max()), 100)
		ax_noise.plot(
			x_t,
			slope * x_t + intercept,
			color="#d62728",
			lw=1.5,
			label=f"Linear Fit (slope={slope:.2f})",
		)
		ax_noise.legend(loc="upper right", framealpha=0.9)
	ax_noise.set_xlabel("Noise Standard Deviation σ", fontsize=10)
	ax_noise.set_ylabel("Residual (True r - Pred r)", fontsize=10)
	ax_noise.set_title("Residual Error vs. Noise Level", fontsize=11, fontweight="bold")
	ax_noise.grid(True, linestyle=":", alpha=0.6)

	fig.suptitle(
		f"{model_dir.name} — {suite_name.replace('_', ' ').title()} Evaluation",
		fontsize=13,
		fontweight="bold",
		y=0.98,
	)
	plt.tight_layout()
	dest = _resolve_output_path(model_dir, suite_name, output_path)
	fig.savefig(dest, dpi=150)
	plt.close(fig)
	return dest


def plot_sweeps(
	model_path: str | Path,
	suite_name: str = "sweeps",
	output_path: str | Path | None = None,
) -> Path:
	"""Render multi-panel MC GDF curves against surrogate predictions.

	Args:
		model_path: Model directory or weights file.
		suite_name: Report suite name ('sweeps' or 'scale_sweeps').
		output_path: Optional explicit PNG output path.

	Returns:
		Path to the rendered PNG.
	"""
	model_dir, payload = _load_report(model_path, suite_name)
	metrics = payload["summary_metrics"]
	regimes = payload["regimes"]
	if not regimes:
		raise ValueError("No regime data found to plot.")

	n_regimes = len(regimes)
	n_cols = min(4, n_regimes)
	n_rows = math.ceil(n_regimes / n_cols)
	fig, axes = plt.subplots(
		n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), dpi=150, squeeze=False
	)
	axes_flat = axes.flat

	for idx, regime in enumerate(regimes):
		ax = axes_flat[idx]
		lams = regime["lambdas"]
		true_r = regime["mc_gdf_norm"]
		pred_r = regime["surrogate_gdf_norm"]
		title = regime["title"]
		sigma_val = float(regime["sigma"])
		sigma_mad_val = float(regime["sigma_mad"])
		r2_val = float(regime["r2"])
		mae_val = float(regime["mae_pct"])
		x = lams / sigma_mad_val

		e_tot = regime["mc_gdf_norm_err"]
		(surrogate_gdf,) = ax.plot(x, pred_r, "--", color="#1f77b4", lw=0.9, zorder=2)
		mc_gdf = ax.errorbar(
			x,
			true_r,
			yerr=e_tot,
			fmt="o",
			color="#1f77b4",
			ms=2.0,
			capsize=4,
			elinewidth=1.2,
			capthick=1.2,
			alpha=0.9,
			zorder=5,
		)
		ax.set_xscale("log")
		ax.set_ylim(-0.02, 1.05)
		ax.set_xlabel("Normalized Regularization λ / σ_MAD", fontsize=8)
		ax.set_ylabel("Capacity r = GDF / N", fontsize=8)
		ax.grid(True, linestyle=":", alpha=0.5)

		short_title = title.replace("Panel ", "P").replace("Denoise — ", "")
		if "σ" not in short_title and "sigma" not in short_title.lower():
			short_title = f"{short_title} (σ={sigma_val:.2f})"
		ax.set_title(short_title, fontsize=9, fontweight="bold")

		ax2 = ax.twinx()
		sure_s = regime["surrogate_sure_mad"]
		sure_m = regime["mc_sure_mad"]
		mc_sure = ax2.errorbar(
			x,
			sure_m,
			yerr=regime["mc_delta_sure_mad"],
			fmt="o",
			color="#2ca02c",
			ms=2.0,
			capsize=4,
			elinewidth=1.2,
			capthick=1.2,
			alpha=0.9,
			zorder=5,
		)
		(surrogate_sure,) = ax2.plot(
			x,
			sure_s,
			"--",
			color="#2ca02c",
			lw=0.9,
			alpha=0.85,
			zorder=2,
		)
		ax2.set_ylabel("SURE Risk (σ_MAD)", fontsize=7, color="#2ca02c")
		ax2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=7)

		mc_optimal = ax.axvline(
			float(regime["selected_lambda"]["mc_sure_mad"]) / sigma_mad_val,
			color="#2ca02c",
			linestyle="-",
			lw=1.0,
			zorder=4,
		)
		surrogate_optimal = ax.axvline(
			float(regime["selected_lambda"]["surrogate_sure_mad"]) / sigma_mad_val,
			color="#2ca02c",
			linestyle="--",
			lw=0.9,
			zorder=4,
		)
		ax.text(
			0.95,
			0.95,
			f"R² = {r2_val:.3f}\nMAE = {mae_val:.2f}%",
			transform=ax.transAxes,
			verticalalignment="top",
			horizontalalignment="right",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"),
			fontsize=6.8,
		)
		ax.legend(
			[surrogate_gdf, surrogate_sure, mc_gdf, mc_sure, mc_optimal, surrogate_optimal],
			[
				"Surrogate GDF",
				"Surrogate SURE (σ_MAD)",
				"MC GDF",
				"MC SURE (σ_MAD)",
				"MC optimal λ (σ_MAD)",
				"Surrogate optimal\nλ (σ_MAD)",
			],
			loc="center left",
			fontsize=6.2,
			framealpha=0.85,
			handler_map={
				mc_optimal: _VerticalLineLegendHandler(),
				surrogate_optimal: _VerticalLineLegendHandler(),
			},
		)

	for remaining in range(n_regimes, n_rows * n_cols):
		axes_flat[remaining].set_visible(False)

	mean_r2 = metrics["mean_r2"]
	mean_mae = metrics["mean_mae_pct"]
	header = (
		f"{model_dir.name} — Multi-Scale Sweeps"
		if "scale" in suite_name
		else (f"{model_dir.name} — Clinical Sweeps")
	)
	fig.suptitle(
		f"{header} | Mean R²: {mean_r2:.4f} | Mean MAE: {mean_mae:.2f}%",
		fontsize=12,
		fontweight="bold",
		y=0.99,
	)
	plt.tight_layout()
	dest = _resolve_output_path(model_dir, suite_name, output_path)
	fig.savefig(dest, dpi=150)
	plt.close(fig)
	return dest


def plot_overview(
	model_path: str | Path = DEFAULT_MODEL_PATH,
	output_path: str | Path = OVERVIEW_FIGURE_PATH,
	*,
	suite_name: str = OVERVIEW_SUITE,
	regime_index: int = OVERVIEW_REGIME_INDEX,
	node_offset: int = OVERVIEW_PATH_NODE_OFFSET,
	device: torch.device = DEFAULT_DEVICE,
) -> Path:
	"""Render the README overview: a TV regularization path next to its sweep panel.

	The left part shows the clean image, the observation, and TV reconstructions at the
	SURE selection λ* of the Monte Carlo reference and `node_offset` grid nodes below and
	above it, each labeled with its Monte Carlo GDF/N. The right part shows the Monte Carlo
	and surrogate GDF and SURE curves of the same regime with both selections.

	Args:
		model_path: Model directory or weights file with a saved sweeps report.
		output_path: PNG output path.
		suite_name: Sweeps report suite name ('sweeps' or 'scale_sweeps').
		regime_index: Regime shown, an index into the report regimes.
		node_offset: Grid-node offset of the under- and over-regularized examples.
		device: Execution device of the TV solves.

	Returns:
		Path to the rendered PNG.

	Raises:
		ValueError: If the report does not match its dataset or the offset nodes fall
			outside the λ grid.
	"""
	_, payload = _load_report(model_path, suite_name)
	sources = payload["source_datasets"]
	if len(sources) != 1:
		raise ValueError(f"Expected one source dataset for {suite_name}; got {len(sources)}.")
	data = load_data_file(REPO_ROOT / sources[0]["test_data"])
	regime = payload["regimes"][regime_index]
	lambdas = regime["lambdas"]
	if not torch.equal(data["lambdas"][regime_index], lambdas):
		raise ValueError(f"Report regime {regime_index} does not match its dataset λ grid.")
	i_star = int(regime["selection_excess"]["SURE"]["reference_index"])
	nodes = (i_star - node_offset, i_star, i_star + node_offset)
	if nodes[0] < 0 or nodes[-1] >= lambdas.numel():
		raise ValueError(f"Nodes {nodes} fall outside the {lambdas.numel()}-node λ grid.")

	x_clean = data["x_clean"][regime_index]
	y = data["observations_y"][regime_index]
	size = x_clean.shape[-1]
	x_clean = x_clean.reshape(size, size)
	y_dev = y.reshape(1, 1, size, size).to(device)
	k = int(data["k_solvers"][regime_index])
	reconstructions = [
		tv_denoise(y_dev, float(lambdas[i]), k).reshape(size, size).cpu() for i in nodes
	]
	ratio = float(lambdas[nodes[2]] / lambdas[nodes[1]])
	sigma_mad = float(regime["sigma_mad"])
	nsr = float(regime["sigma"]) / float(x_clean.std(correction=0))
	modality = regime["title"].split("— ")[1].split(" (")[0]

	path_colors = ("#e41a1c", "#ff7f00", "#984ea3")
	tiles = [
		(x_clean, "Clean image", None),
		(y.reshape(size, size), f"Observation\nσ / std(x) = {nsr:.2f}", None),
		(reconstructions[0], f"λ = λ* / {ratio:.1f}", path_colors[0]),
		(reconstructions[1], "λ = λ* (SURE)", path_colors[1]),
		(reconstructions[2], f"λ = {ratio:.1f} λ*", path_colors[2]),
	]
	fig = plt.figure(figsize=(15.0, 4.0), dpi=150)
	grid = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 0.3, 1.7], wspace=0.05)
	vmin, vmax = float(x_clean.min()), float(x_clean.max())
	for col, (image, label, color) in enumerate(tiles):
		ax = fig.add_subplot(grid[0, col])
		ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(label, fontsize=9)
		if color is not None:
			node = nodes[col - 2]
			for spine in ax.spines.values():
				spine.set_color(color)
				spine.set_linewidth(4.0)
			ax.set_xlabel(f"GDF / N = {float(regime['mc_gdf_norm'][node]):.2f}", fontsize=9)

	ax = fig.add_subplot(grid[0, 6])
	x = lambdas / sigma_mad
	gdf_color, sure_color = "#1f77b4", "#2ca02c"
	ax.errorbar(
		x,
		regime["mc_gdf_norm"],
		yerr=regime["mc_gdf_norm_err"],
		fmt="o",
		color=gdf_color,
		ms=3.0,
		capsize=2,
		zorder=5,
	)
	ax.plot(x, regime["surrogate_gdf_norm"], "-", color=gdf_color, lw=1.2)
	for node, color in zip(nodes, path_colors, strict=True):
		ax.plot(
			float(x[node]),
			float(regime["mc_gdf_norm"][node]),
			"s",
			ms=10,
			mfc="none",
			mec=color,
			mew=2.5,
			zorder=6,
		)
	ax.set_xscale("log")
	# Headroom above the curves holds the legend.
	ax.set_ylim(-0.02, 1.3)
	ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
	ax.set_xlabel(r"$\lambda\,/\,\hat{\sigma}_{\mathrm{MAD}}$", fontsize=10)
	ax.set_ylabel("GDF / N", fontsize=9, color=gdf_color)
	ax.tick_params(axis="y", labelcolor=gdf_color, labelsize=8)
	ax.tick_params(axis="x", labelsize=8)
	ax.grid(True, linestyle=":", alpha=0.5)

	ax_sure = ax.twinx()
	sure_scale = sigma_mad**2
	ax_sure.errorbar(
		x,
		regime["mc_sure_mad"] / sure_scale,
		yerr=regime["mc_delta_sure_mad"] / sure_scale,
		fmt="o",
		color=sure_color,
		ms=3.0,
		capsize=2,
		zorder=5,
	)
	ax_sure.plot(
		x,
		regime["surrogate_sure_mad"] / sure_scale,
		"-",
		color=sure_color,
		lw=1.2,
	)
	sure_top = float(((regime["mc_sure_mad"] + regime["mc_delta_sure_mad"]) / sure_scale).max())
	ax_sure.set_ylim(0.0, 1.3 * sure_top)
	ax_sure.set_yticks([tick / 5 for tick in range(int(5 * sure_top) + 1)])
	ax_sure.set_ylabel(r"SURE$\,/\,\hat{\sigma}_{\mathrm{MAD}}^2$", fontsize=10, color=sure_color)
	ax_sure.tick_params(axis="y", labelcolor=sure_color, labelsize=8)
	selected = regime["selected_lambda"]
	mc_line = ax.axvline(float(selected["mc_sure_mad"]) / sigma_mad, color="black", lw=1.0)
	surr_line = ax.axvline(
		float(selected["surrogate_sure_mad"]) / sigma_mad,
		color="black",
		lw=1.0,
		linestyle="--",
	)
	mc_proxy = Line2D([], [], color="dimgray", marker="o", ms=3.0, linestyle="none")
	surr_proxy = Line2D([], [], color="dimgray", lw=1.2)
	ax.legend(
		handles=[mc_proxy, surr_proxy, mc_line, surr_line],
		labels=["Monte Carlo (±1 SE)", "Surrogate", "λ*, Monte Carlo", "λ*, surrogate"],
		loc="upper center",
		bbox_to_anchor=(0.5, 0.995),
		ncol=2,
		fontsize=7.5,
		framealpha=0.95,
		handler_map={
			mc_line: _VerticalLineLegendHandler(),
			surr_line: _VerticalLineLegendHandler(),
		},
	)
	ax.set_title(
		f"{modality}, {size} px (surrogate trained at {DEFAULT_IMAGE_SIZE} px)", fontsize=9
	)

	dest = Path(output_path)
	dest.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(dest, dpi=150, bbox_inches="tight")
	plt.close(fig)
	return dest


__all__ = ["plot_overview", "plot_parity", "plot_sweeps"]
