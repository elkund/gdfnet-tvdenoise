"""Benchmark suite mapping and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from gdfnet_tvdenoise.benchmarks.parity import evaluate_parity
from gdfnet_tvdenoise.benchmarks.plots import plot_parity, plot_sweeps
from gdfnet_tvdenoise.benchmarks.sweeps import evaluate_sweeps
from gdfnet_tvdenoise.config import BENCHMARK_SUITES, DEFAULT_DEVICE

#: Suite name -> (evaluator, plotter or None).
SUITES: dict[str, tuple[Callable[..., Any], Callable[..., Path] | None]] = {
	"parity": (evaluate_parity, plot_parity),
	"val_parity": (evaluate_parity, None),
	"sweeps": (evaluate_sweeps, plot_sweeps),
	"scale_parity": (evaluate_parity, plot_parity),
	"scale_sweeps": (evaluate_sweeps, plot_sweeps),
}
if tuple(SUITES) != BENCHMARK_SUITES:
	raise RuntimeError(
		f"SUITES keys {tuple(SUITES)} differ from BENCHMARK_SUITES {BENCHMARK_SUITES}."
	)


def run_benchmark(
	model_path: str | Path,
	suite_name: str = "all",
	device: torch.device = DEFAULT_DEVICE,
	plot: bool = True,
) -> dict[str, Path]:
	"""Run one or all benchmark suites and write reports with the model.

	Reports are written to `<model_path>/<suite>.pt` and reruns overwrite the same
	file. No sequential report suffixes are created.

	Args:
		model_path: Model directory containing `weights.pt`, or a weights file path.
		suite_name: One suite name or 'all'.
		device: PyTorch device.
		plot: Whether to render a diagnostic PNG after each evaluation.

	Returns:
		Mapping of suite name to written report path.
	"""
	chosen = BENCHMARK_SUITES if suite_name == "all" else (suite_name,)
	unknown = [name for name in chosen if name not in SUITES]
	if unknown:
		raise ValueError(f"Unknown benchmark suite(s) {unknown}. Available: {list(SUITES)}")

	outputs: dict[str, Path] = {}
	for name in chosen:
		evaluator, plotter = SUITES[name]
		_, report_path = evaluator(model_path, suite_name=name, device=device)
		outputs[name] = report_path
		if plot and plotter is not None:
			plot_benchmark(model_path, name)
	return outputs


def plot_benchmark(
	model_path: str | Path,
	suite_name: str = "all",
	output_path: str | Path | None = None,
) -> Path | list[Path]:
	"""Render benchmark figures from saved reports without re-evaluation.

	With `suite_name="all"`, every suite that has a plotter is rendered to its default
	location and `output_path` is ignored.

	Args:
		model_path: Model directory or weights file.
		suite_name: Suite name, or 'all' for every plottable suite.
		output_path: Optional explicit PNG output path (single suite only).

	Returns:
		Path to the rendered PNG, or a list of paths when `suite_name="all"`.
	"""
	if suite_name == "all":
		return [
			plot_benchmark(model_path, name)
			for name, (_, plot_fn) in SUITES.items()
			if plot_fn is not None
		]
	plotter = SUITES.get(suite_name, (None, None))[1]
	if plotter is None:
		available = [name for name, (_, plot_fn) in SUITES.items() if plot_fn is not None]
		raise ValueError(f"No plot is defined for suite '{suite_name}'. Available: {available}")
	return plotter(model_path, suite_name=suite_name, output_path=output_path)


__all__ = [
	"SUITES",
	"evaluate_parity",
	"evaluate_sweeps",
	"plot_benchmark",
	"run_benchmark",
]
