"""Command-line dispatch for the Python functions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from gdfnet_tvdenoise.benchmarks import SUITES, plot_benchmark, run_benchmark
from gdfnet_tvdenoise.calibration import calibrate_k, generate_flat_error_table
from gdfnet_tvdenoise.config import (
	BENCHMARK_SUITES,
	DEFAULT_DEVICE,
	DEFAULT_IMAGE_SIZE,
	DEFAULT_METROLOGY_NSR_TIERS,
	DEFAULT_PARITY_WORKLOAD,
	DEFAULT_SCALE_PARITY_WORKLOAD,
	DEFAULT_SCALE_SWEEPS_WORKLOAD,
	DEFAULT_SWEEPS_WORKLOAD,
	DEFAULT_TRAIN_BATCH_SIZE,
	DEFAULT_TRAIN_EPOCHS,
	DEFAULT_TRAIN_LR,
	DEFAULT_TRAIN_SEED,
	DEFAULT_TRAINING_WORKLOAD,
	DEFAULT_VAL_PARITY_WORKLOAD,
)
from gdfnet_tvdenoise.data import (
	fetch_data,
	generate_stochastic,
	generate_sweeps,
	prepare_archives,
)
from gdfnet_tvdenoise.inference import compute_mdp, compute_sure, predict_gdf
from gdfnet_tvdenoise.train import train


def _by_name(*workloads: dict[str, object]) -> dict[str, dict[str, object]]:
	"""Map workload names to workloads; raise on duplicate names."""
	mapping = {str(workload["name"]): workload for workload in workloads}
	if len(mapping) != len(workloads):
		names = [str(workload["name"]) for workload in workloads]
		raise ValueError(f"Duplicate workload names: {names}")
	return mapping


#: CLI names for `generate --stochastic` and `generate --sweeps`.
_STOCHASTIC = _by_name(
	DEFAULT_TRAINING_WORKLOAD,
	DEFAULT_PARITY_WORKLOAD,
	DEFAULT_SCALE_PARITY_WORKLOAD,
	DEFAULT_VAL_PARITY_WORKLOAD,
)
_SWEEPS = _by_name(DEFAULT_SWEEPS_WORKLOAD, DEFAULT_SCALE_SWEEPS_WORKLOAD)


def _load_image(path: str | None) -> torch.Tensor:
	"""Load an image from `.npy`/`.pt`, or return a deterministic random 64x64 image."""
	if path is None:
		generator = torch.Generator().manual_seed(0)
		return torch.rand((1, 1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), generator=generator)
	source = Path(path)
	if not source.exists():
		raise FileNotFoundError(f"Image file not found: {source}")
	if source.suffix == ".npy":
		array = np.load(source)
		return torch.as_tensor(array, dtype=torch.float32)
	if source.suffix in (".pt", ".pth"):
		payload = torch.load(source, map_location="cpu")
		return torch.as_tensor(payload, dtype=torch.float32)
	raise ValueError(f"Unsupported image format: {source.suffix}. Use .npy or .pt.")


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
	parser = subparsers.add_parser("generate", help="Fetch sources and generate data artifacts.")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--source", choices=("medmnist", "lits", "all"))
	group.add_argument("--prepare-lits", action="store_true")
	group.add_argument("--stochastic", choices=(*_STOCHASTIC, "all"))
	group.add_argument("--sweeps", choices=(*_SWEEPS, "all"))
	parser.add_argument("--samples", type=int, default=None)
	parser.add_argument("--sweep-points", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--device", type=torch.device, default=DEFAULT_DEVICE)
	parser.add_argument("--archive-dir", type=Path, default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument(
		"--fresh",
		action="store_true",
		help="Discard an existing generation checkpoint and start over.",
	)


def _add_calibrate_parser(subparsers: argparse._SubParsersAction) -> None:
	parser = subparsers.add_parser("calibrate", help="Calibrate the solver budget K.")
	parser.add_argument("--which", choices=("table", "k"), default="k")
	parser.add_argument("--nsr-tiers", type=float, nargs="+", default=None)
	parser.add_argument("--device", type=torch.device, default=DEFAULT_DEVICE)
	parser.add_argument("--archive-dir", type=Path, default=None)
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument(
		"--fresh",
		action="store_true",
		help="Discard an existing K calibration checkpoint and start over.",
	)


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
	parser = subparsers.add_parser("train", help="Train a SpatialGDFNet surrogate.")
	parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
	parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
	parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
	parser.add_argument("--lr", type=float, default=DEFAULT_TRAIN_LR)
	parser.add_argument("--device", type=torch.device, default=DEFAULT_DEVICE)
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument("--data", type=Path, default=None)
	parser.add_argument("--max-samples", type=int, default=None)


def _add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
	parser = subparsers.add_parser("benchmark", help="Evaluate benchmark suites.")
	parser.add_argument(
		"--model-path",
		type=Path,
		required=True,
		help="Model directory (containing weights.pt) or a weights file path.",
	)
	parser.add_argument("--suite", choices=("all", *BENCHMARK_SUITES), default="all")
	parser.add_argument("--device", type=torch.device, default=DEFAULT_DEVICE)
	parser.add_argument("--no-plot", action="store_true", help="Skip rendering plots.")


def _add_plot_parser(subparsers: argparse._SubParsersAction) -> None:
	parser = subparsers.add_parser("plot", help="Render a saved benchmark report.")
	parser.add_argument(
		"--model-path",
		type=Path,
		required=True,
		help="Model directory (containing weights.pt) or a weights file path.",
	)
	parser.add_argument(
		"--suite",
		choices=("all", *[name for name, (_, plotter) in SUITES.items() if plotter is not None]),
		default="all",
		help="Suite to plot, or 'all' (default).",
	)
	parser.add_argument("--output", type=Path, default=None)


def _add_infer_parser(subparsers: argparse._SubParsersAction) -> None:
	parser = subparsers.add_parser("infer", help="Predict GDF for one image and lambda.")
	parser.add_argument("--image", type=str, default=None)
	parser.add_argument("--lambda-tv", type=float, required=True)
	parser.add_argument("--model", type=Path, default=None)
	parser.add_argument("--sigma", type=float, default=None)
	parser.add_argument("--mse", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line parser."""
	parser = argparse.ArgumentParser(prog="python -m gdfnet_tvdenoise")
	subparsers = parser.add_subparsers(dest="command", required=True)
	_add_infer_parser(subparsers)
	_add_generate_parser(subparsers)
	_add_calibrate_parser(subparsers)
	_add_train_parser(subparsers)
	_add_benchmark_parser(subparsers)
	_add_plot_parser(subparsers)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Run the command-line interface.

	Args:
		argv: Optional argument vector. Defaults to `sys.argv[1:]`.

	Returns:
		Process exit code.
	"""
	args = build_parser().parse_args(argv)

	if args.command == "infer":
		y = _load_image(args.image)
		r = predict_gdf(y, args.lambda_tv, args.model)
		print(f"gdf_n={float(r.item()):.8f} shape={tuple(r.shape)}")
		if args.sigma is not None and args.mse is not None:
			print(f"sure={float(compute_sure(args.mse, float(r.item()), args.sigma)):.8f}")
			print(f"mdp={float(compute_mdp(args.mse, float(r.item()), args.sigma)):.8f}")
		return 0

	if args.command == "generate":
		if args.source is not None:
			if args.dry_run:
				print(f"[dry-run] fetch source={args.source} archive_dir={args.archive_dir}")
				return 0
			paths = fetch_data(args.source, archive_dir=args.archive_dir)
			for path in paths:
				print(path)
			return 0
		if args.prepare_lits:
			outputs = prepare_archives("lits", archive_dir=args.archive_dir, dry_run=args.dry_run)
			for path in outputs.values():
				print(path)
			return 0
		common = {
			"device": args.device,
			"archive_dir": args.archive_dir,
			"output_dir": args.output_dir,
			"dry_run": args.dry_run,
			"resume": not args.fresh,
		}
		if args.stochastic is not None:
			names = _STOCHASTIC if args.stochastic == "all" else (args.stochastic,)
			for name in names:
				print(
					generate_stochastic(
						_STOCHASTIC[name], samples=args.samples, seed=args.seed, **common
					)
				)
			return 0
		names = _SWEEPS if args.sweeps == "all" else (args.sweeps,)
		for name in names:
			print(generate_sweeps(_SWEEPS[name], sweep_points=args.sweep_points, **common))
		return 0

	if args.command == "calibrate":
		if args.which == "table":
			generate_flat_error_table(output=args.output)
			return 0
		calibrate_k(
			nsr_tiers=args.nsr_tiers or DEFAULT_METROLOGY_NSR_TIERS,
			device=args.device,
			output=args.output,
			archive_dir=args.archive_dir,
			fresh=args.fresh,
		)
		return 0

	if args.command == "train":
		train(
			seed=args.seed,
			epochs=args.epochs,
			batch_size=args.batch_size,
			lr=args.lr,
			device=args.device,
			output=args.output,
			data_path=args.data,
			max_samples=args.max_samples,
		)
		return 0

	if args.command == "benchmark":
		outputs = run_benchmark(
			args.model_path, args.suite, device=args.device, plot=not args.no_plot
		)
		for name, path in outputs.items():
			print(f"{name}: {path}")
		return 0

	if args.command == "plot":
		results = plot_benchmark(args.model_path, args.suite, args.output)
		for path in results if isinstance(results, list) else [results]:
			print(path)
		return 0

	raise ValueError(f"Unknown command: {args.command}")


__all__ = ["build_parser", "main"]
