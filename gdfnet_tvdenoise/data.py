"""Data fetching, archive preparation, and Monte Carlo data generation."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import math
import statistics
import struct
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gdfnet_tvdenoise.config import (
	ARCHIVE_DIR,
	CANONICAL_MEDMNIST_DATASETS,
	CANONICAL_MEDMNIST_SIZES,
	CHUNK_SIZE_BYTES,
	DEFAULT_CP_SIGMA,
	DEFAULT_CP_TAU,
	DEFAULT_CP_THETA,
	DEFAULT_DEVICE,
	DEFAULT_IMAGE_SIZE,
	DEFAULT_M_BY_SIZE,
	DEFAULT_MAX_RETRIES,
	DEFAULT_METROLOGY_DELTA_K,
	DEFAULT_METROLOGY_NSR_TIERS,
	DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MAX,
	DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MIN,
	DEFAULT_TAU,
	GENERATION_CHECKPOINT_INTERVAL_SECONDS,
	GENERATION_CHECKPOINT_SUFFIX,
	LITS_AIR_HU_MAX,
	LITS_AIR_SHIFT_PX,
	LITS_AIR_SHIFT_STEP_PX,
	LITS_ARCHIVE_MANIFEST_PATH,
	LITS_CROP_MARGIN,
	LITS_HF_REPO,
	LITS_HF_REVISION,
	LITS_HU_MAX,
	LITS_HU_MIN,
	LITS_MAX_AIR_FRACTION,
	LITS_RAW_DIR,
	LITS_SPLIT_PATH,
	LITS_SPLITS,
	MEDMNIST_PLUS_DATASETS,
	METROLOGY_DOMINANCE_RATIO,
	SOLVER_K_TIERS,
)
from gdfnet_tvdenoise.features import image_std
from gdfnet_tvdenoise.helpers import (
	atomic_write,
	compute_mc_gdf,
	display_path,
	estimate_noise_wavelet_mad,
	load_pt,
	release_device_cache,
	save_pt,
)
from gdfnet_tvdenoise.solver import solver_k

# ===========================================================================
# Physics operator
# ===========================================================================


def create_noisy_observation(
	x_clean: torch.Tensor,
	sigma: float,
	seed: int | None = None,
) -> torch.Tensor:
	"""Add Gaussian noise at `sigma` to a clean image.

	With `seed` set, the noise is drawn from a dedicated generator and is reproducible.
	With `seed=None`, the draw uses the global RNG.

	Args:
		x_clean: Clean image tensor.
		sigma: Gaussian noise standard deviation.
		seed: Optional seed for the noise draw.

	Returns:
		Noisy observation tensor of the same shape and dtype as `x_clean`.
	"""
	gen = torch.Generator(device=x_clean.device).manual_seed(seed) if seed is not None else None
	noise = torch.randn(x_clean.shape, generator=gen, device=x_clean.device, dtype=x_clean.dtype)
	return x_clean + sigma * noise


# ===========================================================================
# MedMNIST archive access
# ===========================================================================


def _archive_path(
	dataset_name: str,
	size: int,
	archive_dir: str | Path | None = None,
) -> Path:
	base = Path(archive_dir) if archive_dir is not None else ARCHIVE_DIR
	return base / f"{dataset_name.lower()}_{size}.npz"


def load_medmnist_images(
	dataset_name: str,
	split: str = "train",
	indices: int | Sequence[int] | torch.Tensor | None = None,
	count: int | None = None,
	rng: torch.Generator | None = None,
	size: int = DEFAULT_IMAGE_SIZE,
	archive_dir: str | Path | None = None,
) -> torch.Tensor:
	"""Load images normalized to [0, 1] as a float32 tensor [B, 1, H, W].

	Args:
		dataset_name: Modality name.
		split: Dataset partition ('train', 'val', or 'test').
		indices: Optional specific indices to load.
		count: Optional number of images to sample without replacement.
		rng: Optional torch generator used with `count`. None uses the global torch RNG.
		size: Spatial image resolution.
		archive_dir: Optional explicit archive directory.

	Returns:
		4D float32 tensor normalized to [0, 1].
	"""
	path = _archive_path(dataset_name, size, archive_dir)
	with np.load(path, mmap_mode="r") as archive:
		data = archive[f"{split}_images"]
		if isinstance(indices, int):
			idx = [indices]
		elif isinstance(indices, torch.Tensor):
			idx = indices.tolist()
		elif indices is not None:
			idx = list(indices)
		elif count is not None:
			idx = torch.randperm(len(data), generator=rng)[:count].tolist()
		else:
			idx = slice(None)
		raw = torch.tensor(data[idx]).to(torch.float32) / 255.0

	if raw.shape[-1] == 3:
		raw = 0.2989 * raw[..., 0] + 0.5870 * raw[..., 1] + 0.1140 * raw[..., 2]
	return raw.unsqueeze(1)


def get_medmnist_split_size(
	dataset_name: str,
	split: str = "train",
	size: int = DEFAULT_IMAGE_SIZE,
	archive_dir: str | Path | None = None,
) -> int:
	"""Return the number of images in a canonical archive partition."""
	path = _archive_path(dataset_name, size, archive_dir)
	with np.load(path, mmap_mode="r") as archive:
		return int(len(archive[f"{split}_images"]))


# ===========================================================================
# Source fetching
# ===========================================================================


def compute_file_md5(path: Path) -> str:
	"""Compute the MD5 checksum of a local file in 1 MB chunks."""
	hasher = hashlib.md5()
	with open(path, "rb") as handle:
		while chunk := handle.read(CHUNK_SIZE_BYTES):
			hasher.update(chunk)
	return hasher.hexdigest()


def _download_with_md5(
	url: str,
	dest: Path,
	expected_md5: str,
	max_retries: int = DEFAULT_MAX_RETRIES,
) -> Path:
	"""Download a URL to `dest`, verify MD5, and replace atomically."""
	dest.parent.mkdir(parents=True, exist_ok=True)
	if dest.exists() and compute_file_md5(dest).lower() == expected_md5.lower():
		return dest
	part = dest.with_name(dest.name + ".part")
	last_error: Exception | None = None
	for attempt in range(1, max_retries + 1):
		try:
			request = urllib.request.Request(url, headers={"User-Agent": "gdfnet-tvdenoise"})
			with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as out:
				while chunk := response.read(CHUNK_SIZE_BYTES):
					out.write(chunk)
			actual = compute_file_md5(part)
			if actual.lower() != expected_md5.lower():
				raise ValueError(f"MD5 mismatch for {url}: {actual} != {expected_md5}")
			part.replace(dest)
			return dest
		except Exception as error:  # noqa: BLE001 - retry boundary
			last_error = error
			part.unlink(missing_ok=True)
			if attempt < max_retries:
				time.sleep(min(30.0, 2.0**attempt))
	raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {last_error}")


def fetch_data(
	source: str = "medmnist",
	*,
	sizes: Sequence[int] | None = None,
	datasets: Sequence[str] | None = None,
	archive_dir: str | Path | None = None,
	lits_raw_dir: str | Path | None = None,
	revision: str | None = LITS_HF_REVISION,
	max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[Path]:
	"""Fetch source data for the pipeline.

	`fetch_data("medmnist")` downloads the four canonical modalities at 64/128/224 px and
	writes `data/archives/{name}_{size}.npz`. `fetch_data("lits")` snapshots the MSD Task03
	Liver mirror into `data/lits_raw/`. `fetch_data("all")` runs both.

	Args:
		source: One of 'medmnist', 'lits', or 'all'.
		sizes: Optional resolution subset.
		datasets: Optional modality subset.
		archive_dir: Optional explicit archive directory.
		lits_raw_dir: Optional explicit LiTS source directory.
		revision: Optional Hugging Face revision for the LiTS snapshot.
		max_retries: Maximum download attempts per archive.

	Returns:
		List of fetched paths.
	"""
	chosen_sizes = tuple(sizes) if sizes is not None else CANONICAL_MEDMNIST_SIZES
	chosen_datasets = tuple(datasets) if datasets is not None else MEDMNIST_PLUS_DATASETS
	base = Path(archive_dir) if archive_dir is not None else ARCHIVE_DIR
	outputs: list[Path] = []

	if source in ("medmnist", "all"):
		from medmnist.info import INFO

		base.mkdir(parents=True, exist_ok=True)
		for size in chosen_sizes:
			for dataset in chosen_datasets:
				if dataset == "lits":
					raise ValueError(
						"LiTS is not a MedMNIST+ archive. Use fetch_data('lits') and "
						"prepare_archives('lits')."
					)
				info = INFO.get(dataset)
				if info is None:
					raise ValueError(f"Unknown MedMNIST modality: {dataset}")
				url = info.get("url" if size == 28 else f"url_{size}")
				expected_md5 = info.get("MD5" if size == 28 else f"MD5_{size}")
				if not url or not expected_md5:
					raise ValueError(f"No MedMNIST URL/MD5 for {dataset} at {size} px.")
				dest = base / f"{dataset}_{size}.npz"
				outputs.append(_download_with_md5(url, dest, expected_md5, max_retries))

	if source in ("lits", "all"):
		from huggingface_hub import snapshot_download

		target = Path(lits_raw_dir) if lits_raw_dir is not None else LITS_RAW_DIR
		target.mkdir(parents=True, exist_ok=True)
		snapshot_download(
			repo_id=LITS_HF_REPO,
			repo_type="dataset",
			revision=revision,
			local_dir=target,
			allow_patterns=["imagesTr/*", "labelsTr/*", "README.md"],
		)
		outputs.append(target)

	if source not in ("medmnist", "lits", "all"):
		raise ValueError(f"Unknown data source: {source}")
	return outputs


# ===========================================================================
# LiTS archive preparation
# ===========================================================================


@dataclass(frozen=True)
class SliceRecord:
	"""One eligible axial slice and its uint8 image at every qualifying size."""

	patient: int
	z: int
	images: dict[int, np.ndarray]


def read_nifti_gz(path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
	"""Read a 3D array from a NIfTI `.nii.gz` file using pure Python."""
	with gzip.open(path, "rb") as handle:
		hdr = handle.read(348)
		dim = struct.unpack("<8h", hdr[40:56])
		_, bitpix = struct.unpack("<hh", hdr[70:74])
		(vox_offset,) = struct.unpack("<f", hdr[108:112])
		nx, ny, nz = dim[1], dim[2], dim[3]
		handle.seek(int(vox_offset))
		if bitpix == 8:
			dtype = np.uint8
		elif bitpix == 16:
			dtype = np.int16
		elif bitpix == 32:
			dtype = np.float32
		else:
			dtype = np.int32
		raw = handle.read(nx * ny * nz * (bitpix // 8))
		arr = np.frombuffer(raw, dtype=dtype).reshape((nz, ny, nx))
		return arr, (nz, ny, nx)


def discover_patients(source_dir: Path) -> list[int]:
	"""Return sorted volume indices with both an image and a label file."""
	images_dir = source_dir / "imagesTr"
	labels_dir = source_dir / "labelsTr"
	if not images_dir.is_dir() or not labels_dir.is_dir():
		raise FileNotFoundError(f"{source_dir} must contain 'imagesTr/' and 'labelsTr/'.")
	patients = sorted(
		int(path.name.split("_")[1].split(".")[0]) for path in images_dir.glob("liver_*.nii.gz")
	)
	missing = [index for index in patients if not (labels_dir / f"liver_{index}.nii.gz").is_file()]
	if missing:
		raise FileNotFoundError(f"Missing label files for volumes: {missing[:10]}.")
	if not patients:
		raise FileNotFoundError(f"No liver_*.nii.gz volumes found in {images_dir}.")
	return patients


def scan_patient(
	source_dir: Path,
	vol_idx: int,
	sizes: Sequence[int],
) -> list[SliceRecord]:
	"""Extract every eligible liver-centred square crop of one patient."""
	from PIL import Image

	pil_resampling = Image.Resampling.LANCZOS
	arr, (_, ny, nx) = read_nifti_gz(source_dir / "imagesTr" / f"liver_{vol_idx}.nii.gz")
	lbl, _ = read_nifti_gz(source_dir / "labelsTr" / f"liver_{vol_idx}.nii.gz")
	liver_mask = lbl > 0
	arr_w = np.clip((arr - LITS_HU_MIN) / (LITS_HU_MAX - LITS_HU_MIN) * 255.0, 0.0, 255.0).astype(
		np.uint8
	)

	records: list[SliceRecord] = []
	tier_sizes = tuple(sorted(sizes))
	zs_with_liver = np.where(np.any(liver_mask, axis=(1, 2)))[0]
	for z in zs_with_liver:
		s_mask = liver_mask[z]
		ys, xs = np.where(s_mask)
		side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
		organ_side = int(side)
		if organ_side < tier_sizes[0]:
			continue

		side = min(int(side * LITS_CROP_MARGIN), min(ny, nx))
		yc, xc = int(np.mean(ys)), int(np.mean(xs))
		y1 = max(0, min(ny - side, yc - side // 2))
		x1 = max(0, min(nx - side, xc - side // 2))

		slice_raw = arr[z]
		air_frac = np.mean(slice_raw[y1 : y1 + side, x1 : x1 + side] <= LITS_AIR_HU_MAX)
		if air_frac > LITS_MAX_AIR_FRACTION:
			shifts = range(-LITS_AIR_SHIFT_PX, LITS_AIR_SHIFT_PX + 1, LITS_AIR_SHIFT_STEP_PX)
			for dx in shifts:
				if dx == 0:
					continue
				nx1 = max(0, min(nx - side, x1 + dx))
				n_air = np.mean(slice_raw[y1 : y1 + side, nx1 : nx1 + side] <= LITS_AIR_HU_MAX)
				if n_air < air_frac:
					x1 = nx1
					air_frac = n_air
		if air_frac > LITS_MAX_AIR_FRACTION:
			continue

		crop = arr_w[z, y1 : y1 + side, x1 : x1 + side]
		images = {
			size: np.array(Image.fromarray(crop).resize((size, size), pil_resampling))
			for size in tier_sizes
			if organ_side >= size
		}
		records.append(SliceRecord(patient=vol_idx, z=int(z), images=images))
	return records


def load_lits_split(
	path: str | Path = LITS_SPLIT_PATH,
) -> tuple[dict[str, list[tuple[int, int]]], dict[int, dict[str, int]]]:
	"""Load and validate the authoritative LiTS split manifest.

	Returns:
		Tuple of (per-split (patient, z) slices in archive order, per-size slice counts).
	"""
	payload = torch.load(Path(path), map_location="cpu", weights_only=True)
	slices: dict[str, list[tuple[int, int]]] = {}
	for split in LITS_SPLITS:
		table = payload["slices"][split]
		if table.dtype != torch.int32 or table.ndim != 2 or table.shape[1] != 2:
			raise ValueError(f"LiTS split '{split}' must be an int32 [N, 2] tensor.")
		slices[split] = [(int(patient), int(z)) for patient, z in table.tolist()]
		if len(set(slices[split])) != len(slices[split]):
			raise ValueError(f"LiTS split '{split}' repeats a slice.")
	owner: dict[int, str] = {}
	for split, keys in slices.items():
		for patient in {patient for patient, _ in keys}:
			if patient in owner:
				raise ValueError(
					f"Patient {patient} appears in both '{owner[patient]}' and '{split}'."
				)
			owner[patient] = split
	counts: dict[int, dict[str, int]] = {
		int(size): {split: int(per_split[split]) for split in LITS_SPLITS}
		for size, per_split in payload["counts"].items()
	}
	for size, per_split in counts.items():
		for split, count in per_split.items():
			if not 0 < count <= len(slices[split]):
				raise ValueError(f"LiTS count {count} for '{split}' at {size}px is out of range.")
	return slices, counts


def order_by_split(
	records_by_patient: dict[int, list[SliceRecord]],
	slices: dict[str, list[tuple[int, int]]],
	counts: dict[int, dict[str, int]],
	sizes: Sequence[int],
) -> dict[int, dict[str, list[SliceRecord]]]:
	"""Arrange scanned slices in the manifest order for every requested size.

	Fails fast when the scan disagrees with the manifest: a listed slice is missing or not
	eligible at that size, or a split patient has an eligible slice the manifest omits.
	"""
	scanned = {
		(record.patient, record.z): record
		for records in records_by_patient.values()
		for record in records
	}
	ordered: dict[int, dict[str, list[SliceRecord]]] = {}
	for size in sizes:
		if size not in counts:
			raise ValueError(f"The LiTS split manifest has no counts for {size}px.")
		ordered[size] = {}
		for split, keys in slices.items():
			expected = keys[: counts[size][split]]
			patients = {patient for patient, _ in keys}
			eligible = {
				key
				for key, record in scanned.items()
				if key[0] in patients and size in record.images
			}
			if eligible != set(expected):
				missing = sorted(set(expected) - eligible)[:5]
				extra = sorted(eligible - set(expected))[:5]
				raise RuntimeError(
					f"LiTS scan disagrees with the split manifest for '{split}' at {size}px: "
					f"missing {missing}, unlisted {extra}."
				)
			ordered[size][split] = [scanned[key] for key in expected]
	return ordered


def archive_manifest(
	ordered: dict[int, dict[str, list[SliceRecord]]],
	sizes: Sequence[int],
	source_dir: Path,
) -> dict[str, Any]:
	"""Summarize one archive build into the payload of `LITS_ARCHIVE_MANIFEST_PATH`.

	The counts are the artifact-level facts the manuscript reports. The other fields identify
	the build inputs, so a later build can detect a changed split manifest, changed generator
	constants, or changed source volumes without reading the archives.

	Args:
		ordered: Scanned slices per size and split, in manifest order.
		sizes: Requested image sizes.
		source_dir: LiTS source directory holding `imagesTr/` and `labelsTr/`.

	Returns:
		Manifest payload.
	"""
	counts: dict[int, dict[str, torch.Tensor]] = {}
	patients: set[int] = set()
	for size in sizes:
		counts[size] = {}
		for split in LITS_SPLITS:
			records = ordered[size][split]
			patients.update(record.patient for record in records)
			counts[size][split] = torch.tensor(
				[len({record.patient for record in records}), len(records)],
				dtype=torch.int32,
			)
	with open(LITS_SPLIT_PATH, "rb") as handle:
		split_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
	return {
		"patient_ids": torch.tensor(sorted(patients), dtype=torch.int32),
		"source_bytes": torch.tensor(
			[
				[
					(source_dir / "imagesTr" / f"liver_{patient}.nii.gz").stat().st_size,
					(source_dir / "labelsTr" / f"liver_{patient}.nii.gz").stat().st_size,
				]
				for patient in sorted(patients)
			],
			dtype=torch.int64,
		),
		"sizes": torch.tensor(list(sizes), dtype=torch.int32),
		"counts": counts,
		"split_manifest_sha256": split_sha256,
		"generator": {
			"hu_min": LITS_HU_MIN,
			"hu_max": LITS_HU_MAX,
			"crop_margin": LITS_CROP_MARGIN,
			"max_air_fraction": LITS_MAX_AIR_FRACTION,
			"air_hu_max": LITS_AIR_HU_MAX,
			"air_shift_px": LITS_AIR_SHIFT_PX,
			"air_shift_step_px": LITS_AIR_SHIFT_STEP_PX,
		},
	}


def load_archive_manifest(
	path: str | Path = LITS_ARCHIVE_MANIFEST_PATH,
) -> dict[int, dict[str, tuple[int, int]]]:
	"""Load the archive counts per size and split from the tracked manifest.

	Args:
		path: Manifest path.

	Returns:
		Mapping of size to split to (volume count, slice count).
	"""
	payload = torch.load(Path(path), map_location="cpu", weights_only=True)
	manifest: dict[int, dict[str, tuple[int, int]]] = {}
	for size, per_split in payload["counts"].items():
		manifest[int(size)] = {}
		for split in LITS_SPLITS:
			volumes, slices = (int(value) for value in per_split[split].tolist())
			if not 0 < volumes <= slices:
				raise ValueError(
					f"LiTS archive manifest {path} reports {volumes} volumes and {slices} "
					f"slices for '{split}' at {size}px."
				)
			manifest[int(size)][split] = (volumes, slices)
	return manifest


def prepare_archives(
	which: str = "lits",
	*,
	source_dir: str | Path | None = None,
	archive_dir: str | Path | None = None,
	sizes: Sequence[int] | None = None,
	dry_run: bool = False,
	manifest_path: str | Path | None = None,
) -> dict[int, Path]:
	"""Build the LiTS archives from `data/lits_raw/` with the authoritative split.

	The patient split and the slice order come from the split manifest
	(`config.LITS_SPLIT_PATH`). The build uses no randomness. Each size archive holds the
	first manifest slices of each split. Larger sizes are prefixes of smaller ones.

	Args:
		which: Source identifier. Only 'lits' is supported.
		source_dir: Optional explicit LiTS source directory.
		archive_dir: Optional explicit archive output directory.
		sizes: Optional resolution subset.
		dry_run: If True, scan and verify but write nothing.
		manifest_path: Optional explicit manifest output path; defaults to
			`config.LITS_ARCHIVE_MANIFEST_PATH`.

	Returns:
		Mapping of size to written archive path (empty in dry-run mode).
	"""
	if which != "lits":
		raise ValueError(f"Only 'lits' archive preparation is supported; got {which!r}.")
	chosen_sizes = (
		tuple(sorted(sizes)) if sizes is not None else tuple(sorted(CANONICAL_MEDMNIST_SIZES))
	)
	source = Path(source_dir) if source_dir is not None else LITS_RAW_DIR
	base = Path(archive_dir) if archive_dir is not None else ARCHIVE_DIR

	slices, counts = load_lits_split()
	split_patients = sorted({patient for keys in slices.values() for patient, _ in keys})
	missing_volumes = sorted(set(split_patients) - set(discover_patients(source)))
	if missing_volumes:
		raise FileNotFoundError(
			f"LiTS volumes in the split manifest are missing: {missing_volumes}."
		)
	records_by_patient = {
		vol_idx: scan_patient(source, vol_idx, chosen_sizes) for vol_idx in split_patients
	}
	ordered = order_by_split(records_by_patient, slices, counts, chosen_sizes)
	if dry_run:
		return {}

	base.mkdir(parents=True, exist_ok=True)
	outputs: dict[int, Path] = {}
	for size in chosen_sizes:
		payload: dict[str, np.ndarray] = {}
		for split in LITS_SPLITS:
			records = ordered[size][split]
			payload[f"{split}_images"] = np.stack([record.images[size] for record in records])
			payload[f"{split}_group_ids"] = np.asarray(
				[record.patient for record in records], dtype=np.int64
			)
			payload[f"{split}_slice_ids"] = np.asarray(
				[record.z for record in records], dtype=np.int64
			)
		out_path = base / f"lits_{size}.npz"
		with atomic_write(out_path) as tmp_path, open(tmp_path, "wb") as handle:
			np.savez_compressed(handle, **payload)
		outputs[size] = out_path
	save_pt(
		archive_manifest(ordered, chosen_sizes, source),
		Path(manifest_path) if manifest_path is not None else LITS_ARCHIVE_MANIFEST_PATH,
	)
	return outputs


# ===========================================================================
# Generation infrastructure
# ===========================================================================


def _f32(value: float) -> float:
	"""Round a scalar to float32 precision and return it as a Python float."""
	return float(torch.tensor(value, dtype=torch.float32))


def _resolve_output_dir(workload: dict[str, Any], output_dir: str | Path | None) -> Path:
	"""Return the explicit output directory, else the workload's declared directory."""
	if output_dir is not None:
		return Path(output_dir)
	declared = workload.get("output_dir")
	if declared is None:
		raise ValueError(
			f"Workload '{workload.get('name')}' has no 'output_dir'; pass output_dir explicitly."
		)
	return Path(str(declared))


def _next_free_dataset_path(directory: Path, name: str) -> Path:
	"""Return the next free `{name}_00N.pt`, treating existing `.pt.xz` files as taken."""
	index = 1
	while any((directory / f"{name}_{index:03d}{suffix}").exists() for suffix in (".pt", ".pt.xz")):
		index += 1
	return directory / f"{name}_{index:03d}.pt"


def _checkpoint_path(output_path: Path) -> Path:
	"""Return the resume-checkpoint path for a generation target."""
	return output_path.with_name(output_path.name + GENERATION_CHECKPOINT_SUFFIX)


def _capture_rng_state(device: torch.device) -> dict[str, torch.Tensor]:
	"""Capture the global CPU RNG state and the device RNG state when available."""
	state = {"cpu": torch.get_rng_state()}
	mps = getattr(torch, "mps", None)
	if device.type == "cuda":
		state["device"] = torch.cuda.get_rng_state(device)
	elif device.type == "mps" and mps is not None and hasattr(mps, "get_rng_state"):
		state["device"] = mps.get_rng_state()
	return state


def _restore_rng_state(state: dict[str, torch.Tensor], device: torch.device) -> None:
	"""Restore a global RNG state captured by `_capture_rng_state`."""
	torch.set_rng_state(state["cpu"])
	mps = getattr(torch, "mps", None)
	if device.type == "cuda" and "device" in state:
		torch.cuda.set_rng_state(state["device"], device)
	elif (
		device.type == "mps"
		and "device" in state
		and mps is not None
		and hasattr(mps, "set_rng_state")
	):
		mps.set_rng_state(state["device"])


def _archive_signature(archive_dir: str | Path | None) -> str:
	"""Return the resolved archive directory used to validate a checkpoint."""
	base = Path(archive_dir) if archive_dir is not None else ARCHIVE_DIR
	return str(base.resolve())


def _load_checkpoint(
	path: Path,
	signature: dict[str, Any],
	required: Sequence[str],
) -> dict[str, Any] | None:
	"""Load a generation checkpoint and reject a different workload configuration."""
	if not path.exists():
		return None
	state = load_pt(path)
	if state.get("signature") != signature:
		raise ValueError(
			f"Checkpoint {path} was written for a different workload configuration; "
			"delete it or use --fresh to start over."
		)
	for key in required:
		if key not in state:
			raise ValueError(f"Checkpoint {path} is missing '{key}'.")
	return state


def _solver_config() -> dict[str, Any]:
	"""Return the solver stopping-rule provenance shared by both generation paths."""
	return {
		"algorithm": "Chambolle-Pock (PDHG 2011)",
		"stopping_rule": "calibrated fixed K, linear in log eta between tier values",
		"k_tiers": {int(size): list(ks) for size, ks in SOLVER_K_TIERS.items()},
		"nsr_tiers": list(DEFAULT_METROLOGY_NSR_TIERS),
		"dominance_ratio": METROLOGY_DOMINANCE_RATIO,
		"delta_k": DEFAULT_METROLOGY_DELTA_K,
		"tau": DEFAULT_CP_TAU,
		"sigma": DEFAULT_CP_SIGMA,
		"theta": DEFAULT_CP_THETA,
		"tau_mc": DEFAULT_TAU,
	}


# ===========================================================================
# Stochastic generation
# ===========================================================================


def _package_records_to_tensors(
	records: list[dict[str, Any]],
	empty_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
	"""Package stochastic sample records into the canonical tensor dictionary."""
	if not records:
		return {
			"observations_y": torch.empty((0, 1, empty_size, empty_size), dtype=torch.float32),
			"x_clean": torch.empty((0, 1, empty_size, empty_size), dtype=torch.float32),
			"dataset_names": [],
			"img_indices": torch.empty((0,), dtype=torch.int32),
			"mc_gdf": torch.empty((0,), dtype=torch.float32),
			"gdf_n": torch.empty((0,), dtype=torch.float32),
			"lambda_tv": torch.empty((0,), dtype=torch.float32),
			"sigma": torch.empty((0,), dtype=torch.float32),
			"image_sizes": torch.empty((0, 2), dtype=torch.int32),
			"n_pixels": torch.empty((0,), dtype=torch.int32),
			"sample_durations": torch.empty((0,), dtype=torch.float32),
			"solve_times": torch.empty((0,), dtype=torch.float32),
			"k_solvers": torch.empty((0,), dtype=torch.int32),
			"gdf_n_se": torch.empty((0,), dtype=torch.float32),
		}

	obs_list = [
		torch.as_tensor(record["observations_y"], dtype=torch.float32) for record in records
	]
	clean_list = [torch.as_tensor(record["x_clean"], dtype=torch.float32) for record in records]
	can_stack = all(obs.shape == obs_list[0].shape for obs in obs_list)
	payload: dict[str, Any] = {
		"observations_y": torch.stack(obs_list, dim=0) if can_stack else obs_list,
		"x_clean": torch.stack(clean_list, dim=0) if can_stack else clean_list,
		"dataset_names": [record["dataset_name"] for record in records],
		"img_indices": torch.tensor([record["img_idx"] for record in records], dtype=torch.int32),
		"mc_gdf": torch.tensor([record["mc_gdf"] for record in records], dtype=torch.float32),
		"gdf_n": torch.tensor([record["gdf_n"] for record in records], dtype=torch.float32),
		"lambda_tv": torch.tensor([record["lambda_tv"] for record in records], dtype=torch.float32),
		"sigma": torch.tensor([record["sigma"] for record in records], dtype=torch.float32),
		"image_sizes": torch.tensor(
			[record["image_size"] for record in records], dtype=torch.int32
		),
		"n_pixels": torch.tensor([record["n_pixels"] for record in records], dtype=torch.int32),
		"sample_durations": torch.tensor(
			[record["sample_duration"] for record in records], dtype=torch.float32
		),
		"solve_times": torch.tensor(
			[record["solve_time"] for record in records], dtype=torch.float32
		),
		"k_solvers": torch.tensor([record["k_solver"] for record in records], dtype=torch.int32),
		"gdf_n_se": torch.tensor([record["gdf_n_se"] for record in records], dtype=torch.float32),
	}
	return payload


def _scaled_split_counts(splits: dict[str, int], samples: int | None) -> dict[str, int]:
	"""Return per-split, per-resolution sample counts, rescaled when `samples` is given.

	Every split after the first gets its rounded share; the first split gets the rest.
	"""
	if samples is None:
		return dict(splits)
	total = sum(splits.values())
	names = list(splits)
	counts = {name: int(round(samples * splits[name] / total)) for name in names[1:]}
	return {names[0]: samples - sum(counts.values()), **counts}


def generate_stochastic(
	workload: dict[str, Any],
	*,
	samples: int | None = None,
	seed: int | None = None,
	device: torch.device = DEFAULT_DEVICE,
	archive_dir: str | Path | None = None,
	output_dir: str | Path | None = None,
	dry_run: bool = False,
	resume: bool = True,
) -> Path:
	"""Generate one stochastic dataset as a new `{name}_00N.pt`.

	Each sample draws a modality, an image, a relative noise level and a λ at random. The
	workload fixes the name, the image sizes, the sampling configuration and the number of
	samples per archive split and per image size. A workload with several splits
	(training) writes one sub-dict per split; a single-split workload writes one flat dict.

	Args:
		workload: Stochastic workload, e.g. `config.DEFAULT_TRAINING_WORKLOAD`,
			`DEFAULT_PARITY_WORKLOAD`, `DEFAULT_SCALE_PARITY_WORKLOAD`, or
			`DEFAULT_VAL_PARITY_WORKLOAD`.
		samples: Optional sample count per resolution. The split proportions are kept.
		seed: Optional random seed for sampling. None leaves sampling unseeded.
		device: Execution device.
		archive_dir: Optional explicit archive directory.
		output_dir: Optional output directory. Defaults to the workload's `output_dir`.
		dry_run: If True, print the resolved configuration and write nothing.
		resume: If True, continue from an existing checkpoint for the output path.

	Returns:
		Output path, the next free `{name}_00N.pt` in the output directory.
	"""
	name = str(workload["name"])
	sampling_config = dict(workload["sampling_config"])
	parsed_sizes = [int(value) for value in workload["sizes"]]
	if not parsed_sizes:
		raise ValueError(f"Workload '{name}' has no image sizes.")
	per_scale_splits = _scaled_split_counts(dict(workload["splits"]), samples)
	split_targets = {split: count * len(parsed_sizes) for split, count in per_scale_splits.items()}
	samples = sum(split_targets.values())
	base = _resolve_output_dir(workload, output_dir)
	output_path = _next_free_dataset_path(base, name)

	if dry_run:
		print(
			f"[dry-run] stochastic {name}: sizes={parsed_sizes} "
			f"splits_per_scale={per_scale_splits} splits_total={split_targets} "
			f"-> {output_path}"
		)
		return output_path

	nsr_min = float(sampling_config["nsr_min"])
	nsr_max = float(sampling_config["nsr_max"])
	if nsr_min <= 0.0 or nsr_max <= nsr_min:
		raise ValueError(f"Invalid relative noise range: [{nsr_min}, {nsr_max}].")

	split_ends = dict(zip(split_targets, itertools.accumulate(split_targets.values()), strict=True))
	single_split = next(iter(split_targets)) if len(split_targets) == 1 else None

	master_rng = torch.Generator().manual_seed(seed) if seed is not None else None
	signature = {
		"kind": "stochastic",
		"name": name,
		"sizes": parsed_sizes,
		"per_scale_splits": per_scale_splits,
		"total_samples": samples,
		"sampling_config": sampling_config,
		"seed": seed,
		"archive_dir": _archive_signature(archive_dir),
		"solver_config": _solver_config(),
	}
	checkpoint_path = _checkpoint_path(output_path)
	if not resume:
		checkpoint_path.unlink(missing_ok=True)
	state = _load_checkpoint(
		checkpoint_path,
		signature,
		(
			"certified",
			"seen_samples",
			"cfg_idx",
			"master_rng_state",
			"rng_state",
			"elapsed_seconds",
		),
	)

	seen_samples: set[tuple[str, str, int]] = set()
	certified: list[dict[str, Any]] = []
	cfg_idx = 0
	elapsed_before = 0.0
	if state is not None:
		certified = state["certified"]
		seen_samples = {tuple(key) for key in state["seen_samples"]}
		cfg_idx = int(state["cfg_idx"])
		elapsed_before = float(state["elapsed_seconds"])
		if master_rng is not None:
			if state["master_rng_state"] is None:
				raise ValueError(f"Checkpoint {checkpoint_path} has no generator state.")
			master_rng.set_state(state["master_rng_state"])
		_restore_rng_state(state["rng_state"], device)
		print(
			f"[resume] {name}: {len(certified)}/{samples} samples from "
			f"{display_path(checkpoint_path)}"
		)

	modality_split_sizes: dict[tuple[str, str, int], int] = {}
	for size in parsed_sizes:
		for modality in CANONICAL_MEDMNIST_DATASETS:
			for split in split_targets:
				modality_split_sizes[(modality, split, size)] = get_medmnist_split_size(
					modality, split=split, size=size, archive_dir=archive_dir
				)

	max_attempts = max(samples * 10, 50)
	job_start = time.perf_counter()
	last_checkpoint = job_start

	def save_checkpoint() -> None:
		"""Persist the resumable generator state."""
		save_pt(
			{
				"signature": signature,
				"certified": certified,
				"seen_samples": sorted(seen_samples),
				"cfg_idx": cfg_idx,
				"master_rng_state": (master_rng.get_state() if master_rng is not None else None),
				"rng_state": _capture_rng_state(device),
				"elapsed_seconds": elapsed_before + (time.perf_counter() - job_start),
			},
			checkpoint_path,
		)

	try:
		while len(certified) < samples and cfg_idx < max_attempts:
			cfg_idx += 1
			sample_start = time.perf_counter()
			size = parsed_sizes[len(certified) % len(parsed_sizes)]
			n_pix = size * size

			n_cert = len(certified)
			phase = next(split for split, end in split_ends.items() if n_cert < end)

			dataset_index = int(
				torch.randint(len(CANONICAL_MEDMNIST_DATASETS), (1,), generator=master_rng)
			)
			dataset_name = CANONICAL_MEDMNIST_DATASETS[dataset_index]
			total_available = modality_split_sizes[(dataset_name, phase, size)]
			img_idx = int(torch.randint(total_available, (1,), generator=master_rng))
			key = (phase, dataset_name, img_idx)
			if key in seen_samples:
				continue
			seen_samples.add(key)

			x_clean = load_medmnist_images(
				dataset_name, split=phase, indices=img_idx, size=size, archive_dir=archive_dir
			).to(device)
			x_std = image_std(x_clean)
			u_nsr = float(torch.rand((), generator=master_rng))
			nsr = _f32(nsr_min + (nsr_max - nsr_min) * u_nsr)
			sigma = _f32(nsr * x_std)
			noise_seed = (
				int(torch.randint(0, 2**31 - 1, (1,), generator=master_rng))
				if seed is not None
				else None
			)
			y = create_noisy_observation(x_clean, sigma, seed=noise_seed)

			dist = str(sampling_config["lambda_dist"])
			u = float(torch.rand((), generator=master_rng))
			if dist == "regime_logit":
				u_min = float(sampling_config["u_min"])
				u_max = float(sampling_config["u_max"])
				u_lam = u_min + (u_max - u_min) * u
				log_lam = _f32(
					float(sampling_config["denoise_x0"])
					+ float(sampling_config["denoise_w"]) * math.log10((1.0 - u_lam) / u_lam)
				)
			elif dist == "uniform":
				log_min = math.log10(float(sampling_config["lambda_min"]))
				log_max = math.log10(float(sampling_config["lambda_max"]))
				log_lam = _f32(log_min + (log_max - log_min) * u)
			else:
				raise ValueError(f"Unsupported lambda_dist: '{dist}'")
			lambda_tv = _f32(10.0**log_lam)

			active_m = DEFAULT_M_BY_SIZE[size]

			result = compute_mc_gdf(
				y=y,
				lambda_tv=lambda_tv,
				m_mc=active_m,
				k=solver_k(size, nsr),
				tau=DEFAULT_TAU,
			)
			solve_time = float(time.perf_counter() - sample_start)

			certified.append(
				{
					"observations_y": y[0].detach().cpu(),
					"x_clean": x_clean[0].detach().cpu(),
					"dataset_name": dataset_name,
					"phase": phase,
					"img_idx": img_idx,
					"image_size": (size, size),
					"n_pixels": n_pix,
					"sigma": sigma,
					"nsr": nsr,
					"k_solver": result.k_solver,
					"lambda_tv": lambda_tv,
					"mc_gdf": result.gdf_n * n_pix,
					"gdf_n": result.gdf_n,
					"gdf_n_se": result.gdf_n_se,
					"sample_duration": float(time.perf_counter() - sample_start),
					"solve_time": solve_time,
					"m_mc": active_m,
					"seed": noise_seed,
				}
			)
			release_device_cache()
			now = time.perf_counter()
			if now - last_checkpoint >= GENERATION_CHECKPOINT_INTERVAL_SECONDS:
				save_checkpoint()
				last_checkpoint = now
	finally:
		if len(certified) < samples:
			save_checkpoint()

	if len(certified) < samples:
		raise RuntimeError(
			f"Generator produced {len(certified)}/{samples} certified samples after "
			f"{cfg_idx} attempts."
		)

	total_time = elapsed_before + float(time.perf_counter() - job_start)
	provenance = {
		"test_dataset_name": name if single_split is not None else None,
		"split": single_split,
		"n_samples": len(certified),
		"scales": parsed_sizes,
		"seed": seed,
		"sampling_config": sampling_config,
		"solver_config": {
			**_solver_config(),
			"m_by_size": dict(
				sorted({rec["image_size"][0]: rec["m_mc"] for rec in certified}.items())
			),
		},
		"timing": {"total_time_seconds": total_time},
	}

	if single_split is not None:
		payload = _package_records_to_tensors(certified, empty_size=parsed_sizes[0])
		result = save_pt({**payload, "provenance": provenance}, output_path)
	else:
		records_by_phase: dict[str, list[dict[str, Any]]] = {split: [] for split in split_targets}
		for record in certified:
			records_by_phase[record["phase"]].append(record)
		payload = {
			split: _package_records_to_tensors(records, empty_size=parsed_sizes[0])
			for split, records in records_by_phase.items()
		}
		provenance.update(
			{f"n_{split}": len(records) for split, records in records_by_phase.items()}
		)
		result = save_pt({**payload, "provenance": provenance}, output_path)
	checkpoint_path.unlink(missing_ok=True)
	return result


# ===========================================================================
# Sweeps generation
# ===========================================================================


def generate_sweeps(
	workload: dict[str, Any],
	*,
	sweep_points: int | None = None,
	device: torch.device = DEFAULT_DEVICE,
	archive_dir: str | Path | None = None,
	output_dir: str | Path | None = None,
	dry_run: bool = False,
	resume: bool = True,
) -> Path:
	"""Generate one sweeps dataset as a new `{name}_00N.pt`.

	Each regime is a fixed image and noise level, evaluated on a fixed grid of
	`sweep_points` λ values. The workload fixes the name, the scales, the regimes and the
	default grid size.

	Args:
		workload: Sweeps workload, `config.DEFAULT_SWEEPS_WORKLOAD` or
			`DEFAULT_SCALE_SWEEPS_WORKLOAD`.
		sweep_points: Optional number of λ values per regime.
		device: Execution device.
		archive_dir: Optional explicit archive directory.
		output_dir: Optional output directory. Defaults to the workload's `output_dir`.
		dry_run: If True, print the resolved configuration and write nothing.
		resume: If True, continue from an existing checkpoint for the output path.

	Returns:
		Output path, the next free `{name}_00N.pt` in the output directory.
	"""
	name = str(workload["name"])
	test_dataset_name = name
	parsed_scales = [int(value) for value in workload["scales"]]
	regimes = [dict(regime) for regime in workload["regimes"]]
	sweep_points = int(sweep_points) if sweep_points is not None else int(workload["sweep_points"])
	base = _resolve_output_dir(workload, output_dir)
	output_path = _next_free_dataset_path(base, name)

	if dry_run:
		print(
			f"[dry-run] sweeps {name}: scales={parsed_scales} regimes={len(regimes)} "
			f"sweep_points={sweep_points} -> {output_path}"
		)
		return output_path

	signature = {
		"kind": "sweeps",
		"name": name,
		"scales": parsed_scales,
		"sweep_points": sweep_points,
		"regimes": regimes,
		"archive_dir": _archive_signature(archive_dir),
		"solver_config": _solver_config(),
	}
	checkpoint_path = _checkpoint_path(output_path)
	if not resume:
		checkpoint_path.unlink(missing_ok=True)
	state = _load_checkpoint(
		checkpoint_path,
		signature,
		(
			"next_regime",
			"gdf_n_all",
			"gdf_n_se_all",
			"ssrs_all",
			"solve_times_all",
			"k_solvers_all",
			"sigmas",
			"sigma_mads",
			"n_pixels",
			"observations_y",
			"mc_gdf_baselines",
			"lambdas_all",
			"m_by_size",
			"rng_state",
			"elapsed_seconds",
		),
	)

	all_regimes: list[dict[str, Any]] = []
	for size in parsed_scales:
		image_map: dict[tuple[str, int, str], torch.Tensor] = {}
		for regime in regimes:
			tag = str(regime["img_tag"])
			img_idx = int(regime.get("img_idx", 0))
			split_name = str(regime.get("split", "test"))
			key = (tag, img_idx, split_name)
			if key not in image_map:
				if tag not in CANONICAL_MEDMNIST_DATASETS:
					raise ValueError(f"Unknown image tag '{tag}' for sweeps regime.")
				image_map[key] = load_medmnist_images(
					tag,
					split=split_name,
					indices=img_idx,
					size=size,
					archive_dir=archive_dir,
				).to(device)
		for regime in regimes:
			regime_copy = dict(regime)
			if len(parsed_scales) > 1:
				regime_copy["title"] = f"{regime['title']} ({size}x{size})"
			regime_copy["shape"] = (size, size)
			img_idx = int(regime_copy.get("img_idx", 0))
			split_name = str(regime_copy.get("split", "test"))
			regime_copy["img"] = image_map[(str(regime_copy.pop("img_tag")), img_idx, split_name)]
			all_regimes.append(regime_copy)

	job_start = time.perf_counter()
	n_total = len(all_regimes)
	gdf_n_all: list[list[float]] = []
	gdf_n_se_all: list[list[float]] = []
	ssrs_all: list[list[float]] = []
	solve_times_all: list[list[float]] = []
	k_solvers_all: list[int] = []
	sigmas: list[float] = []
	sigma_mads: list[float] = []
	n_pixels: list[int] = []
	observations_y: list[torch.Tensor] = []
	mc_gdf_baselines: list[float] = []
	lambdas_all: list[torch.Tensor] = []
	m_by_size: dict[int, int] = {}
	next_regime = 0
	elapsed_before = 0.0
	if state is not None:
		next_regime = int(state["next_regime"])
		gdf_n_all = state["gdf_n_all"]
		gdf_n_se_all = state["gdf_n_se_all"]
		ssrs_all = state["ssrs_all"]
		solve_times_all = state["solve_times_all"]
		k_solvers_all = state["k_solvers_all"]
		sigmas = state["sigmas"]
		sigma_mads = state["sigma_mads"]
		n_pixels = state["n_pixels"]
		observations_y = state["observations_y"]
		mc_gdf_baselines = state["mc_gdf_baselines"]
		lambdas_all = state["lambdas_all"]
		m_by_size = state["m_by_size"]
		elapsed_before = float(state["elapsed_seconds"])
		_restore_rng_state(state["rng_state"], device)
		print(
			f"[resume] {name}: {next_regime}/{n_total} regimes from {display_path(checkpoint_path)}"
		)

	def save_checkpoint() -> None:
		"""Persist the resumable generator state."""
		save_pt(
			{
				"signature": signature,
				"next_regime": next_regime,
				"gdf_n_all": gdf_n_all,
				"gdf_n_se_all": gdf_n_se_all,
				"ssrs_all": ssrs_all,
				"solve_times_all": solve_times_all,
				"k_solvers_all": k_solvers_all,
				"sigmas": sigmas,
				"sigma_mads": sigma_mads,
				"n_pixels": n_pixels,
				"observations_y": observations_y,
				"mc_gdf_baselines": mc_gdf_baselines,
				"lambdas_all": lambdas_all,
				"m_by_size": m_by_size,
				"rng_state": _capture_rng_state(device),
				"elapsed_seconds": elapsed_before + (time.perf_counter() - job_start),
			},
			checkpoint_path,
		)

	try:
		for regime_index, regime in enumerate(all_regimes):
			if regime_index < next_regime:
				continue
			img_tensor = regime["img"]
			reg_nsr = float(regime.get("nsr", 0.50))
			img_std = image_std(img_tensor)
			sigma_val = _f32(reg_nsr * img_std)
			h, w = regime["shape"]
			n_pix = h * w
			y = create_noisy_observation(img_tensor, sigma_val)
			sigma_mad_val = _f32(estimate_noise_wavelet_mad(y).sigma)

			active_m = DEFAULT_M_BY_SIZE[h]
			m_by_size[h] = active_m
			k_regime = solver_k(h, reg_nsr)

			reg_lambdas = (
				torch.logspace(
					DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MIN,
					DEFAULT_SWEEP_LOG_LAMBDA_OVER_SIGMA_MAD_MAX,
					sweep_points,
					dtype=torch.float32,
				)
				* sigma_mad_val
			)

			reg_gdf_n: list[float] = []
			reg_gdf_n_se: list[float] = []
			reg_ssrs: list[float] = []
			reg_solve_times: list[float] = []
			for lam in reg_lambdas.tolist():
				start = time.perf_counter()
				result = compute_mc_gdf(
					y=y,
					lambda_tv=lam,
					m_mc=active_m,
					k=k_regime,
					tau=DEFAULT_TAU,
				)
				solve_time = float(time.perf_counter() - start)
				with torch.no_grad():
					y_hat = result.x_hat
					ssr_val = float(torch.sum((y - y_hat) ** 2).item())
				reg_gdf_n.append(result.gdf_n)
				reg_gdf_n_se.append(result.gdf_n_se)
				reg_ssrs.append(ssr_val)
				reg_solve_times.append(solve_time)

			# Append the regime only when it is complete, so an interrupted regime
			# leaves no entries in the checkpoint.
			sigmas.append(sigma_val)
			n_pixels.append(n_pix)
			mc_gdf_baselines.append(float(n_pix))
			observations_y.append(y.detach().cpu())
			sigma_mads.append(sigma_mad_val)
			lambdas_all.append(reg_lambdas)
			gdf_n_all.append(reg_gdf_n)
			gdf_n_se_all.append(reg_gdf_n_se)
			ssrs_all.append(reg_ssrs)
			solve_times_all.append(reg_solve_times)
			k_solvers_all.append(k_regime)
			release_device_cache()
			next_regime = regime_index + 1
			save_checkpoint()
	finally:
		if next_regime < n_total:
			save_checkpoint()

	per_regime = {
		"gdf_n": gdf_n_all,
		"gdf_n_se": gdf_n_se_all,
		"ssrs": ssrs_all,
		"solve_times": solve_times_all,
		"k_solvers": k_solvers_all,
		"sigmas": sigmas,
		"sigma_mads": sigma_mads,
		"n_pixels": n_pixels,
		"observations_y": observations_y,
		"mc_gdf_baseline": mc_gdf_baselines,
		"lambdas": lambdas_all,
	}
	lengths = {key: len(values) for key, values in per_regime.items()}
	if set(lengths.values()) != {n_total}:
		raise RuntimeError(f"Per-regime arrays do not match {n_total} regimes: {lengths}")

	regimes_metadata = []
	for regime in all_regimes:
		regime_copy = regime.copy()
		del regime_copy["img"]
		regimes_metadata.append(regime_copy)

	can_concat = all(obs.shape == observations_y[0].shape for obs in observations_y)
	obs_payload = torch.cat(observations_y, dim=0) if can_concat else observations_y
	clean_imgs = [regime["img"] for regime in all_regimes]
	can_concat_clean = all(clean.shape == clean_imgs[0].shape for clean in clean_imgs)
	clean_payload = torch.cat(clean_imgs, dim=0) if can_concat_clean else clean_imgs

	payload = {
		"observations_y": obs_payload,
		"x_clean": clean_payload,
		"gdf_n": torch.tensor(gdf_n_all, dtype=torch.float32),
		"gdf_n_se": torch.tensor(gdf_n_se_all, dtype=torch.float32),
		"lambdas": torch.stack(lambdas_all),
		"ssrs": torch.tensor(ssrs_all, dtype=torch.float32),
		"solve_times": torch.tensor(solve_times_all, dtype=torch.float32),
		"sigmas": torch.tensor(sigmas, dtype=torch.float32),
		"sigma_mads": torch.tensor(sigma_mads, dtype=torch.float32),
		"n_pixels": torch.tensor(n_pixels, dtype=torch.int32),
		"mc_gdf_baseline": torch.tensor(mc_gdf_baselines, dtype=torch.float32),
		"k_solvers": torch.tensor(k_solvers_all, dtype=torch.int32),
	}
	if not torch.isfinite(payload["gdf_n"]).all():
		raise RuntimeError("Generated gdf_n contains NaN or Inf.")
	all_flat_times = [value for row in solve_times_all for value in row]
	provenance = {
		"test_dataset_name": test_dataset_name,
		"sweep_points": sweep_points,
		"n_points_per_regime": sweep_points,
		"n_regimes": n_total,
		"scales": parsed_scales,
		"m_by_size": dict(sorted(m_by_size.items())),
		"tau": DEFAULT_TAU,
		"shapes": [[regime["shape"][0], regime["shape"][1]] for regime in regimes_metadata],
		"regimes": regimes_metadata,
		"solver_config": {
			**_solver_config(),
			"m_by_size": dict(sorted(m_by_size.items())),
		},
		"timing": {
			"total_time_seconds": elapsed_before + float(time.perf_counter() - job_start),
			"mean_time_per_solve_sec": statistics.fmean(all_flat_times) if all_flat_times else 0.0,
		},
	}
	result = save_pt({**payload, "provenance": provenance}, output_path)
	checkpoint_path.unlink(missing_ok=True)
	return result


__all__ = [
	"create_noisy_observation",
	"fetch_data",
	"generate_stochastic",
	"generate_sweeps",
	"get_medmnist_split_size",
	"load_medmnist_images",
	"prepare_archives",
]
