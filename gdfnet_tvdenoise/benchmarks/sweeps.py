"""Continuous regularization sweep benchmark and embedded selection-excess statistics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from gdfnet_tvdenoise.config import (
	DEFAULT_DATASETS,
	DEFAULT_DEVICE,
	DEFAULT_WAVELET_MAD_N_COEFFICIENTS,
	SELECTION_MDP_TARGET,
	SELECTION_QUADRATURE_ABS_TOL,
	SELECTION_QUADRATURE_LIMIT,
	SELECTION_QUADRATURE_REFINEMENT_FACTOR,
	SELECTION_QUADRATURE_REL_TOL,
	WAVELET_MAD_VARIANCE_FACTOR,
)
from gdfnet_tvdenoise.features import extract_features
from gdfnet_tvdenoise.helpers import (
	compute_mdp,
	compute_mdp_known_sigma_uncertainty,
	compute_mdp_selection_excess,
	compute_mdp_uncertainty,
	compute_regression_metrics,
	compute_sure,
	compute_sure_mad_uncertainty,
	compute_sure_selection_excess,
	compute_sure_uncertainty,
	display_path,
	estimate_noise_wavelet_mad,
	load_data_file,
	release_device_cache,
	resolve_model_paths,
	save_pt,
	verify_folded_limits,
)
from gdfnet_tvdenoise.inference import predict
from gdfnet_tvdenoise.model import SpatialGDFNet


def _selection_cases(
	regime_index: int,
	title: str,
	n_pixels: int,
	lambdas: torch.Tensor,
	gdf_n: torch.Tensor,
	gdf_se: torch.Tensor,
	sigma_mad: float,
	n_coefficients: int,
	sure_mad_mc: torch.Tensor,
	sure_mad_surr: torch.Tensor,
	ratio_mc: torch.Tensor,
	ratio_surr: torch.Tensor,
) -> list[dict[str, Any]]:
	"""Compute SURE and MDP selection-excess cases for one regime."""
	size = math.isqrt(n_pixels)
	if size**2 != n_pixels:
		raise ValueError(f"Stored pixel count {n_pixels!r} is not a square.")
	if not bool(torch.all(torch.diff(lambdas) > 0.0)):
		raise ValueError("Stored candidate grid must be strictly ascending.")
	if bool((gdf_se < 0.0).any()):
		raise ValueError("Saved capacity standard errors must be nonnegative.")
	index_sure_surr = int(torch.argmin(sure_mad_surr))
	index_sure_ref = int(torch.argmin(sure_mad_mc))
	sure_result = compute_sure_selection_excess(
		sure_mad_mc,
		gdf_n,
		gdf_se,
		sigma_mad,
		n_coefficients,
		index_sure_surr,
		index_sure_ref,
	)
	index_mdp_surr = int(torch.argmin(torch.abs(ratio_surr - SELECTION_MDP_TARGET)))
	index_mdp_ref = int(torch.argmin(torch.abs(ratio_mc - SELECTION_MDP_TARGET)))
	mdp_result = compute_mdp_selection_excess(
		ratio_mc,
		gdf_n,
		gdf_se,
		sigma_mad,
		n_coefficients,
		index_mdp_surr,
		index_mdp_ref,
	)
	common = {
		"size": size,
		"regime_index": regime_index,
		"title": title,
	}
	sure_case = {
		**common,
		"criterion": "SURE",
		"surrogate_index": index_sure_surr,
		"reference_index": index_sure_ref,
		"same_node": index_sure_surr == index_sure_ref,
		"excess": sure_result.excess,
		"excess_se": sure_result.excess_se,
		"excess_z": sure_result.excess_z,
		"reference_criterion_at_surrogate_selection": float(sure_mad_mc[index_sure_surr]),
		"reference_criterion_at_reference_selection": float(sure_mad_mc[index_sure_ref]),
		"quadrature_error_estimate": sure_result.quadrature_error,
		"refinement_relative_excess_se_change": sure_result.refinement_relative_change,
		"linear_excess_se": sure_result.linear_se,
		"ratio_error_correlation": sure_result.correlation,
	}
	mdp_case = {
		**common,
		"criterion": "MDP",
		"surrogate_index": index_mdp_surr,
		"reference_index": index_mdp_ref,
		"same_node": index_mdp_surr == index_mdp_ref,
		"excess": mdp_result.excess,
		"excess_se": mdp_result.excess_se,
		"excess_z": mdp_result.excess_z,
		"reference_criterion_at_surrogate_selection": abs(
			float(ratio_mc[index_mdp_surr]) - SELECTION_MDP_TARGET
		),
		"reference_criterion_at_reference_selection": abs(
			float(ratio_mc[index_mdp_ref]) - SELECTION_MDP_TARGET
		),
		"quadrature_error_estimate": mdp_result.quadrature_error,
		"refinement_relative_excess_se_change": mdp_result.refinement_relative_change,
		"linear_excess_se": mdp_result.linear_se,
		"ratio_error_correlation": mdp_result.correlation,
	}
	return [sure_case, mdp_case]


def _summarize_selection_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Group selection-excess cases by criterion and resolution."""
	groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
	for case in cases:
		groups[(case["criterion"], case["size"])].append(case)
	summaries: list[dict[str, Any]] = []
	for (criterion, size), group in sorted(groups.items()):
		changed = [case["excess_z"] for case in group if not case["same_node"]]
		summaries.append(
			{
				"criterion": criterion,
				"size": size,
				"case_count": len(group),
				"same_node_matches": sum(case["same_node"] for case in group),
				"different_node_count": len(changed),
				"different_node_median_excess_z": statistics.median(changed) if changed else None,
				"different_node_max_excess_z": float(max(changed)) if changed else None,
			}
		)
	return summaries


# ===========================================================================
# Sweeps evaluation
# ===========================================================================


def evaluate_sweeps(
	model_path: str | Path,
	suite_name: str = "sweeps",
	device: torch.device = DEFAULT_DEVICE,
) -> tuple[dict[str, Any], Path]:
	"""Evaluate a surrogate on regularization sweeps.

	The surrogate is evaluated on the Monte Carlo node grid. Selections are node-only:
	no interpolation between candidates, so every curve in the report shares one grid.

	Args:
		model_path: Model directory or weights file.
		suite_name: 'sweeps' or 'scale_sweeps'.
		device: PyTorch device.

	Returns:
		Tuple of (report payload, report path).
	"""
	weights, model_dir = resolve_model_paths(model_path)
	model = SpatialGDFNet.load(weights, device=device)
	test_data_paths = DEFAULT_DATASETS[suite_name]
	if len(test_data_paths) != 1:
		raise ValueError(f"Sweeps suite {suite_name!r} requires exactly one dataset.")
	test_data = test_data_paths[0]
	arrays = load_data_file(test_data)

	regimes_metadata = arrays["provenance"].get("regimes", [])
	n_regimes = len(regimes_metadata) or len(arrays["gdf_n"])
	obs_y_arr = arrays["observations_y"]
	lambdas_arr = arrays["lambdas"]
	gdf_n_arr = arrays["gdf_n"]
	ssrs_arr = arrays["ssrs"]
	sigmas_arr = arrays["sigmas"]
	sigma_mads_arr = arrays.get("sigma_mads")
	if sigma_mads_arr is None:
		raise ValueError(
			f"Test dataset '{suite_name}' is missing the required 'sigma_mads' tensor."
		)
	npix_arr = arrays["n_pixels"]
	gdf_n_se_arr = arrays.get("gdf_n_se")
	if gdf_n_se_arr is None:
		raise ValueError(f"Test dataset '{suite_name}' is missing the required 'gdf_n_se' tensor.")

	regime_records: list[dict[str, Any]] = []
	selection_cases: list[dict[str, Any]] = []
	selection_verification = verify_folded_limits()

	for idx in range(n_regimes):
		meta_info = regimes_metadata[idx] if idx < len(regimes_metadata) else {}
		title = meta_info.get("title", f"Regime {idx + 1}")
		modality = meta_info.get("img_tag", f"Modality {idx + 1}")

		lams = lambdas_arr[idx]
		true_r = gdf_n_arr[idx]
		n_pix = int(npix_arr[idx])
		sigma = float(sigmas_arr[idx])

		y_sample = obs_y_arr[idx]
		if y_sample.ndim == 3:
			y_sample = y_sample.unsqueeze(0)

		n_nodes = len(lams)
		y_batch = y_sample.expand(n_nodes, 1, y_sample.shape[-2], y_sample.shape[-1]).to(device)
		lams_device = lams.to(device)

		feats = extract_features(y_batch, lams_device)
		pred_r = predict(model, y_batch, lams_device, feats=feats)
		release_device_cache()

		reg_m = compute_regression_metrics(true_r, pred_r)
		r2 = reg_m["r2"]
		mae_pct = reg_m["mae"] * 100

		gdf_se = gdf_n_se_arr[idx]

		mse_meas = ssrs_arr[idx] / n_pix

		sure_surr = compute_sure(mse_meas, pred_r, sigma)
		sure_mc = compute_sure(mse_meas, true_r, sigma)
		d_sure_mc = compute_sure_uncertainty(sigma, gdf_se)
		lam_s = float(lams[int(torch.argmin(sure_surr))])
		lam_ref = float(lams[int(torch.argmin(sure_mc))])

		noise_estimate = estimate_noise_wavelet_mad(y_sample)
		sigma_mad = float(sigma_mads_arr[idx])
		recomputed_sigma_mad = float(torch.tensor(noise_estimate.sigma, dtype=torch.float32))
		if recomputed_sigma_mad != sigma_mad:
			raise ValueError(
				f"Stored sigma_mads[{idx}]={sigma_mad!r} does not match the recomputed "
				f"sigma_MAD {recomputed_sigma_mad!r}."
			)
		n_mad_coefficients = noise_estimate.n_coefficients
		if n_mad_coefficients != DEFAULT_WAVELET_MAD_N_COEFFICIENTS:
			raise ValueError(
				f"Expected {DEFAULT_WAVELET_MAD_N_COEFFICIENTS} MAD coefficients; "
				f"got {n_mad_coefficients}."
			)
		sure_mad_surr = compute_sure(mse_meas, pred_r, sigma_mad)
		sure_mad_mc = compute_sure(mse_meas, true_r, sigma_mad)
		d_sure_mad_mc = compute_sure_mad_uncertainty(
			sigma_mad, true_r, gdf_se=gdf_se, n_coefficients=n_mad_coefficients
		)
		lam_s_mad = float(lams[int(torch.argmin(sure_mad_surr))])
		lam_ref_mad = float(lams[int(torch.argmin(sure_mad_mc))])

		morozov_ratio_mc = compute_mdp(mse_meas, true_r, sigma)
		morozov_ratio_surr = compute_mdp(mse_meas, pred_r, sigma)
		d_morozov_mc = compute_mdp_known_sigma_uncertainty(morozov_ratio_mc, true_r, gdf_se)
		lam_s_morozov = float(lams[int(torch.argmin(torch.abs(morozov_ratio_surr - 1.0)))])
		lam_ref_morozov = float(lams[int(torch.argmin(torch.abs(morozov_ratio_mc - 1.0)))])

		morozov_ratio_mad_mc = compute_mdp(mse_meas, true_r, sigma_mad)
		morozov_ratio_mad_surr = compute_mdp(mse_meas, pred_r, sigma_mad)
		d_morozov_mad_mc = compute_mdp_uncertainty(
			morozov_ratio_mad_mc,
			true_r,
			gdf_se=gdf_se,
			sigma_mad=sigma_mad,
			n_coefficients=n_mad_coefficients,
		)
		lam_s_morozov_mad = float(lams[int(torch.argmin(torch.abs(morozov_ratio_mad_surr - 1.0)))])
		lam_ref_morozov_mad = float(lams[int(torch.argmin(torch.abs(morozov_ratio_mad_mc - 1.0)))])

		anchor_r = feats[:, 1].cpu()

		case_records = _selection_cases(
			regime_index=idx,
			title=title,
			n_pixels=n_pix,
			lambdas=lams,
			gdf_n=true_r,
			gdf_se=gdf_se,
			sigma_mad=sigma_mad,
			n_coefficients=n_mad_coefficients,
			sure_mad_mc=sure_mad_mc,
			sure_mad_surr=sure_mad_surr,
			ratio_mc=morozov_ratio_mad_mc,
			ratio_surr=morozov_ratio_mad_surr,
		)
		selection_cases.extend(case_records)

		regime_records.append(
			{
				"regime_index": idx,
				"anchor_r": anchor_r,
				"title": title,
				"modality": modality,
				"sigma": sigma,
				"n_pixels": n_pix,
				"lambdas": lams,
				"true_r": true_r,
				"pred_r": pred_r,
				"sure_surr": sure_surr,
				"sure_mc": sure_mc,
				"delta_sure_mc": d_sure_mc,
				"lambda_star_surr": lam_s,
				"lambda_star_ref": lam_ref,
				"gdf_se": gdf_se,
				"r2": r2,
				"mae_pct": mae_pct,
				"sigma_mad": sigma_mad,
				"sure_mad_surr": sure_mad_surr,
				"sure_mad_mc": sure_mad_mc,
				"delta_sure_mad_mc": d_sure_mad_mc,
				"lambda_star_mad_surr": lam_s_mad,
				"lambda_star_mad_ref": lam_ref_mad,
				"morozov_ratio_mc": morozov_ratio_mc,
				"morozov_ratio_surr": morozov_ratio_surr,
				"delta_morozov_mc": d_morozov_mc,
				"lambda_star_morozov_surr": lam_s_morozov,
				"lambda_star_morozov_ref": lam_ref_morozov,
				"morozov_ratio_mad_mc": morozov_ratio_mad_mc,
				"morozov_ratio_mad_surr": morozov_ratio_mad_surr,
				"delta_morozov_mad_mc": d_morozov_mad_mc,
				"lambda_star_morozov_mad_surr": lam_s_morozov_mad,
				"lambda_star_morozov_mad_ref": lam_ref_morozov_mad,
				"selection_excess": {case["criterion"]: case for case in case_records},
			}
		)

	summary_metrics = {
		"mean_r2": statistics.fmean([record["r2"] for record in regime_records]),
		"mean_mae_pct": statistics.fmean([record["mae_pct"] for record in regime_records]),
		"n_regimes": n_regimes,
	}

	selection_summaries = _summarize_selection_cases(selection_cases)
	payload = {
		"suite_name": suite_name,
		"model": model_dir.name,
		"evaluated_at": datetime.now(UTC).isoformat(),
		"summary_metrics": summary_metrics,
		"regimes": [
			{
				"regime_index": record["regime_index"],
				"title": record["title"],
				"modality": record["modality"],
				"sigma": record["sigma"],
				"sigma_mad": record["sigma_mad"],
				"r2": record["r2"],
				"mae_pct": record["mae_pct"],
				"n_pixels": record["n_pixels"],
				"lambdas": record["lambdas"],
				"mc_gdf": record["true_r"] * record["n_pixels"],
				"mc_gdf_norm": record["true_r"],
				"mc_gdf_norm_err": record["gdf_se"],
				"surrogate_gdf_norm": record["pred_r"],
				"anchor_gdf_norm": record["anchor_r"],
				"mc_sure": record["sure_mc"],
				"mc_delta_sure": record["delta_sure_mc"],
				"surrogate_sure": record["sure_surr"],
				"mc_sure_mad": record["sure_mad_mc"],
				"mc_delta_sure_mad": record["delta_sure_mad_mc"],
				"surrogate_sure_mad": record["sure_mad_surr"],
				"mc_morozov_ratio": record["morozov_ratio_mc"],
				"surrogate_morozov_ratio": record["morozov_ratio_surr"],
				"mc_delta_morozov": record["delta_morozov_mc"],
				"mc_morozov_ratio_mad": record["morozov_ratio_mad_mc"],
				"surrogate_morozov_ratio_mad": record["morozov_ratio_mad_surr"],
				"mc_delta_morozov_mad": record["delta_morozov_mad_mc"],
				"selected_lambda": {
					"mc_sure": record["lambda_star_ref"],
					"surrogate_sure": record["lambda_star_surr"],
					"mc_sure_mad": record["lambda_star_mad_ref"],
					"surrogate_sure_mad": record["lambda_star_mad_surr"],
					"mc_morozov": record["lambda_star_morozov_ref"],
					"surrogate_morozov": record["lambda_star_morozov_surr"],
					"mc_morozov_mad": record["lambda_star_morozov_mad_ref"],
					"surrogate_morozov_mad": record["lambda_star_morozov_mad_surr"],
				},
				"selection_excess": record["selection_excess"],
			}
			for record in regime_records
		],
		"split": "test",
		"test_data": display_path(test_data),
		"source_datasets": [{"dataset_name": suite_name, "test_data": display_path(test_data)}],
		"selection_excess": {
			"statistic": "z = D / SE[D], reference-subtracted criterion excess in standard errors",
			"model": "joint Gaussian ratio errors; folded absolute deviations; shared noise estimate",
			"conditioning": "fixed observation and selected candidates; no reoptimization",
			"mad_variance_factor": WAVELET_MAD_VARIANCE_FACTOR,
			"quadrature": {
				"absolute_tolerance": SELECTION_QUADRATURE_ABS_TOL,
				"relative_tolerance": SELECTION_QUADRATURE_REL_TOL,
				"subdivision_limit": SELECTION_QUADRATURE_LIMIT,
				"refinement_factor": SELECTION_QUADRATURE_REFINEMENT_FACTOR,
			},
			"verification_absolute_errors": selection_verification,
			"max_refinement_relative_excess_se_change": max(
				case["refinement_relative_excess_se_change"] for case in selection_cases
			),
			"summaries": selection_summaries,
			"by_resolution": {
				str(size): {
					summary["criterion"]: summary
					for summary in selection_summaries
					if summary["size"] == size
				}
				for size in sorted({summary["size"] for summary in selection_summaries})
			},
			"cases": selection_cases,
		},
	}
	report_path = model_dir / f"{suite_name}.pt"
	save_pt(payload, report_path)
	print(
		f"Completed {suite_name}: model={model_dir.name} | "
		f"mean R2={summary_metrics['mean_r2']:.4f} | "
		f"mean MAE={summary_metrics['mean_mae_pct']:.2f}%"
	)
	return payload, report_path


__all__ = ["evaluate_sweeps"]
