"""Conservative, deterministic pixel QC for native RGB histology crops.

This is a technical empty-field filter, not a diagnostic/artifact classifier.
The original RGB image is never modified. Lumina are not segmented or rejected
as objects: a crop with sufficient stained tissue remains eligible.

Stained tissue = (HSV saturation >= 0.05) AND (mean channel OD >= 0.05),
where saturation = (max(R,G,B)-min(R,G,B))/max(R,G,B), with black assigned 0,
and OD = -ln((channel+1)/256). No smoothing, filling, morphology, or component
filtering is performed. Thresholds are inclusive. Tissue acceptance is >=20%.

Flat images and white fields are rejected independently of tissue fraction.
Blur, weak staining, and dense/dark staining are only heuristic review flags;
absence of flags does not establish that an image is artifact-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_QC_CONFIG = {
    "algorithm_version": "rgb-stained-tissue-v1",
    "mask_formula": "HSV_S >= saturation_min AND mean(-ln((RGB+1)/256)) >= od_min",
    "saturation_min": 0.05,
    "od_min": 0.05,
    "min_tissue_fraction": 0.20,
    "near_uniform_max_channel_std": 1.0,
    "empty_white_all_channels_min": 250,
    "artifact_flag_min_tissue_fraction": 0.20,
    "blur_laplacian_variance_below": 30.0,
    "weak_stain_mean_tissue_od_below": 0.12,
    "weak_stain_mean_tissue_saturation_below": 0.12,
    "dark_gray_below": 80.0,
    "dark_od_min": 0.90,
    "dark_fraction_flag_above": 0.15,
    "artifact_flags_are_rejections": False,
    "absence_of_flags_is_artifact_clearance": False,
    "mask_morphology": "none",
}


def resolved_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the complete algorithm contract, validating all supplied keys."""
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - set(DEFAULT_QC_CONFIG)
    if unknown:
        raise ValueError(f"Unknown QC configuration keys: {sorted(unknown)}")
    cfg = {**DEFAULT_QC_CONFIG, **supplied}
    for name in ("algorithm_version", "mask_formula", "mask_morphology",
                 "artifact_flags_are_rejections", "absence_of_flags_is_artifact_clearance"):
        if cfg[name] != DEFAULT_QC_CONFIG[name]:
            raise ValueError(f"Unsupported QC algorithm contract: {name}={cfg[name]!r}")
    for name in ("saturation_min", "min_tissue_fraction", "artifact_flag_min_tissue_fraction",
                 "weak_stain_mean_tissue_saturation_below", "dark_fraction_flag_above"):
        if not np.isfinite(cfg[name]) or not 0 <= cfg[name] <= 1:
            raise ValueError(f"{name} must be finite and in [0,1]")
    for name in ("od_min", "near_uniform_max_channel_std", "blur_laplacian_variance_below",
                 "weak_stain_mean_tissue_od_below", "dark_od_min"):
        if not np.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("empty_white_all_channels_min", "dark_gray_below"):
        if not np.isfinite(cfg[name]) or not 0 <= cfg[name] <= 255:
            raise ValueError(f"{name} must be finite and in [0,255]")
    return cfg


def _rgb_array(image_or_array: Any) -> np.ndarray:
    arr = np.asarray(image_or_array)
    if arr.ndim != 3 or arr.shape[2] != 3 or min(arr.shape[:2]) < 3:
        raise ValueError(f"Expected HxWx3 RGB image with both dimensions >=3; got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected RGB uint8 pixels; got {arr.dtype}")
    return arr


def _mask_components(arr: np.ndarray, cfg: dict[str, Any]):
    rgb = arr.astype(np.float64)
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    saturation = np.divide(high-low, high, out=np.zeros_like(high), where=high > 0)
    od = -np.log((rgb+1.0)/256.0).mean(axis=2)
    tissue = (saturation >= cfg["saturation_min"]) & (od >= cfg["od_min"])
    return rgb, saturation, od, tissue


def stained_tissue_mask(image_or_array: Any, config: dict[str, Any] | None = None) -> np.ndarray:
    """Boolean mask only for technical estimates; never alter exported pixels."""
    arr = _rgb_array(image_or_array)
    return _mask_components(arr, resolved_config(config))[3]


def analyze_rgb(image_or_array: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return JSON-serializable QC. Decode errors are handled by the caller.

    The finite 4-neighbour Laplacian is evaluated on interior tissue pixels
    without padding. Its variance and thresholds use raw uint8 intensity units.
    Near-uniform means *every* spatial channel standard deviation is below 1.
    Artifact flags apply only when enough tissue is present. These heuristics
    need visual review and do not replace pathology or artifact assessment.
    """
    cfg = resolved_config(config)
    arr = _rgb_array(image_or_array)
    rgb, saturation, od, tissue = _mask_components(arr, cfg)
    count = int(tissue.sum())
    fraction = count / int(tissue.size)
    std = rgb.std(axis=(0, 1))
    gray = rgb[..., 0]*0.299 + rgb[..., 1]*0.587 + rgb[..., 2]*0.114
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
           + gray[1:-1, 2:] - 4*gray[1:-1, 1:-1])
    interior = tissue[1:-1, 1:-1]
    lap_var = float(lap[interior].var()) if interior.any() else None
    mean_od = float(od[tissue].mean()) if count else None
    mean_saturation = float(saturation[tissue].mean()) if count else None
    dark = tissue & (gray < cfg["dark_gray_below"]) & (od >= cfg["dark_od_min"])
    dark_fraction = float(dark.mean())

    reasons = []
    if float(std.max()) < cfg["near_uniform_max_channel_std"]:
        reasons.append("uniform_or_near_uniform")
    if bool((arr >= cfg["empty_white_all_channels_min"]).all()):
        reasons.append("empty_white")
    if fraction < cfg["min_tissue_fraction"]:
        reasons.append("low_tissue_fraction")

    flags = []
    if fraction >= cfg["artifact_flag_min_tissue_fraction"]:
        if lap_var is not None and lap_var < cfg["blur_laplacian_variance_below"]:
            flags.append("suspected_blur_review")
        if (mean_od < cfg["weak_stain_mean_tissue_od_below"]
                or mean_saturation < cfg["weak_stain_mean_tissue_saturation_below"]):
            flags.append("weak_stain_review")
        if dark_fraction > cfg["dark_fraction_flag_above"]:
            flags.append("dark_dense_tissue_possible_fold_review")

    return {
        "tissue_fraction": fraction,
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "qc_flags": flags,
        "metrics": {
            "algorithm_version": cfg["algorithm_version"],
            "pixel_count": int(tissue.size),
            "tissue_pixel_count": count,
            "spatial_channel_std": [float(v) for v in std],
            "max_spatial_channel_std": float(std.max()),
            "rgb_min": int(arr.min()),
            "rgb_max": int(arr.max()),
            "mean_tissue_od": mean_od,
            "mean_tissue_saturation": mean_saturation,
            "tissue_laplacian_variance": lap_var,
            "dark_stained_fraction_of_patch": dark_fraction,
            "artifact_review_status": "heuristic_flags_only_not_validated_artifact_free",
        },
    }


if __name__ == "__main__":
    import argparse
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--write-defaults", type=Path)
    args = parser.parse_args()
    if args.write_defaults:
        args.write_defaults.write_text(json.dumps(DEFAULT_QC_CONFIG, indent=2)+"\n", encoding="utf-8")
    if args.image:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        with Image.open(args.image) as image:
            image.load()
            print(json.dumps(analyze_rgb(image, config), ensure_ascii=False, indent=2))
