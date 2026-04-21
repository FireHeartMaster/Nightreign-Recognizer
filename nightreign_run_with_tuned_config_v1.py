#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_backend_module(path: Path):
    spec = importlib.util.spec_from_file_location("nightreign_tuned_backend_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clone_type_config(cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in cfg.items()}


def capture_base_state(module) -> Dict[str, Any]:
    return {
        "TYPE_CONFIG": clone_type_config(module.TYPE_CONFIG),
        "DEFAULT_TEMPLATE_SCALES": tuple(module.DEFAULT_TEMPLATE_SCALES),
        "CROP_REFINE_MARGIN_WEIGHT": float(module.CROP_REFINE_MARGIN_WEIGHT),
        "CROP_REFINE_BLACK_WEIGHT": float(module.CROP_REFINE_BLACK_WEIGHT),
        "CROP_REFINE_EDGE_WEIGHT": float(module.CROP_REFINE_EDGE_WEIGHT),
        "EARTH_SECONDARY_THRESHOLD": float(module.EARTH_SECONDARY_THRESHOLD),
        "classify_shifting_earth": module.classify_shifting_earth,
    }


def restore_base_state(module, base: Dict[str, Any]) -> None:
    module.TYPE_CONFIG = clone_type_config(base["TYPE_CONFIG"])
    module.DEFAULT_TEMPLATE_SCALES = tuple(base["DEFAULT_TEMPLATE_SCALES"])
    module.CROP_REFINE_MARGIN_WEIGHT = float(base["CROP_REFINE_MARGIN_WEIGHT"])
    module.CROP_REFINE_BLACK_WEIGHT = float(base["CROP_REFINE_BLACK_WEIGHT"])
    module.CROP_REFINE_EDGE_WEIGHT = float(base["CROP_REFINE_EDGE_WEIGHT"])
    module.EARTH_SECONDARY_THRESHOLD = float(base["EARTH_SECONDARY_THRESHOLD"])
    module.classify_shifting_earth = base["classify_shifting_earth"]


def install_earth_classifier(module, low_weight, grad_weight) -> None:
    if low_weight is None and grad_weight is None:
        return
    lw = 0.65 if low_weight is None else float(low_weight)
    gw = 0.35 if grad_weight is None else float(grad_weight)
    total = abs(lw) + abs(gw)
    if total <= 1e-8:
        lw, gw = 0.65, 0.35
    else:
        lw, gw = lw / total, gw / total

    def classify_shifting_earth(panel_bgr, package):
        shot_low = module.preprocess_for_similarity(panel_bgr)
        shot_grad = module.preprocess_for_gradients(panel_bgr)
        shot_low_n = module.normalize(shot_low)
        shot_grad_n = module.normalize(shot_grad)
        ranked = []
        for idx, earth_name in enumerate(package.earth_names):
            low_corr = module.corr(shot_low_n, module.normalize(package.terrain_low[idx]))
            grad_corr = module.corr(shot_grad_n, module.normalize(package.terrain_grad[idx]))
            score = float(lw) * low_corr + float(gw) * grad_corr
            ranked.append((earth_name, score, {"low_corr": low_corr, "grad_corr": grad_corr}))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    module.classify_shifting_earth = classify_shifting_earth


def apply_config_to_module(module, base_state: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    restore_base_state(module, base_state)

    if "template_scales" in cfg and cfg["template_scales"]:
        module.DEFAULT_TEMPLATE_SCALES = tuple(float(x) for x in cfg["template_scales"])

    if "crop_refine_margin_weight" in cfg:
        module.CROP_REFINE_MARGIN_WEIGHT = float(cfg["crop_refine_margin_weight"])
    if "crop_refine_black_weight" in cfg:
        module.CROP_REFINE_BLACK_WEIGHT = float(cfg["crop_refine_black_weight"])
    if "crop_refine_edge_weight" in cfg:
        module.CROP_REFINE_EDGE_WEIGHT = float(cfg["crop_refine_edge_weight"])
    if "earth_secondary_threshold" in cfg:
        module.EARTH_SECONDARY_THRESHOLD = float(cfg["earth_secondary_threshold"])

    type_overrides = cfg.get("type_overrides", {})
    if type_overrides:
        for type_name, overrides in type_overrides.items():
            if type_name not in module.TYPE_CONFIG:
                continue
            for key, value in overrides.items():
                module.TYPE_CONFIG[type_name][key] = float(value)

    install_earth_classifier(
        module,
        cfg.get("earth_low_weight"),
        cfg.get("earth_grad_weight"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Nightreign recognizer with a tuned config JSON.")
    p.add_argument("--backend", type=Path, required=True, help="Path to the recognizer backend script.")
    p.add_argument("--config-json", type=Path, required=True, help="Path to the tuned config JSON from the tuner.")
    p.add_argument("--package", type=Path, required=True, help="Path to the .npz recognition package.")

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Full screenshot image file to process.")
    source.add_argument("--panel-image", type=Path, help="Already-cropped square map panel image to process.")
    source.add_argument("--monitor", type=int, help="Capture this monitor index with mss (1-based).")

    p.add_argument("--top", type=int, default=20, help="How many ranked layouts to print.")
    p.add_argument("--debug", action="store_true", help="Save debug images.")
    p.add_argument("--crop-out", type=Path, default=Path("recognition_panel_used.png"))
    p.add_argument("--detections-out", type=Path, default=Path("recognition_panel_used_with_slot_labels.png"))
    p.add_argument("--detections-search-out", type=Path, default=Path("recognition_panel_used_with_slot_labels_search.png"))
    p.add_argument("--json-out", type=Path, default=None, help="Optional path to save the recognizer JSON report.")
    p.add_argument("--search-radius-scale", type=float, default=None, help="Optional override. If omitted, use value from tuned config, or 1.0 if missing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    backend = load_backend_module(args.backend)
    base_state = capture_base_state(backend)
    cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    apply_config_to_module(backend, base_state, cfg)

    search_radius_scale = args.search_radius_scale
    if search_radius_scale is None:
        search_radius_scale = float(cfg.get("search_radius_scale", 1.0))

    ns = argparse.Namespace(
        package=args.package,
        image=args.image,
        panel_image=args.panel_image,
        monitor=args.monitor,
        top=args.top,
        debug=args.debug,
        crop_out=args.crop_out,
        detections_out=args.detections_out,
        detections_search_out=args.detections_search_out,
        search_radius_scale=search_radius_scale,
        json_out=args.json_out,
    )

    print("Loaded tuned config:")
    print(json.dumps(cfg, indent=2))
    print(f"\nUsing search_radius_scale={search_radius_scale:.3f}\n")

    backend.run_recognition(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
