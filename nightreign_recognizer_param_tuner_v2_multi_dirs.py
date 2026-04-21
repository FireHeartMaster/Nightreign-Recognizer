#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class EvalCase:
    layout_number: int
    image_path: Path
    source_tag: str
    augment_index: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark and random-search tuner for the Nightreign prototype recognizer, using one or more image directories."
    )
    p.add_argument("--backend", type=Path, required=True, help="Path to the prototype recognizer backend script.")
    p.add_argument("--package", type=Path, required=True, help="Path to the .npz recognition package.")
    p.add_argument(
        "--images-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing numbered reference images. Repeat this flag to include multiple directories.",
    )
    p.add_argument("--search-space-json", type=Path, required=False, help="JSON file describing the parameter search space.")
    p.add_argument("--trials", type=int, default=25, help="Number of random-search trials, excluding the baseline.")
    p.add_argument("--seed", type=int, default=12345, help="Random seed.")
    p.add_argument("--augments-per-image", type=int, default=0, help="How many light augmented copies to evaluate per image.")
    p.add_argument("--max-images-per-dir", type=int, default=0, help="Optional cap per image directory. 0 means all.")
    p.add_argument("--top-k", type=int, default=5, help="Largest top-k metric to compute.")
    p.add_argument("--out-json", type=Path, default=Path("nightreign_tuning_results_multi.json"), help="Where to save the full tuning results JSON.")
    p.add_argument("--out-csv", type=Path, default=Path("nightreign_tuning_trials_multi.csv"), help="Where to save the per-trial CSV summary.")
    p.add_argument("--best-config-out", type=Path, default=Path("nightreign_best_tuned_config_multi.json"), help="Where to save the best parameter config JSON.")
    return p.parse_args()


def load_backend_module(path: Path):
    spec = importlib.util.spec_from_file_location("nightreign_tuning_backend_multi", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_reference_images(images_dirs: Sequence[Path], max_images_per_dir: int) -> List[EvalCase]:
    cases: List[EvalCase] = []
    for directory in images_dirs:
        dir_cases: List[EvalCase] = []
        source_tag = directory.name
        for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            m = re.search(r"(\d+)$", path.stem)
            if not m:
                continue
            layout_number = int(m.group(1))
            dir_cases.append(EvalCase(layout_number=layout_number, image_path=path, source_tag=source_tag, augment_index=0))
        if not dir_cases:
            raise FileNotFoundError(f"No numbered reference images found in {directory}")
        if max_images_per_dir > 0:
            dir_cases = dir_cases[:max_images_per_dir]
        cases.extend(dir_cases)
    return cases


def make_augmented_cases(base_cases: Sequence[EvalCase], augments_per_image: int) -> List[EvalCase]:
    if augments_per_image <= 0:
        return list(base_cases)
    out = list(base_cases)
    for case in base_cases:
        for aug_idx in range(1, augments_per_image + 1):
            out.append(EvalCase(layout_number=case.layout_number, image_path=case.image_path, source_tag=case.source_tag, augment_index=aug_idx))
    return out


def deterministic_rng(seed: int, layout_number: int, augment_index: int, source_tag: str) -> random.Random:
    source_hash = abs(hash(source_tag)) % (10**9)
    return random.Random((seed * 1000003) ^ (layout_number * 9973) ^ (augment_index * 7919) ^ source_hash)


def augment_panel_image(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    scale = rng.uniform(0.94, 1.06)
    new_w = max(64, int(round(w * scale)))
    new_h = max(64, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(out, (new_w, new_h), interpolation=interp)

    canvas = np.zeros_like(out)
    if scale >= 1.0:
        x0 = max(0, (new_w - w) // 2)
        y0 = max(0, (new_h - h) // 2)
        canvas = scaled[y0:y0 + h, x0:x0 + w].copy()
    else:
        x0 = max(0, (w - new_w) // 2 + rng.randint(-6, 6))
        y0 = max(0, (h - new_h) // 2 + rng.randint(-6, 6))
        x0 = max(0, min(w - new_w, x0))
        y0 = max(0, min(h - new_h, y0))
        canvas[y0:y0 + new_h, x0:x0 + new_w] = scaled
        mask = (canvas.sum(axis=2) == 0).astype(np.uint8) * 255
        if int(mask.sum()) > 0:
            blur = cv2.GaussianBlur(canvas, (0, 0), 7.0)
            canvas[mask > 0] = blur[mask > 0]
    out = canvas

    alpha = rng.uniform(0.92, 1.10)
    beta = rng.uniform(-14.0, 14.0)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.55:
        sigma = rng.uniform(0.4, 1.2)
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    else:
        blur = cv2.GaussianBlur(out, (0, 0), 0.8)
        out = cv2.addWeighted(out, 1.22, blur, -0.22, 0)

    if rng.random() < 0.7:
        noise_std = rng.uniform(1.0, 5.0)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, noise_std, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.65:
        quality = int(round(rng.uniform(72, 96)))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec

    return out


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


def install_earth_classifier(module, low_weight: Optional[float], grad_weight: Optional[float]) -> None:
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


def load_package_metadata(package_path: Path) -> Dict[int, Dict[str, Any]]:
    with np.load(package_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    layouts = metadata["layouts"]
    return {int(item["layout_number"]): item for item in layouts}


def pick_from_space(space_value: Any, rng: random.Random) -> Any:
    if isinstance(space_value, list):
        if not space_value:
            raise ValueError("Search-space list cannot be empty.")
        return copy.deepcopy(rng.choice(space_value))
    return copy.deepcopy(space_value)


def sample_config_from_space(space: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key, value in space.items():
        if key == "type_overrides":
            type_cfg: Dict[str, Any] = {}
            for type_name, overrides in value.items():
                type_cfg[type_name] = {}
                for o_key, o_val in overrides.items():
                    type_cfg[type_name][o_key] = pick_from_space(o_val, rng)
            cfg[key] = type_cfg
        else:
            cfg[key] = pick_from_space(value, rng)
    return cfg


def find_rank(results: Sequence[Dict[str, Any]], layout_number: int) -> Optional[int]:
    for idx, item in enumerate(results, start=1):
        if int(item["layout_number"]) == int(layout_number):
            return idx
    return None


def slot_metrics_for_case(payload: Dict[str, Any], gt_layout: Dict[str, Any]) -> Dict[str, float]:
    out = {
        "slot_evaluated": 0.0,
        "slot_correct": 0.0,
        "slot_unknown": 0.0,
        "slot_wrong_label": 0.0,
        "slot_false_empty": 0.0,
        "slot_false_positive_empty": 0.0,
    }
    candidate_earths = payload.get("candidate_earths", [])
    if not candidate_earths:
        return out
    truth_earth = str(gt_layout["shifting_earth"])
    if truth_earth not in payload.get("observations_by_earth", {}):
        return out
    obs = payload["observations_by_earth"][truth_earth]["slots"]
    exp = gt_layout["slot_labels"]
    if len(obs) != len(exp):
        return out
    for pred_slot, exp_label in zip(obs, exp):
        out["slot_evaluated"] += 1.0
        pred_label = str(pred_slot["label"])
        exp_label = str(exp_label)
        if pred_label == exp_label:
            out["slot_correct"] += 1.0
        elif pred_label == "unknown":
            out["slot_unknown"] += 1.0
        elif pred_label == "empty" and exp_label != "empty":
            out["slot_false_empty"] += 1.0
        elif pred_label != "empty" and exp_label == "empty":
            out["slot_false_positive_empty"] += 1.0
        else:
            out["slot_wrong_label"] += 1.0
    return out


def compute_objective(metrics: Dict[str, float]) -> float:
    return (
        100.0 * metrics["layout_top1_acc"]
        + 40.0 * metrics["layout_top3_acc"]
        + 12.0 * metrics["layout_top5_acc"]
        + 20.0 * metrics["slot_correct_rate"]
        - 9.0 * metrics["slot_unknown_rate"]
        - 25.0 * metrics["slot_wrong_label_rate"]
        - 18.0 * metrics["slot_false_positive_empty_rate"]
        - 12.0 * metrics["slot_false_empty_rate"]
        + 8.0 * metrics["earth_top1_acc"]
        + 15.0 * metrics["mrr"]
    )


def evaluate_config(
    module,
    package_path: Path,
    cases: Sequence[EvalCase],
    gt_layouts: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    temp_dir: Path,
    base_state: Dict[str, Any],
    seed: int,
    top_k: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    apply_config_to_module(module, base_state, config)

    totals = {
        "n": 0.0,
        "earth_top1_correct": 0.0,
        "layout_top1_correct": 0.0,
        "layout_top3_correct": 0.0,
        "layout_top5_correct": 0.0,
        "mrr_sum": 0.0,
        "rank_sum": 0.0,
        "rank_count": 0.0,
        "slot_evaluated": 0.0,
        "slot_correct": 0.0,
        "slot_unknown": 0.0,
        "slot_wrong_label": 0.0,
        "slot_false_empty": 0.0,
        "slot_false_positive_empty": 0.0,
    }
    case_rows: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        gt_layout = gt_layouts.get(case.layout_number)
        if gt_layout is None:
            continue
        img = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if case.augment_index > 0:
            rng = deterministic_rng(seed, case.layout_number, case.augment_index, case.source_tag)
            img = augment_panel_image(img, rng)

        tmp_img_path = temp_dir / f"eval_{idx:05d}_{case.source_tag}_layout_{case.layout_number}_aug_{case.augment_index}.png"
        cv2.imwrite(str(tmp_img_path), img)

        ns = argparse.Namespace(
            package=package_path,
            image=None,
            panel_image=tmp_img_path,
            monitor=None,
            top=max(20, top_k),
            debug=False,
            crop_out=temp_dir / "unused_crop.png",
            detections_out=temp_dir / "unused_det.png",
            detections_search_out=temp_dir / "unused_det_search.png",
            search_radius_scale=float(config.get("search_radius_scale", 1.0)),
            json_out=None,
        )
        payload = module.run_recognition(ns)
        truth_earth = str(gt_layout["shifting_earth"])
        earth_ranking = payload.get("earth_ranking", [])
        pred_earth = earth_ranking[0]["name"] if earth_ranking else None
        results = payload.get("results", [])
        true_rank = find_rank(results, case.layout_number)

        totals["n"] += 1.0
        if pred_earth == truth_earth:
            totals["earth_top1_correct"] += 1.0
        if true_rank is not None:
            totals["mrr_sum"] += 1.0 / float(true_rank)
            totals["rank_sum"] += float(true_rank)
            totals["rank_count"] += 1.0
            if true_rank == 1:
                totals["layout_top1_correct"] += 1.0
            if true_rank <= 3:
                totals["layout_top3_correct"] += 1.0
            if true_rank <= 5:
                totals["layout_top5_correct"] += 1.0

        slot_m = slot_metrics_for_case(payload, gt_layout)
        for k in ("slot_evaluated", "slot_correct", "slot_unknown", "slot_wrong_label", "slot_false_empty", "slot_false_positive_empty"):
            totals[k] += slot_m[k]

        case_rows.append({
            "layout_number": case.layout_number,
            "source_tag": case.source_tag,
            "augment_index": case.augment_index,
            "truth_earth": truth_earth,
            "pred_earth": pred_earth,
            "true_rank": true_rank,
            "top1_layout": results[0]["layout_number"] if results else None,
            "slot_correct": slot_m["slot_correct"],
            "slot_unknown": slot_m["slot_unknown"],
            "slot_wrong_label": slot_m["slot_wrong_label"],
            "slot_false_empty": slot_m["slot_false_empty"],
            "slot_false_positive_empty": slot_m["slot_false_positive_empty"],
        })

    n = max(1.0, totals["n"])
    slot_n = max(1.0, totals["slot_evaluated"])
    metrics = {
        "cases": totals["n"],
        "earth_top1_acc": totals["earth_top1_correct"] / n,
        "layout_top1_acc": totals["layout_top1_correct"] / n,
        "layout_top3_acc": totals["layout_top3_correct"] / n,
        "layout_top5_acc": totals["layout_top5_correct"] / n,
        "mrr": totals["mrr_sum"] / n,
        "avg_true_rank": totals["rank_sum"] / max(1.0, totals["rank_count"]),
        "slot_correct_rate": totals["slot_correct"] / slot_n,
        "slot_unknown_rate": totals["slot_unknown"] / slot_n,
        "slot_wrong_label_rate": totals["slot_wrong_label"] / slot_n,
        "slot_false_empty_rate": totals["slot_false_empty"] / slot_n,
        "slot_false_positive_empty_rate": totals["slot_false_positive_empty"] / slot_n,
    }
    metrics["objective"] = compute_objective(metrics)
    return metrics, case_rows


def write_trial_csv(path: Path, trials: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    rows = []
    for item in trials:
        row = {
            "trial_index": item["trial_index"],
            "label": item["label"],
            "objective": item["metrics"]["objective"],
            "earth_top1_acc": item["metrics"]["earth_top1_acc"],
            "layout_top1_acc": item["metrics"]["layout_top1_acc"],
            "layout_top3_acc": item["metrics"]["layout_top3_acc"],
            "layout_top5_acc": item["metrics"]["layout_top5_acc"],
            "mrr": item["metrics"]["mrr"],
            "avg_true_rank": item["metrics"]["avg_true_rank"],
            "slot_correct_rate": item["metrics"]["slot_correct_rate"],
            "slot_unknown_rate": item["metrics"]["slot_unknown_rate"],
            "slot_wrong_label_rate": item["metrics"]["slot_wrong_label_rate"],
            "slot_false_empty_rate": item["metrics"]["slot_false_empty_rate"],
            "slot_false_positive_empty_rate": item["metrics"]["slot_false_positive_empty_rate"],
            "config_json": json.dumps(item["config"], sort_keys=True),
        }
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "trial_index", "label", "objective", "earth_top1_acc", "layout_top1_acc",
            "layout_top3_acc", "layout_top5_acc", "mrr", "avg_true_rank",
            "slot_correct_rate", "slot_unknown_rate", "slot_wrong_label_rate",
            "slot_false_empty_rate", "slot_false_positive_empty_rate", "config_json"
        ])
        writer.writeheader()
        writer.writerows(rows)


def default_search_space() -> Dict[str, Any]:
    return {
        "search_radius_scale": [0.70, 0.80, 0.90, 1.00, 1.10],
        "earth_low_weight": [0.55, 0.65, 0.75],
        "earth_grad_weight": [0.45, 0.35, 0.25],
        "crop_refine_margin_weight": [0.25, 0.35, 0.45],
        "crop_refine_black_weight": [0.08, 0.12, 0.18],
        "crop_refine_edge_weight": [0.02, 0.03, 0.05],
        "earth_secondary_threshold": [0.015, 0.025, 0.035],
        "template_scales": [
            [0.60, 0.72, 0.86, 1.00, 1.16, 1.34, 1.54],
            [0.66, 0.80, 0.92, 1.00, 1.10, 1.24, 1.40],
            [0.72, 0.86, 1.00, 1.16, 1.34],
        ],
        "type_overrides": {
            "Church": {
                "accept": [0.36, 0.40, 0.44],
                "margin": [0.018, 0.025, 0.032],
                "empty_threshold": [0.24, 0.28, 0.32],
                "edge_weight": [0.56, 0.62, 0.68]
            },
            "Sorcerer's Rise": {
                "accept": [0.32, 0.36, 0.40],
                "margin": [0.015, 0.020, 0.028],
                "empty_threshold": [0.22, 0.26, 0.30],
                "edge_weight": [0.58, 0.64, 0.70]
            },
            "Fort": {
                "accept": [0.39, 0.43, 0.47],
                "margin": [0.022, 0.030, 0.038],
                "empty_threshold": [0.26, 0.30, 0.34]
            },
            "Great Church": {
                "accept": [0.39, 0.43, 0.47],
                "margin": [0.022, 0.030, 0.038],
                "empty_threshold": [0.26, 0.30, 0.34]
            },
            "Main Encampment": {
                "accept": [0.36, 0.40, 0.44],
                "margin": [0.020, 0.028, 0.036],
                "empty_threshold": [0.25, 0.29, 0.33]
            },
            "Ruins": {
                "accept": [0.38, 0.42, 0.46],
                "margin": [0.020, 0.028, 0.036],
                "empty_threshold": [0.26, 0.30, 0.34]
            },
        },
    }


def summarize_source_breakdown(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # best trial per-source summary using case rows
    if not trials:
        return {}
    best = max(trials, key=lambda t: t["metrics"]["objective"])
    by_source: Dict[str, Dict[str, float]] = {}
    rows = best.get("case_rows", [])
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["source_tag"], []).append(row)
    for source, items in grouped.items():
        n = max(1, len(items))
        top1 = sum(1 for r in items if r.get("true_rank") == 1) / n
        top3 = sum(1 for r in items if r.get("true_rank") is not None and r["true_rank"] <= 3) / n
        mrr = sum((1.0 / r["true_rank"]) for r in items if r.get("true_rank")) / n
        by_source[source] = {
            "cases": len(items),
            "layout_top1_acc": top1,
            "layout_top3_acc": top3,
            "mrr": mrr,
        }
    return by_source


def main() -> int:
    args = parse_args()
    backend = load_backend_module(args.backend)
    base_state = capture_base_state(backend)
    gt_layouts = load_package_metadata(args.package)
    base_cases = list_reference_images(args.images_dir, args.max_images_per_dir)
    cases = make_augmented_cases(base_cases, args.augments_per_image)

    if args.search_space_json is not None:
        search_space = json.loads(args.search_space_json.read_text(encoding="utf-8"))
    else:
        search_space = default_search_space()

    rng = random.Random(args.seed)
    trial_summaries: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="nightreign_tune_multi_") as td:
        temp_dir = Path(td)

        baseline_cfg: Dict[str, Any] = {"search_radius_scale": 1.0}
        baseline_metrics, baseline_rows = evaluate_config(
            backend, args.package, cases, gt_layouts, baseline_cfg, temp_dir, base_state, args.seed, args.top_k
        )
        trial_summaries.append({
            "trial_index": 0,
            "label": "baseline",
            "config": baseline_cfg,
            "metrics": baseline_metrics,
            "case_rows": baseline_rows,
        })
        print(
            f"[baseline] objective={baseline_metrics['objective']:.3f} "
            f"layout_top1={baseline_metrics['layout_top1_acc']:.3f} "
            f"slot_correct={baseline_metrics['slot_correct_rate']:.3f} "
            f"slot_unknown={baseline_metrics['slot_unknown_rate']:.3f} "
            f"slot_wrong={baseline_metrics['slot_wrong_label_rate']:.3f}"
        )

        for trial_idx in range(1, args.trials + 1):
            cfg = sample_config_from_space(search_space, rng)
            metrics, case_rows = evaluate_config(
                backend, args.package, cases, gt_layouts, cfg, temp_dir, base_state, args.seed, args.top_k
            )
            trial_summaries.append({
                "trial_index": trial_idx,
                "label": "random_search",
                "config": cfg,
                "metrics": metrics,
                "case_rows": case_rows,
            })
            print(
                f"[trial {trial_idx:03d}] objective={metrics['objective']:.3f} "
                f"layout_top1={metrics['layout_top1_acc']:.3f} "
                f"layout_top3={metrics['layout_top3_acc']:.3f} "
                f"slot_correct={metrics['slot_correct_rate']:.3f} "
                f"slot_unknown={metrics['slot_unknown_rate']:.3f} "
                f"slot_wrong={metrics['slot_wrong_label_rate']:.3f}"
            )

    trial_summaries.sort(key=lambda item: item["metrics"]["objective"], reverse=True)
    best = trial_summaries[0]
    source_breakdown = summarize_source_breakdown(trial_summaries)

    ensure_parent(args.out_json)
    args.out_json.write_text(json.dumps({
        "backend": str(args.backend),
        "package": str(args.package),
        "images_dirs": [str(p) for p in args.images_dir],
        "seed": args.seed,
        "augments_per_image": args.augments_per_image,
        "top_k": args.top_k,
        "best_trial": best,
        "best_source_breakdown": source_breakdown,
        "trials": trial_summaries,
    }, indent=2), encoding="utf-8")

    write_trial_csv(args.out_csv, trial_summaries)

    ensure_parent(args.best_config_out)
    args.best_config_out.write_text(json.dumps(best["config"], indent=2), encoding="utf-8")

    print("\nBest trial:")
    print(json.dumps({
        "trial_index": best["trial_index"],
        "label": best["label"],
        "metrics": best["metrics"],
        "config": best["config"],
        "best_source_breakdown": source_breakdown,
    }, indent=2))
    print(f"\nSaved results JSON to: {args.out_json}")
    print(f"Saved trial CSV to:    {args.out_csv}")
    print(f"Saved best config to:  {args.best_config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
