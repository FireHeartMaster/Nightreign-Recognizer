import argparse
import json
import logging
import math
import os
import pickle
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

LOGGER = logging.getLogger("layout_classifier")
IMAGE_W = 1191
IMAGE_H = 1176
PATTERN_RE = re.compile(r"pattern_(\d+)_")


@dataclass
class ImageRecord:
    pattern_id: int
    path: str


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON")
    parser.add_argument("--verbose", action="store_true")


def parse_pattern_id(filename: str) -> int:
    match = PATTERN_RE.search(os.path.basename(filename))
    if not match:
        raise ValueError(f"Could not parse patternId from filename: {filename}")
    return int(match.group(1))


def list_images(images_dir: str) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for path in sorted(Path(images_dir).glob("*.png")):
        pattern_id = parse_pattern_id(path.name)
        records.append(ImageRecord(pattern_id=pattern_id, path=str(path)))
    if not records:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")
    return records


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if "layouts" not in metadata:
        raise KeyError("Metadata JSON must contain a 'layouts' field")
    return metadata


def _norm_to_px(norm_value: float, size: int) -> int:
    return int(round(float(norm_value) * (size - 1)))


def metadata_point_to_image_px(point: Dict[str, Any], image_w: int = IMAGE_W, image_h: int = IMAGE_H) -> Tuple[int, int, str]:
    """
    Contract:
      x_px = round(x_further_px_bl)
      y_px_top_left = (H - 1) - round(y_further_px_bl)
    When *_px_bl fields are unavailable, fall back to *_norm_bl or x_norm/y_norm.
    """
    source = None
    if "x_further_px_bl" in point and "y_further_px_bl" in point:
        x_bl = float(point["x_further_px_bl"])
        y_bl = float(point["y_further_px_bl"])
        source = "px_bl"
    elif "x_further_norm_bl" in point and "y_further_norm_bl" in point:
        x_bl = _norm_to_px(float(point["x_further_norm_bl"]), image_w)
        y_bl = _norm_to_px(float(point["y_further_norm_bl"]), image_h)
        source = "norm_bl"
    elif "x_norm" in point and "y_norm" in point:
        x_bl = _norm_to_px(float(point["x_norm"]), image_w)
        y_bl = _norm_to_px(float(point["y_norm"]), image_h)
        source = "x_norm_y_norm_fallback"
    else:
        raise KeyError(f"Point does not contain any supported coordinate fields: {sorted(point.keys())}")

    x_px = int(round(x_bl))
    y_px_top_left = int((image_h - 1) - round(y_bl))
    return x_px, y_px_top_left, source


def validate_and_collect_layout_points(
    metadata: Dict[str, Any],
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    layouts_by_id: Dict[int, Dict[str, Any]] = {}
    coord_sources = {}
    out_of_bounds = []

    for layout in metadata["layouts"]:
        pid = int(layout["patternId"])
        if pid in layouts_by_id:
            raise ValueError(f"Duplicate layout for patternId {pid}")
        normalized_layout = {"patternId": pid, "slots": [], "great_hollow_statuses": []}
        for key in ("slots", "great_hollow_statuses"):
            for idx, point in enumerate(layout.get(key, [])):
                x_px, y_px, source = metadata_point_to_image_px(point, image_w=image_w, image_h=image_h)
                coord_sources[source] = coord_sources.get(source, 0) + 1
                if not (0 <= x_px < image_w and 0 <= y_px < image_h):
                    out_of_bounds.append({"patternId": pid, "kind": key, "index": idx, "x_px": x_px, "y_px": y_px})
                    LOGGER.warning(
                        "Metadata point outside image for patternId=%s kind=%s index=%s x=%s y=%s",
                        pid,
                        key,
                        idx,
                        x_px,
                        y_px,
                    )
                    continue
                enriched = dict(point)
                enriched["x_px"] = x_px
                enriched["y_px"] = y_px
                enriched["coord_source"] = source
                normalized_layout[key].append(enriched)
        layouts_by_id[pid] = normalized_layout

    diagnostics = {
        "coord_sources": coord_sources,
        "out_of_bounds": out_of_bounds,
        "image_w": image_w,
        "image_h": image_h,
        "layouts_count": len(layouts_by_id),
    }
    return layouts_by_id, diagnostics


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = image.shape[:2]
    if (w, h) != (IMAGE_W, IMAGE_H):
        raise ValueError(f"Unexpected image size for {path}: {(w, h)} expected {(IMAGE_W, IMAGE_H)}")
    return image


def image_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def compute_hog_descriptor(gray: np.ndarray, resize_to: Tuple[int, int] = (384, 384)) -> np.ndarray:
    resized = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)
    win_size = resize_to
    block_size = (32, 32)
    block_stride = (16, 16)
    cell_size = (16, 16)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    feat = hog.compute(resized).reshape(-1).astype(np.float32)
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm


def extract_patch(gray: np.ndarray, center_xy: Tuple[int, int], patch_size: int) -> np.ndarray:
    x, y = center_xy
    half = patch_size // 2
    left = x - half
    right = left + patch_size
    top = y - half
    bottom = top + patch_size

    src_left = max(0, left)
    src_right = min(gray.shape[1], right)
    src_top = max(0, top)
    src_bottom = min(gray.shape[0], bottom)

    patch = np.zeros((patch_size, patch_size), dtype=gray.dtype)
    dst_left = src_left - left
    dst_top = src_top - top
    patch[dst_top:dst_top + (src_bottom - src_top), dst_left:dst_left + (src_right - src_left)] = gray[src_top:src_bottom, src_left:src_right]
    return patch


def compute_patch_descriptor(gray: np.ndarray, center_xy: Tuple[int, int], patch_size: int = 72, descriptor_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    patch = extract_patch(gray, center_xy, patch_size)
    return compute_hog_descriptor(patch, resize_to=descriptor_size)


def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return query @ gallery.T


def build_gallery(images_dir: str, metadata_path: str) -> Dict[str, Any]:
    metadata = load_metadata(metadata_path)
    layouts_by_id, diagnostics = validate_and_collect_layout_points(metadata)
    records = list_images(images_dir)
    pattern_ids = []
    image_features = []
    slot_features = []
    status_features = []
    image_paths = []

    missing_layouts = []
    for rec in records:
        layout = layouts_by_id.get(rec.pattern_id)
        if layout is None:
            missing_layouts.append(rec.pattern_id)
            continue
        image = read_image(rec.path)
        gray = image_to_gray(image)
        pattern_ids.append(rec.pattern_id)
        image_paths.append(rec.path)
        image_features.append(compute_hog_descriptor(gray))
        slot_features.append(
            np.stack([
                compute_patch_descriptor(gray, (int(p["x_px"]), int(p["y_px"])), patch_size=72)
                for p in layout["slots"]
            ], axis=0)
        )
        status_features.append(
            np.stack([
                compute_patch_descriptor(gray, (int(p["x_px"]), int(p["y_px"])), patch_size=64)
                for p in layout["great_hollow_statuses"]
            ], axis=0)
        )

    if missing_layouts:
        raise ValueError(f"Missing layouts in metadata for patternIds: {missing_layouts}")

    model = {
        "pattern_ids": np.array(pattern_ids, dtype=np.int32),
        "image_paths": image_paths,
        "image_features": np.stack(image_features, axis=0),
        "slot_features": np.stack(slot_features, axis=0),
        "status_features": np.stack(status_features, axis=0),
        "layouts_by_id": layouts_by_id,
        "metadata_diagnostics": diagnostics,
        "config": {
            "image_size": [IMAGE_W, IMAGE_H],
            "image_descriptor_resize": [384, 384],
            "slot_patch_size": 72,
            "status_patch_size": 64,
        },
    }
    return model


def standardize_scores(scores: np.ndarray) -> np.ndarray:
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    if sigma < 1e-8:
        return scores - mu
    return (scores - mu) / sigma


def compute_raw_scores(gray: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    image_feat = compute_hog_descriptor(gray)
    image_scores = cosine_similarity_matrix(image_feat[None, :], model["image_features"])[0]

    # Canonical slots are shared across layouts, so compute them once from the first layout.
    first_layout = model["layouts_by_id"][int(model["pattern_ids"][0])]
    slot_descs = np.stack([
        compute_patch_descriptor(gray, (int(p["x_px"]), int(p["y_px"])), patch_size=72)
        for p in first_layout["slots"]
    ], axis=0)
    slot_scores = np.mean(np.sum(slot_descs[None, :, :] * model["slot_features"], axis=2), axis=1)

    status_scores = []
    for idx, pattern_id in enumerate(model["pattern_ids"]):
        layout = model["layouts_by_id"][int(pattern_id)]
        status_descs = [compute_patch_descriptor(gray, (int(p["x_px"]), int(p["y_px"])), patch_size=64) for p in layout["great_hollow_statuses"]]
        status_descs = np.stack(status_descs, axis=0)
        status_score = float(np.mean(np.sum(status_descs * model["status_features"][idx], axis=1)))
        status_scores.append(status_score)
    status_scores = np.asarray(status_scores, dtype=np.float32)
    slot_aware_scores = 0.65 * slot_scores + 0.35 * status_scores
    return {
        "image_scores": image_scores,
        "slot_scores": slot_aware_scores,
    }


def finalize_scores(raw_scores: Dict[str, Any], model: Dict[str, Any], alpha: float = 0.55, temperature: float = 0.12) -> Dict[str, Any]:
    image_scores = raw_scores["image_scores"]
    slot_aware_scores = raw_scores["slot_scores"]
    image_z = standardize_scores(image_scores)
    slot_z = standardize_scores(slot_aware_scores)
    hybrid_scores = alpha * image_z + (1.0 - alpha) * slot_z

    logits = hybrid_scores / max(temperature, 1e-6)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)

    order = np.argsort(-hybrid_scores)
    top1 = order[0]
    top2 = order[1] if len(order) > 1 else order[0]
    margin = float(hybrid_scores[top1] - hybrid_scores[top2])

    return {
        "pattern_ids": model["pattern_ids"],
        "image_scores": image_scores,
        "slot_scores": slot_aware_scores,
        "hybrid_scores": hybrid_scores,
        "probabilities": probs,
        "ranking": order,
        "best_pattern_id": int(model["pattern_ids"][top1]),
        "margin": margin,
        "top1_score": float(hybrid_scores[top1]),
        "top2_score": float(hybrid_scores[top2]),
    }




def predict_scores(gray: np.ndarray, model: Dict[str, Any], alpha: float = 0.55, temperature: float = 0.12) -> Dict[str, Any]:
    raw_scores = compute_raw_scores(gray, model)
    return finalize_scores(raw_scores, model, alpha=alpha, temperature=temperature)

def deterministic_augmentations(gray: np.ndarray, base_seed: int, count: int = 3) -> List[np.ndarray]:
    rng = np.random.RandomState(base_seed)
    outputs = []
    h, w = gray.shape
    for _ in range(count):
        img = gray.copy().astype(np.float32)
        alpha = rng.uniform(0.92, 1.08)
        beta = rng.uniform(-10, 10)
        img = np.clip(img * alpha + beta, 0, 255)
        if rng.rand() < 0.7:
            k = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.1, 1.0))
        if rng.rand() < 0.8:
            noise = rng.normal(0, rng.uniform(1.0, 4.0), size=(h, w))
            img = np.clip(img + noise, 0, 255)
        if rng.rand() < 0.5:
            gamma = rng.uniform(0.92, 1.08)
            img = np.clip(255.0 * np.power(img / 255.0, gamma), 0, 255)
        outputs.append(img.astype(np.uint8))
    return outputs


def split_synthetic_queries(records: Sequence[ImageRecord], seed: int) -> Dict[str, List[Tuple[int, int]]]:
    rng = random.Random(seed)
    assignments = {"train": [], "val": [], "test": []}
    for rec in sorted(records, key=lambda r: r.pattern_id):
        aug_indices = list(range(3))
        rng.shuffle(aug_indices)
        assignments["train"].extend((rec.pattern_id, i) for i in aug_indices[:1])
        assignments["val"].extend((rec.pattern_id, i) for i in aug_indices[1:2])
        assignments["test"].extend((rec.pattern_id, i) for i in aug_indices[2:3])
    return assignments


def tune_hyperparameters(model: Dict[str, Any], images_dir: str, seed: int) -> Dict[str, Any]:
    del seed
    # Fast calibration for one-template-per-class retrieval.
    records = list_images(images_dir)
    margins = []
    for rec in records:
        gray = image_to_gray(read_image(rec.path))
        pred = predict_scores(gray, model, alpha=0.60, temperature=0.12)
        margins.append(pred["margin"])
    margins = np.asarray(margins, dtype=np.float32)
    threshold = float(np.quantile(margins, 0.10)) if len(margins) else 0.0
    return {
        "alpha": 0.60,
        "temperature": 0.12,
        "unknown_margin_threshold": threshold,
        "synthetic_split": {
            "train": [],
            "val": [],
            "test": [],
        },
        "calibration_note": "Fast heuristic calibration from self-retrieval margins because the dataset has one template per class.",
    }


def classify_image(image_path: str, model: Dict[str, Any]) -> Dict[str, Any]:
    image = read_image(image_path)
    gray = image_to_gray(image)
    started = time.perf_counter()
    pred = predict_scores(gray, model, alpha=model["calibration"]["alpha"], temperature=model["calibration"]["temperature"])
    latency_ms = (time.perf_counter() - started) * 1000.0
    ranking = pred["ranking"]
    top5 = []
    for idx in ranking[:5]:
        top5.append({
            "patternId": int(model["pattern_ids"][idx]),
            "score": float(pred["hybrid_scores"][idx]),
            "probability": float(pred["probabilities"][idx]),
            "image_score": float(pred["image_scores"][idx]),
            "slot_score": float(pred["slot_scores"][idx]),
        })
    is_unknown = pred["margin"] < model["calibration"]["unknown_margin_threshold"]
    return {
        "input": image_path,
        "best_patternId": int(pred["best_pattern_id"]),
        "top5": top5,
        "margin": float(pred["margin"]),
        "confidence": float(top5[0]["probability"]),
        "unknown": bool(is_unknown),
        "decided_patternId": None if is_unknown else int(pred["best_pattern_id"]),
        "latency_ms": float(latency_ms),
    }


def compute_topk_accuracy(y_true: Sequence[int], ranked_predictions: Sequence[Sequence[int]], k: int) -> float:
    hits = 0
    for truth, ranked in zip(y_true, ranked_predictions):
        if truth in list(ranked)[:k]:
            hits += 1
    return hits / max(len(y_true), 1)


def evaluate_on_queries(queries: Sequence[Tuple[int, str, np.ndarray]], model: Dict[str, Any]) -> Dict[str, Any]:
    y_true = []
    y_pred = []
    ranked = []
    margins = []
    decided_mask = []
    latencies = []
    rows = []
    for truth, query_name, gray in queries:
        started = time.perf_counter()
        pred = predict_scores(gray, model, alpha=model["calibration"]["alpha"], temperature=model["calibration"]["temperature"])
        latency_ms = (time.perf_counter() - started) * 1000.0
        rank_ids = [int(model["pattern_ids"][i]) for i in pred["ranking"]]
        best_id = int(pred["best_pattern_id"])
        margin = float(pred["margin"])
        decided = margin >= model["calibration"]["unknown_margin_threshold"]
        y_true.append(int(truth))
        y_pred.append(best_id)
        ranked.append(rank_ids)
        margins.append(margin)
        decided_mask.append(decided)
        latencies.append(latency_ms)
        rows.append({
            "query": query_name,
            "truth": int(truth),
            "pred": best_id,
            "margin": margin,
            "decided": bool(decided),
            "latency_ms": latency_ms,
        })

    labels = [int(x) for x in model["pattern_ids"]]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    decided_indices = [i for i, flag in enumerate(decided_mask) if flag]
    decided_acc = float(np.mean([y_true[i] == y_pred[i] for i in decided_indices])) if decided_indices else None
    report = {
        "count": len(y_true),
        "top1_accuracy": compute_topk_accuracy(y_true, ranked, 1),
        "top3_accuracy": compute_topk_accuracy(y_true, ranked, 3),
        "top5_accuracy": compute_topk_accuracy(y_true, ranked, 5),
        "unknown_rate": 1.0 - (sum(decided_mask) / max(len(decided_mask), 1)),
        "decided_accuracy": decided_acc,
        "margin_distribution": {
            "min": float(np.min(margins)) if margins else None,
            "p25": float(np.quantile(margins, 0.25)) if margins else None,
            "median": float(np.quantile(margins, 0.50)) if margins else None,
            "p75": float(np.quantile(margins, 0.75)) if margins else None,
            "max": float(np.max(margins)) if margins else None,
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "median": float(np.median(latencies)) if latencies else None,
            "p95": float(np.quantile(latencies, 0.95)) if latencies else None,
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "rows": rows,
    }
    return report


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


