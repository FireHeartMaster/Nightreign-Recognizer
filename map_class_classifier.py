#!/usr/bin/env python3
"""Map class classifier for labeled image sets stored in folders or zip files.

This classifier is intentionally simple and robust for a small number of visual classes.
It does not use deep learning. Instead, it builds a descriptor for each training image from:
- a normalized grayscale thumbnail
- a normalized edge thumbnail
- an HSV color histogram

Prediction uses cosine similarity against all training exemplars and a weighted top-k vote.

Example:
    python map_class_classifier.py fit \
        --sources Limveld.zip Mountaintop.zip "The Crater.zip" \
        --model map_class_model.pkl

    python map_class_classifier.py predict \
        --model map_class_model.pkl \
        --input some_image.png

    python map_class_classifier.py predict \
        --model map_class_model.pkl \
        --input ./images \
        --output-csv results.csv --output-json results.json --recursive
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
EARTH_SELECTOR_INPUT_SIZE = 1536
EARTH_SELECTOR_BORDER_CROP_PX = 256
EARTH_SELECTOR_CROPPED_SIZE = EARTH_SELECTOR_INPUT_SIZE - (2 * EARTH_SELECTOR_BORDER_CROP_PX)


class ClassifierError(RuntimeError):
    pass


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def iter_images_in_folder(folder: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from sorted(p for p in folder.rglob("*") if p.is_file() and is_image_file(p))
    else:
        yield from sorted(p for p in folder.iterdir() if p.is_file() and is_image_file(p))


def read_image_any(path_or_bytes) -> np.ndarray:
    if isinstance(path_or_bytes, (str, os.PathLike, Path)):
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
        if img is None:
            raise ClassifierError(f"Could not read image: {path_or_bytes}")
        return img

    data = np.frombuffer(path_or_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ClassifierError("Could not decode image bytes from zip archive.")
    return img


def crop_border_fraction(img: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0:
        return img
    h, w = img.shape[:2]
    dy = int(round(h * frac))
    dx = int(round(w * frac))
    if dy * 2 >= h or dx * 2 >= w:
        return img
    return img[dy : h - dy, dx : w - dx]


def preprocess_earth_selector_panel(
    img: np.ndarray,
    input_size: int = EARTH_SELECTOR_INPUT_SIZE,
    border_crop_px: int = EARTH_SELECTOR_BORDER_CROP_PX,
) -> np.ndarray:
    """Step 1 canonical preprocessing helper: 1536 -> crop 256 per side -> 1024."""
    if img is None or img.size == 0:
        raise ClassifierError("Cannot preprocess an empty image.")
    resized = cv2.resize(img, (int(input_size), int(input_size)), interpolation=cv2.INTER_AREA)
    border = int(border_crop_px)
    if border <= 0:
        return resized
    if border * 2 >= resized.shape[0] or border * 2 >= resized.shape[1]:
        raise ClassifierError("Invalid border crop for earth-selector preprocessing.")
    return resized[border : resized.shape[0] - border, border : resized.shape[1] - border]


def compute_descriptor(
    img: np.ndarray,
    thumb_size: int = 24,
    hist_h_bins: int = 12,
    hist_s_bins: int = 4,
    crop_frac: float = 0.02,
    use_canonical_preprocess: bool = True,
) -> np.ndarray:
    """Build a compact global descriptor for the full map image."""
    if use_canonical_preprocess:
        work = preprocess_earth_selector_panel(img)
    else:
        work = crop_border_fraction(img, crop_frac)
    work = cv2.resize(work, (96, 96), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Low-resolution luminance thumbnail
    small = cv2.resize(gray, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32).reshape(-1)
    small = (small - small.mean()) / (small.std() + 1e-6)

    # Edge thumbnail
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge = cv2.resize(mag, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA).reshape(-1)
    edge = (edge - edge.mean()) / (edge.std() + 1e-6)

    # Global color distribution
    hist = cv2.calcHist([hsv], [0, 1], None, [hist_h_bins, hist_s_bins], [0, 180, 0, 256])
    hist = hist.reshape(-1).astype(np.float32)
    hist /= hist.sum() + 1e-6

    descriptor = np.concatenate([
        0.55 * small,
        0.30 * edge,
        3.00 * hist,
    ]).astype(np.float32)
    descriptor /= np.linalg.norm(descriptor) + 1e-6
    return descriptor


def _iter_training_images_from_zip(zip_path: Path) -> Iterable[Tuple[str, str, bytes]]:
    class_name = zip_path.stem
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            if Path(info.filename).suffix.lower() not in IMAGE_EXTS:
                continue
            yield class_name, info.filename, zf.read(info.filename)


def _iter_training_images_from_folder(folder: Path, recursive: bool = True) -> Iterable[Tuple[str, str, Path]]:
    class_name = folder.name
    for p in iter_images_in_folder(folder, recursive=recursive):
        yield class_name, str(p.name), p


def collect_training_samples(sources: Sequence[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    descriptors: List[np.ndarray] = []
    labels: List[str] = []
    origins: List[str] = []

    for src in sources:
        path = Path(src)
        if not path.exists():
            raise ClassifierError(f"Training source does not exist: {src}")

        if path.is_dir():
            items = _iter_training_images_from_folder(path, recursive=True)
        elif path.suffix.lower() == ".zip":
            items = _iter_training_images_from_zip(path)
        else:
            raise ClassifierError(f"Unsupported training source: {src}. Use a folder or .zip file.")

        count_before = len(labels)
        for class_name, inner_name, payload in items:
            img = read_image_any(payload)
            descriptors.append(compute_descriptor(img))
            labels.append(class_name)
            origins.append(f"{path.name}:{inner_name}" if path.suffix.lower() == ".zip" else str(payload))

        if len(labels) == count_before:
            raise ClassifierError(f"No images found in training source: {src}")

    if not descriptors:
        raise ClassifierError("No training images found.")

    return np.stack(descriptors).astype(np.float32), labels, origins


def build_model(sources: Sequence[str], top_k: int = 5) -> Dict:
    descriptors, labels, origins = collect_training_samples(sources)
    classes = sorted(set(labels))
    return {
        "version": 1,
        "classes": classes,
        "labels": labels,
        "origins": origins,
        "descriptors": descriptors,
        "top_k": max(1, int(top_k)),
        "descriptor_config": {
            "thumb_size": 24,
            "hist_h_bins": 12,
            "hist_s_bins": 4,
            "crop_frac": 0.02,
            "use_canonical_preprocess": True,
            "input_size": EARTH_SELECTOR_INPUT_SIZE,
            "border_crop_px": EARTH_SELECTOR_BORDER_CROP_PX,
            "cropped_size": EARTH_SELECTOR_CROPPED_SIZE,
        },
    }


def save_model(model: Dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str) -> Dict:
    with open(path, "rb") as f:
        model = pickle.load(f)
    required = {"classes", "labels", "origins", "descriptors", "top_k"}
    missing = required - set(model.keys())
    if missing:
        raise ClassifierError(f"Model file is missing fields: {sorted(missing)}")
    model["descriptors"] = np.asarray(model["descriptors"], dtype=np.float32)
    return model


def classify_descriptor(model: Dict, descriptor: np.ndarray, exclude_origin: str | None = None) -> Dict:
    descs = model["descriptors"]
    labels = model["labels"]
    origins = model["origins"]
    sims = descs @ descriptor.astype(np.float32)

    if exclude_origin is not None:
        for i, origin in enumerate(origins):
            if origin == exclude_origin:
                sims[i] = -np.inf

    k = min(int(model.get("top_k", 5)), len(sims))
    if k <= 0:
        raise ClassifierError("Model has no training descriptors.")

    top_idx = np.argsort(sims)[-k:][::-1]
    class_scores: Dict[str, float] = defaultdict(float)
    neighbors = []
    for idx in top_idx:
        label = labels[idx]
        score = float(sims[idx])
        class_scores[label] += score
        neighbors.append({
            "label": label,
            "score": score,
            "origin": origins[idx],
        })

    ranked = sorted(class_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    runner_label, runner_score = (ranked[1] if len(ranked) > 1 else (None, float("-inf")))

    margin = best_score - runner_score if runner_label is not None else best_score
    if margin >= 1.0:
        confidence = "strong"
    elif margin >= 0.35:
        confidence = "medium"
    else:
        confidence = "weak"

    return {
        "label": best_label,
        "score": best_score,
        "runner_up": runner_label,
        "runner_up_score": runner_score if runner_label is not None else None,
        "margin": margin,
        "confidence": confidence,
        "neighbors": neighbors,
        "class_scores": [{"label": lbl, "score": float(scr)} for lbl, scr in ranked],
    }


def classify_image(model: Dict, image_path: str) -> Dict:
    img = read_image_any(image_path)
    descriptor = compute_descriptor(img)
    result = classify_descriptor(model, descriptor)
    result["file"] = str(image_path)
    return result


def evaluate_leave_one_out(model: Dict) -> Dict:
    descs = model["descriptors"]
    labels = np.array(model["labels"])
    origins = model["origins"]

    results = []
    for i in range(len(descs)):
        pred = classify_descriptor(model, descs[i], exclude_origin=origins[i])
        results.append({
            "origin": origins[i],
            "true_label": labels[i],
            "predicted_label": pred["label"],
            "confidence": pred["confidence"],
            "margin": pred["margin"],
            "correct": bool(pred["label"] == labels[i]),
        })

    accuracy = sum(r["correct"] for r in results) / max(1, len(results))
    confusion: Dict[str, Dict[str, int]] = {
        true: {pred: 0 for pred in model["classes"]}
        for true in model["classes"]
    }
    for r in results:
        confusion[r["true_label"]][r["predicted_label"]] += 1

    return {
        "num_samples": len(results),
        "accuracy": accuracy,
        "results": results,
        "confusion": confusion,
    }


def predict_folder(model: Dict, folder: str, recursive: bool = False) -> List[Dict]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ClassifierError(f"Input folder does not exist: {folder}")

    results = []
    for p in iter_images_in_folder(folder_path, recursive=recursive):
        try:
            result = classify_image(model, str(p))
            result["status"] = "ok"
            result["error"] = ""
        except Exception as e:  # noqa: BLE001
            result = {
                "file": str(p),
                "status": "error",
                "error": str(e),
                "label": None,
                "score": None,
                "runner_up": None,
                "runner_up_score": None,
                "margin": None,
                "confidence": None,
                "neighbors": [],
                "class_scores": [],
            }
        results.append(result)
    return results


def write_csv(results: List[Dict], output_csv: str) -> None:
    fieldnames = [
        "file",
        "label",
        "score",
        "runner_up",
        "runner_up_score",
        "margin",
        "confidence",
        "status",
        "error",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to the classifier model .pkl file.")


def cmd_fit(args: argparse.Namespace) -> int:
    model = build_model(args.sources, top_k=args.top_k)
    save_model(model, args.model)
    print(f"Model written to: {args.model}")
    print(f"Classes: {', '.join(model['classes'])}")
    print(f"Training images: {len(model['labels'])}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    input_path = Path(args.input)

    if input_path.is_dir():
        results = predict_folder(model, str(input_path), recursive=args.recursive)
        ok_count = sum(r.get("status") == "ok" for r in results)
        err_count = len(results) - ok_count

        for r in results:
            if r.get("status") == "ok":
                print(
                    f"[OK] {Path(r['file']).name} -> {r['label']} "
                    f"({r['confidence']}, margin={r['margin']:.3f})"
                )
            else:
                print(f"[ERROR] {Path(r['file']).name} -> {r['error']}")

        print(f"\nDone. Processed {len(results)} image(s): {ok_count} ok, {err_count} error(s).")

        if args.output_csv:
            write_csv(results, args.output_csv)
            print(f"CSV written to: {args.output_csv}")
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"JSON written to: {args.output_json}")
    else:
        result = classify_image(model, args.input)
        print(json.dumps(result, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    report = evaluate_leave_one_out(model)
    print(json.dumps({
        "num_samples": report["num_samples"],
        "accuracy": report["accuracy"],
        "confusion": report["confusion"],
    }, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed evaluation written to: {args.output_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify map images into one of several labeled classes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_p = subparsers.add_parser("fit", help="Build a model from class folders or zip files.")
    fit_p.add_argument("--sources", nargs="+", required=True, help="Training sources: each source is a class folder or .zip file.")
    fit_p.add_argument("--model", required=True, help="Output model .pkl file.")
    fit_p.add_argument("--top-k", type=int, default=5, help="How many nearest neighbors to use during prediction.")
    fit_p.set_defaults(func=cmd_fit)

    pred_p = subparsers.add_parser("predict", help="Classify one image or every image in a folder.")
    add_shared_args(pred_p)
    pred_p.add_argument("--input", required=True, help="Path to a query image or a folder of query images.")
    pred_p.add_argument("--recursive", action="store_true", help="Recurse into subfolders when --input is a folder.")
    pred_p.add_argument("--output-csv", help="Optional CSV path for folder predictions.")
    pred_p.add_argument("--output-json", help="Optional JSON path for folder predictions.")
    pred_p.set_defaults(func=cmd_predict)

    eval_p = subparsers.add_parser("evaluate", help="Run leave-one-out evaluation on a saved model.")
    add_shared_args(eval_p)
    eval_p.add_argument("--output-json", help="Optional path for a detailed evaluation report.")
    eval_p.set_defaults(func=cmd_evaluate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except ClassifierError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
