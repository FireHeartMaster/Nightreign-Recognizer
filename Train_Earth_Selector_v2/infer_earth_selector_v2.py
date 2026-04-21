#!/usr/bin/env python3
import os, re, csv, json, zipfile, pickle, shutil, tempfile, time, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score

LOGGER = logging.getLogger("earth_selector_v2")
ANNOT_RE = re.compile(r"\.\s+(\S+)\s+\|\s+Shifting Earth=(.+)$")

@dataclass
class Sample:
    file_path: str
    label: str
    filename: str
    source_zip: str

def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='[%(levelname)s] %(message)s')

def parse_annotation_text(text: str) -> List[Tuple[str, str]]:
    rows = []
    for line in text.splitlines():
        if '| Shifting Earth=' not in line:
            continue
        m = ANNOT_RE.search(line.strip())
        if m:
            rows.append((m.group(1).strip(), m.group(2).strip()))
    if not rows:
        raise ValueError("Could not parse any annotation rows from annotation file.")
    return rows

def unpack_sources(source_zips: List[str], work_dir: str) -> List[Sample]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    samples: List[Sample] = []
    for zip_path in source_zips:
        zp = Path(zip_path)
        if not zp.exists():
            raise FileNotFoundError(f"Source zip not found: {zip_path}")
        extract_dir = work / zp.stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, 'r') as zf:
            zf.extractall(extract_dir)
        ann_files = list(extract_dir.glob('*annotations.txt'))
        if len(ann_files) != 1:
            raise ValueError(f"Expected exactly one annotations file in {zip_path}, found {len(ann_files)}.")
        ann_rows = parse_annotation_text(ann_files[0].read_text(encoding='utf-8'))
        for filename, label in ann_rows:
            file_path = extract_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Annotation references missing image: {file_path}")
            samples.append(Sample(str(file_path), label, filename, str(zp)))
    return samples

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unreadable or corrupt image: {path}")
    return img

def letterbox_resize(img: np.ndarray, size: Tuple[int, int] = (192, 192)) -> np.ndarray:
    target_h, target_w = size
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {img.shape}")
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def make_hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor((192, 192), (32, 32), (16, 16), (16, 16), 9)

def extract_features_from_image(img: np.ndarray, hog: Optional[cv2.HOGDescriptor] = None) -> np.ndarray:
    if hog is None:
        hog = make_hog_descriptor()
    lb = letterbox_resize(img, (192, 192))
    gray = cv2.cvtColor(lb, cv2.COLOR_BGR2GRAY)
    hog_feat = hog.compute(gray).reshape(-1).astype(np.float32)
    hog_feat /= (np.linalg.norm(hog_feat) + 1e-8)

    hsv = cv2.cvtColor(lb, cv2.COLOR_BGR2HSV)
    hist_parts = []
    for ch, bins in enumerate((12, 12, 12)):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256]).reshape(-1).astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        hist_parts.append(hist)

    small = cv2.resize(lb, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    color_mean = (small.mean(axis=(0, 1)) / 255.0).astype(np.float32)
    color_std = (small.std(axis=(0, 1)) / 255.0).astype(np.float32)

    edges = cv2.Canny(gray, 100, 200)
    row_profile = (edges.mean(axis=1)[::8] / 255.0).astype(np.float32)
    col_profile = (edges.mean(axis=0)[::8] / 255.0).astype(np.float32)

    feat = np.concatenate([hog_feat, *hist_parts, color_mean, color_std, row_profile, col_profile]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat

def extract_features_from_path(path: str, hog: Optional[cv2.HOGDescriptor] = None) -> np.ndarray:
    return extract_features_from_image(read_image(path), hog=hog)

def build_feature_matrix(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    hog = make_hog_descriptor()
    feats, labels, bad_files = [], [], []
    for sample in samples:
        try:
            feats.append(extract_features_from_path(sample.file_path, hog=hog))
            labels.append(sample.label)
        except Exception as e:
            LOGGER.warning("Skipping unreadable/corrupt image %s: %s", sample.file_path, e)
            bad_files.append(sample.file_path)
    if not feats:
        raise ValueError("No readable images found.")
    return np.stack(feats).astype(np.float32), np.array(labels), bad_files

def compute_centroids(X: np.ndarray, y: np.ndarray, labels: List[str]) -> np.ndarray:
    centroids = []
    for label in labels:
        mu = X[y == label].mean(axis=0)
        mu /= (np.linalg.norm(mu) + 1e-8)
        centroids.append(mu.astype(np.float32))
    return np.stack(centroids)

def predict_proba_from_centroids(X: np.ndarray, centroids: np.ndarray, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
    sims = X @ centroids.T
    logits = sims / max(float(temperature), 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return sims, probs

def score_probability_row(prob_row: np.ndarray, labels: List[str], unknown_margin_threshold: float, unknown_conf_threshold: float) -> Dict[str, Any]:
    order = np.argsort(-prob_row)
    topk = []
    for idx in order[: min(6, len(order))]:
        topk.append({"label": str(labels[idx]), "probability": float(prob_row[idx]), "score": float(prob_row[idx])})
    top1 = order[0]
    top2 = order[1] if len(order) > 1 else order[0]
    confidence = float(prob_row[top1])
    margin = float(prob_row[top1] - prob_row[top2])
    unknown = bool((margin < unknown_margin_threshold) or (confidence < unknown_conf_threshold))
    return {
        "label": str(labels[top1]),
        "confidence": confidence,
        "margin_top1_top2": margin,
        "topk": topk,
        "unknown": unknown,
        "decided_label": None if unknown else str(labels[top1]),
    }

def train_selector(samples: List[Sample], seed: int = 42, temperature: float = 0.05) -> Dict[str, Any]:
    X, y, bad_files = build_feature_matrix(samples)
    labels = sorted(set(y.tolist()))
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(outer.split(X, y))
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(inner.split(X_trainval, y_trainval))
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    centroids_val = compute_centroids(X_train, y_train, labels)
    _, val_probs = predict_proba_from_centroids(X_val, centroids_val, temperature=temperature)
    val_margins, val_confs = [], []
    for row in val_probs:
        order = np.argsort(-row)
        val_margins.append(float(row[order[0]] - row[order[1]]))
        val_confs.append(float(row[order[0]]))
    unknown_margin_threshold = float(np.quantile(np.array(val_margins), 0.10))
    unknown_conf_threshold = float(np.quantile(np.array(val_confs), 0.10))

    centroids_final = compute_centroids(X_trainval, y_trainval, labels)
    _, test_probs = predict_proba_from_centroids(X_test, centroids_final, temperature=temperature)
    test_pred = np.array(labels)[np.argmax(test_probs, axis=1)]

    per_class_accuracy = {}
    for cls in labels:
        mask = (y_test == cls)
        per_class_accuracy[str(cls)] = float((test_pred[mask] == y_test[mask]).mean())

    per_image, unknown_count, decided_correct, latencies_ms = [], 0, [], []
    for i, sample_idx in enumerate(test_idx):
        started = time.perf_counter()
        scored = score_probability_row(test_probs[i], labels, unknown_margin_threshold, unknown_conf_threshold)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(latency_ms)
        truth = str(y[sample_idx])
        if scored["unknown"]:
            unknown_count += 1
        else:
            decided_correct.append(int(scored["decided_label"] == truth))
        per_image.append({"file": samples[sample_idx].filename, "truth": truth, **scored, "latency_ms": latency_ms})

    eval_report = {
        "seed": seed,
        "num_samples_total": int(len(samples)),
        "num_bad_files_skipped": int(len(bad_files)),
        "bad_files": bad_files,
        "labels": labels,
        "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "thresholds": {
            "temperature": float(temperature),
            "unknown_margin_threshold": unknown_margin_threshold,
            "unknown_conf_threshold": unknown_conf_threshold,
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "unknown_rate": float(unknown_count / len(y_test)),
            "accuracy_on_decided_samples": float(np.mean(decided_correct)) if decided_correct else None,
            "mean_latency_ms_scoring_only": float(np.mean(latencies_ms)) if latencies_ms else None,
        },
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_test, test_pred, labels=labels).tolist()},
        "test_predictions": per_image,
        "model_info": {
            "feature_type": "letterboxed_192_hog_hsvhist_colorstats_edgeprofiles",
            "classifier": "cosine-to-class-centroids + softmax",
            "handles_mixed_input_sizes": True,
        },
    }
    model = {
        "labels": labels,
        "centroids": centroids_final.astype(np.float32),
        "temperature": float(temperature),
        "unknown_margin_threshold": unknown_margin_threshold,
        "unknown_conf_threshold": unknown_conf_threshold,
        "feature_config": {"letterbox_size": [192, 192], "hog_window": [192, 192]},
        "seed": int(seed),
        "eval_summary": {"accuracy": eval_report["metrics"]["accuracy"], "unknown_rate": eval_report["metrics"]["unknown_rate"]},
    }
    return {"model": model, "eval_report": eval_report}

def predict_one_path(path: str, model: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    feat = extract_features_from_path(path).reshape(1, -1)
    centroids = np.asarray(model["centroids"], dtype=np.float32)
    _, probs = predict_proba_from_centroids(feat, centroids, temperature=float(model["temperature"]))
    scored = score_probability_row(probs[0], model["labels"], float(model["unknown_margin_threshold"]), float(model["unknown_conf_threshold"]))
    scored["file"] = path
    scored["latency_ms"] = (time.perf_counter() - started) * 1000.0
    return scored

def iter_input_images(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files: List[str] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            files.extend(sorted(str(x) for x in p.glob(ext)))
        if not files:
            raise FileNotFoundError(f"No supported images found in folder: {input_path}")
        return files
    raise FileNotFoundError(f"Input not found: {input_path}")
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Infer Shifting Earth category from one image or a folder.")
    parser.add_argument("--model", required=True, help="Path to trained model.")
    parser.add_argument("--input", required=True, help="Image file or folder.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output for folder inference.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output for folder inference.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    files = iter_input_images(args.input)
    results = []
    for path in files:
        try:
            results.append(predict_one_path(path, model))
        except Exception as e:
            results.append({"file": path, "error": str(e), "label": None, "confidence": None, "margin_top1_top2": None, "topk": [], "unknown": True, "decided_label": None})
    if len(results) == 1 and args.output_json is None and args.output_csv is None:
        print(json.dumps(results[0], indent=2))
        return
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label", "confidence", "margin_top1_top2", "unknown", "decided_label"])
            for row in results:
                writer.writerow([row.get("file"), row.get("label"), row.get("confidence"), row.get("margin_top1_top2"), row.get("unknown"), row.get("decided_label")])
    print(json.dumps(results, indent=2))
if __name__ == "__main__":
    main()
