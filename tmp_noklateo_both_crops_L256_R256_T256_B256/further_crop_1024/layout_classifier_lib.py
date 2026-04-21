import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger('noklateo_layout_classifier')
IMAGE_W = 1024
IMAGE_H = 1024
PATTERN_RE = re.compile(r'pattern_(\d+)_')


@dataclass
class ImageRecord:
    pattern_id: int
    path: str
    sha256: str


@dataclass
class QuerySample:
    pattern_id: int
    query_name: str
    gray: np.ndarray


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def parse_args_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--metadata', required=True, help='Path to Noklateo strict metadata JSON')
    parser.add_argument('--verbose', action='store_true')


def parse_pattern_id(filename: str) -> int:
    match = PATTERN_RE.search(os.path.basename(filename))
    if not match:
        raise ValueError(f'Could not parse patternId from filename: {filename}')
    return int(match.group(1))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def list_images(images_dir: str) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for path in sorted(Path(images_dir).glob('*.png')):
        records.append(ImageRecord(pattern_id=parse_pattern_id(path.name), path=str(path), sha256=file_sha256(str(path))))
    if not records:
        raise FileNotFoundError(f'No PNG images found in {images_dir}')
    return records


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    if 'layouts' not in metadata:
        raise KeyError("Metadata JSON must contain 'layouts'")
    target = metadata.get('coordinate_system', {}).get('target', {})
    size = target.get('size', {})
    if size.get('width') not in (None, IMAGE_W) or size.get('height') not in (None, IMAGE_H):
        raise ValueError(f'Metadata target size mismatch: {size} expected width={IMAGE_W} height={IMAGE_H}')
    return metadata


def _norm_to_px(norm_value: float, size: int) -> int:
    return int(round(float(norm_value) * (size - 1)))


def metadata_point_to_image_px(point: Dict[str, Any], image_w: int = IMAGE_W, image_h: int = IMAGE_H) -> Tuple[int, int, str]:
    """
    Strict contract:
      x_px = round(x_further_px_bl)
      y_px_top_left = (H - 1) - round(y_further_px_bl)

    Primary source: *_px_bl.
    Fallback: *_norm_bl only.
    No other coordinate system is allowed.
    """
    if 'x_further_px_bl' in point and 'y_further_px_bl' in point:
        x_bl = float(point['x_further_px_bl'])
        y_bl = float(point['y_further_px_bl'])
        source = 'px_bl'
    elif 'x_further_norm_bl' in point and 'y_further_norm_bl' in point:
        x_bl = _norm_to_px(float(point['x_further_norm_bl']), image_w)
        y_bl = _norm_to_px(float(point['y_further_norm_bl']), image_h)
        source = 'norm_bl'
    else:
        raise KeyError(
            'Point must contain x_further_px_bl/y_further_px_bl or x_further_norm_bl/y_further_norm_bl. '
            f'Got keys: {sorted(point.keys())}'
        )
    x_px = int(round(x_bl))
    y_px_top_left = int((image_h - 1) - round(y_bl))
    return x_px, y_px_top_left, source


def validate_and_collect_layout_points(
    metadata: Dict[str, Any], image_w: int = IMAGE_W, image_h: int = IMAGE_H
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    layouts_by_id: Dict[int, Dict[str, Any]] = {}
    coord_sources: Dict[str, int] = {}
    out_of_bounds: List[Dict[str, Any]] = []

    for layout in metadata['layouts']:
        pid = int(layout['patternId'])
        if pid in layouts_by_id:
            raise ValueError(f'Duplicate layout for patternId {pid}')
        normalized = {'patternId': pid, 'slots': [], 'landmarks': []}
        for key in ('slots', 'landmarks'):
            for idx, point in enumerate(layout.get(key, [])):
                x_px, y_px, source = metadata_point_to_image_px(point, image_w=image_w, image_h=image_h)
                coord_sources[source] = coord_sources.get(source, 0) + 1
                if not (0 <= x_px < image_w and 0 <= y_px < image_h):
                    event = {'patternId': pid, 'kind': key, 'index': idx, 'x_px': x_px, 'y_px': y_px}
                    out_of_bounds.append(event)
                    LOGGER.warning('Metadata point outside image bounds: %s', event)
                    continue
                enriched = dict(point)
                enriched['x_px'] = x_px
                enriched['y_px'] = y_px
                enriched['coord_source'] = source
                normalized[key].append(enriched)
        layouts_by_id[pid] = normalized

    diagnostics = {
        'image_w': image_w,
        'image_h': image_h,
        'layouts_count': len(layouts_by_id),
        'coord_sources': coord_sources,
        'out_of_bounds': out_of_bounds,
    }
    return layouts_by_id, diagnostics


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    h, w = image.shape[:2]
    if (w, h) != (IMAGE_W, IMAGE_H):
        raise ValueError(f'Unexpected image size {(w, h)} for {path}; expected {(IMAGE_W, IMAGE_H)}')
    return image


def image_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def compute_hog_descriptor(gray: np.ndarray, resize_to: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(resize_to, (32, 32), (16, 16), (16, 16), 9)
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


def compute_patch_descriptor(gray: np.ndarray, center_xy: Tuple[int, int], patch_size: int, descriptor_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    patch = extract_patch(gray, center_xy, patch_size)
    return compute_hog_descriptor(patch, descriptor_size)


def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return query @ gallery.T


def deterministic_augmentations(gray: np.ndarray, base_seed: int, count: int = 3) -> List[np.ndarray]:
    rng = np.random.RandomState(base_seed)
    outputs = []
    for _ in range(count):
        img = gray.astype(np.float32).copy()
        img = np.clip(img * rng.uniform(0.92, 1.08) + rng.uniform(-10, 10), 0, 255)
        if rng.rand() < 0.7:
            k = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.1, 1.0))
        if rng.rand() < 0.7:
            noise = rng.normal(0, rng.uniform(1.0, 4.0), size=img.shape)
            img = np.clip(img + noise, 0, 255)
        if rng.rand() < 0.5:
            gamma = rng.uniform(0.92, 1.08)
            img = np.clip(255.0 * np.power(img / 255.0, gamma), 0, 255)
        outputs.append(img.astype(np.uint8))
    return outputs


def detect_duplicates(records: Sequence[ImageRecord]) -> Dict[str, Any]:
    buckets: Dict[str, List[int]] = {}
    for r in records:
        buckets.setdefault(r.sha256, []).append(r.pattern_id)
    duplicates = [ids for ids in buckets.values() if len(ids) > 1]
    return {
        'exact_duplicate_groups': duplicates,
        'duplicate_group_count': len(duplicates),
        'total_images': len(records),
        'unique_hashes': len(buckets),
    }


def standardize_scores(scores: np.ndarray) -> np.ndarray:
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    if sigma < 1e-8:
        return scores - mu
    return (scores - mu) / sigma


def build_gallery(images_dir: str, metadata_path: str) -> Dict[str, Any]:
    metadata = load_metadata(metadata_path)
    layouts_by_id, diagnostics = validate_and_collect_layout_points(metadata)
    records = list_images(images_dir)
    if len(records) != len(layouts_by_id):
        LOGGER.warning('Image count (%s) != layout count (%s)', len(records), len(layouts_by_id))

    pattern_ids: List[int] = []
    image_paths: List[str] = []
    image_features: List[np.ndarray] = []
    slot_features: List[np.ndarray] = []
    landmark_features: List[np.ndarray] = []

    missing_layouts: List[int] = []
    for rec in records:
        layout = layouts_by_id.get(rec.pattern_id)
        if layout is None:
            missing_layouts.append(rec.pattern_id)
            continue
        gray = image_to_gray(read_image(rec.path))
        pattern_ids.append(rec.pattern_id)
        image_paths.append(rec.path)
        image_features.append(compute_hog_descriptor(gray, resize_to=(384, 384)))
        slot_features.append(np.stack([
            compute_patch_descriptor(gray, (int(p['x_px']), int(p['y_px'])), patch_size=72)
            for p in layout['slots']
        ], axis=0))
        landmark_features.append(np.stack([
            compute_patch_descriptor(gray, (int(p['x_px']), int(p['y_px'])), patch_size=64)
            for p in layout['landmarks']
        ], axis=0))

    if missing_layouts:
        raise ValueError(f'Metadata layouts missing for patternIds: {missing_layouts}')

    model = {
        'pattern_ids': np.asarray(pattern_ids, dtype=np.int32),
        'image_paths': image_paths,
        'image_features': np.stack(image_features, axis=0),
        'slot_features': np.stack(slot_features, axis=0),
        'landmark_features': landmark_features,
        'layouts_by_id': layouts_by_id,
        'metadata_diagnostics': diagnostics,
        'duplicate_report': detect_duplicates(records),
        'config': {
            'image_size': [IMAGE_W, IMAGE_H],
            'slot_patch_size': 72,
            'landmark_patch_size': 64,
            'image_descriptor_resize': [384, 384],
            'earth': metadata.get('shifting_earth', 'Noklateo'),
            'metadata_path': metadata_path,
        },
    }
    return model


def compute_raw_scores(gray: np.ndarray, model: Dict[str, Any]) -> Dict[str, np.ndarray]:
    image_feat = compute_hog_descriptor(gray, resize_to=(384, 384))
    image_scores = cosine_similarity_matrix(image_feat[None, :], model['image_features'])[0]

    first_layout = model['layouts_by_id'][int(model['pattern_ids'][0])]
    query_slot_descs = np.stack([
        compute_patch_descriptor(gray, (int(p['x_px']), int(p['y_px'])), patch_size=72)
        for p in first_layout['slots']
    ], axis=0)
    slot_scores = np.mean(np.sum(query_slot_descs[None, :, :] * model['slot_features'], axis=2), axis=1)

    landmark_scores: List[float] = []
    for idx, pid in enumerate(model['pattern_ids']):
        layout = model['layouts_by_id'][int(pid)]
        query_landmarks = np.stack([
            compute_patch_descriptor(gray, (int(p['x_px']), int(p['y_px'])), patch_size=64)
            for p in layout['landmarks']
        ], axis=0)
        gallery_landmarks = model['landmark_features'][idx]
        landmark_scores.append(float(np.mean(np.sum(query_landmarks * gallery_landmarks, axis=1))))
    landmark_scores_np = np.asarray(landmark_scores, dtype=np.float32)
    slot_aware_scores = 0.70 * slot_scores + 0.30 * landmark_scores_np
    return {'image_scores': image_scores, 'slot_scores': slot_aware_scores}


def finalize_scores(raw_scores: Dict[str, np.ndarray], model: Dict[str, Any], alpha: float, temperature: float) -> Dict[str, Any]:
    image_z = standardize_scores(raw_scores['image_scores'])
    slot_z = standardize_scores(raw_scores['slot_scores'])
    hybrid_scores = alpha * image_z + (1.0 - alpha) * slot_z
    logits = hybrid_scores / max(temperature, 1e-6)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    ranking = np.argsort(-hybrid_scores)
    top1 = int(ranking[0])
    top2 = int(ranking[1]) if len(ranking) > 1 else int(ranking[0])
    return {
        'pattern_ids': model['pattern_ids'],
        'image_scores': raw_scores['image_scores'],
        'slot_scores': raw_scores['slot_scores'],
        'hybrid_scores': hybrid_scores,
        'probabilities': probs,
        'ranking': ranking,
        'best_pattern_id': int(model['pattern_ids'][top1]),
        'margin': float(hybrid_scores[top1] - hybrid_scores[top2]),
    }


def predict_scores(gray: np.ndarray, model: Dict[str, Any], alpha: Optional[float] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
    cal = model.get('calibration', {})
    alpha = float(cal.get('alpha', 0.60) if alpha is None else alpha)
    temperature = float(cal.get('temperature', 0.12) if temperature is None else temperature)
    return finalize_scores(compute_raw_scores(gray, model), model, alpha=alpha, temperature=temperature)


def build_query_splits(images_dir: str, seed: int) -> Dict[str, List[QuerySample]]:
    """
    With one image per patternId, a normal class-held-out split is not valid for a closed-set pattern matcher.
    Instead we build deterministic augmentation-based query splits while keeping the full gallery fixed.
    """
    records = list_images(images_dir)
    split_names = ['train', 'val', 'test']
    queries: Dict[str, List[QuerySample]] = {k: [] for k in split_names}
    for rec in records:
        gray = image_to_gray(read_image(rec.path))
        augments = deterministic_augmentations(gray, base_seed=seed + rec.pattern_id, count=3)
        for split_name, aug in zip(split_names, augments):
            queries[split_name].append(QuerySample(rec.pattern_id, f'{Path(rec.path).name}::{split_name}_aug', aug))
    return queries


def calibrate_model(model: Dict[str, Any], images_dir: str, seed: int) -> Dict[str, Any]:
    queries = build_query_splits(images_dir, seed)
    val_margins: List[float] = []
    for q in queries['val']:
        pred = predict_scores(q.gray, model, alpha=0.60, temperature=0.12)
        val_margins.append(float(pred['margin']))
    threshold = float(np.quantile(np.asarray(val_margins, dtype=np.float32), 0.10)) if val_margins else 0.0
    return {
        'alpha': 0.60,
        'temperature': 0.12,
        'unknown_margin_threshold': threshold,
        'split_protocol': 'deterministic augmentation queries with full gallery fixed; one template per patternId in dataset',
    }


def classify_image(image_path: str, model: Dict[str, Any]) -> Dict[str, Any]:
    gray = image_to_gray(read_image(image_path))
    started = time.perf_counter()
    pred = predict_scores(gray, model)
    latency_ms = (time.perf_counter() - started) * 1000.0
    top5 = []
    for idx in pred['ranking'][:5]:
        i = int(idx)
        top5.append({
            'patternId': int(model['pattern_ids'][i]),
            'score': float(pred['hybrid_scores'][i]),
            'probability': float(pred['probabilities'][i]),
            'image_score': float(pred['image_scores'][i]),
            'slot_score': float(pred['slot_scores'][i]),
        })
    threshold = float(model['calibration']['unknown_margin_threshold'])
    is_unknown = float(pred['margin']) < threshold
    return {
        'input': image_path,
        'best_patternId': int(pred['best_pattern_id']),
        'top5': top5,
        'margin': float(pred['margin']),
        'confidence': float(top5[0]['probability']),
        'unknown': bool(is_unknown),
        'decided_patternId': None if is_unknown else int(pred['best_pattern_id']),
        'latency_ms': float(latency_ms),
        'unknown_margin_threshold': threshold,
    }


def compute_topk_accuracy(y_true: Sequence[int], ranked_predictions: Sequence[Sequence[int]], k: int) -> float:
    if not y_true:
        return 0.0
    return float(np.mean([truth in list(ranked)[:k] for truth, ranked in zip(y_true, ranked_predictions)]))


def confusion_matrix_manual(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    index = {int(label): i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for truth, pred in zip(y_true, y_pred):
        cm[index[int(truth)], index[int(pred)]] += 1
    return cm


def evaluate_queries(queries: Sequence[QuerySample], model: Dict[str, Any]) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred: List[int] = []
    ranked: List[List[int]] = []
    margins: List[float] = []
    decided: List[bool] = []
    latencies: List[float] = []
    rows: List[Dict[str, Any]] = []

    threshold = float(model['calibration']['unknown_margin_threshold'])
    for q in queries:
        started = time.perf_counter()
        pred = predict_scores(q.gray, model)
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranking = [int(model['pattern_ids'][i]) for i in pred['ranking']]
        best_id = int(pred['best_pattern_id'])
        margin = float(pred['margin'])
        is_decided = margin >= threshold

        y_true.append(int(q.pattern_id))
        y_pred.append(best_id)
        ranked.append(ranking)
        margins.append(margin)
        decided.append(is_decided)
        latencies.append(latency_ms)
        rows.append({
            'query': q.query_name,
            'truth': int(q.pattern_id),
            'pred': best_id,
            'margin': margin,
            'unknown': not is_decided,
            'latency_ms': latency_ms,
        })

    labels = [int(x) for x in model['pattern_ids']]
    cm = confusion_matrix_manual(y_true, y_pred, labels)
    decided_idx = [i for i, flag in enumerate(decided) if flag]
    decided_acc = float(np.mean([y_true[i] == y_pred[i] for i in decided_idx])) if decided_idx else None
    return {
        'count': len(queries),
        'top1_accuracy': compute_topk_accuracy(y_true, ranked, 1),
        'top3_accuracy': compute_topk_accuracy(y_true, ranked, 3),
        'top5_accuracy': compute_topk_accuracy(y_true, ranked, 5),
        'unknown_rate': 1.0 - (sum(decided) / max(len(decided), 1)),
        'decided_accuracy': decided_acc,
        'margin_distribution': {
            'min': float(np.min(margins)) if margins else None,
            'p25': float(np.quantile(margins, 0.25)) if margins else None,
            'median': float(np.quantile(margins, 0.50)) if margins else None,
            'p75': float(np.quantile(margins, 0.75)) if margins else None,
            'max': float(np.max(margins)) if margins else None,
        },
        'latency_ms': {
            'mean': float(np.mean(latencies)) if latencies else None,
            'median': float(np.median(latencies)) if latencies else None,
            'p95': float(np.quantile(latencies, 0.95)) if latencies else None,
        },
        'confusion_matrix': {'labels': labels, 'matrix': cm.tolist()},
        'rows': rows,
    }


def evaluate_gallery_images(images_dir: str, model: Dict[str, Any]) -> Dict[str, Any]:
    queries = [QuerySample(r.pattern_id, Path(r.path).name, image_to_gray(read_image(r.path))) for r in list_images(images_dir)]
    return evaluate_queries(queries, model)


def save_pickle(obj: Any, path: str) -> None:
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
