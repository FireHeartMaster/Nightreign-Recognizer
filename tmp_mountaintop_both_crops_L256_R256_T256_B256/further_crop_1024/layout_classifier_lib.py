
import argparse
import json
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

LOGGER = logging.getLogger("layout_classifier")
IMAGE_W = 1024
IMAGE_H = 1024
PATTERN_RE = re.compile(r"pattern_(\d+)_")

@dataclass
class ImageRecord:
    pattern_id: int
    path: str


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='[%(levelname)s] %(message)s')


def parse_pattern_id(filename: str) -> int:
    m = PATTERN_RE.search(os.path.basename(filename))
    if not m:
        raise ValueError(f'Could not parse patternId from filename: {filename}')
    return int(m.group(1))


def list_images(images_dir: str) -> List[ImageRecord]:
    records = [ImageRecord(parse_pattern_id(p.name), str(p)) for p in sorted(Path(images_dir).glob('*.png'))]
    if not records:
        raise FileNotFoundError(f'No PNG images found in {images_dir}')
    return records


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'layouts' not in data:
        raise KeyError("Metadata JSON must contain 'layouts'")
    return data


def _norm_to_px(norm_value: float, size: int) -> int:
    return int(round(float(norm_value) * (size - 1)))


def metadata_point_to_image_px(point: Dict[str, Any], image_w: int = IMAGE_W, image_h: int = IMAGE_H) -> Tuple[int, int, str]:
    if 'x_further_px_bl' in point and 'y_further_px_bl' in point:
        x_bl = float(point['x_further_px_bl'])
        y_bl = float(point['y_further_px_bl'])
        src = 'px_bl'
    elif 'x_further_norm_bl' in point and 'y_further_norm_bl' in point:
        x_bl = _norm_to_px(float(point['x_further_norm_bl']), image_w)
        y_bl = _norm_to_px(float(point['y_further_norm_bl']), image_h)
        src = 'norm_bl'
    else:
        raise KeyError(f'Unsupported coordinate fields: {sorted(point.keys())}')
    x_px = int(round(x_bl))
    y_px = (image_h - 1) - int(round(y_bl))
    return x_px, y_px, src


def validate_and_collect_layout_points(metadata: Dict[str, Any], image_w: int = IMAGE_W, image_h: int = IMAGE_H):
    layouts_by_id = {}
    out_of_bounds = []
    coord_sources = {}
    for layout in metadata['layouts']:
        pid = int(layout['patternId'])
        norm = {'patternId': pid, 'slots': [], 'landmarks': []}
        for key in ('slots', 'landmarks'):
            for idx, point in enumerate(layout.get(key, [])):
                x_px, y_px, src = metadata_point_to_image_px(point, image_w, image_h)
                coord_sources[src] = coord_sources.get(src, 0) + 1
                if not (0 <= x_px < image_w and 0 <= y_px < image_h):
                    out_of_bounds.append({'patternId': pid, 'kind': key, 'index': idx, 'x_px': x_px, 'y_px': y_px})
                    LOGGER.warning('Metadata point outside image for patternId=%s kind=%s index=%s x=%s y=%s', pid, key, idx, x_px, y_px)
                    continue
                enriched = dict(point)
                enriched['x_px'] = x_px
                enriched['y_px'] = y_px
                enriched['coord_source'] = src
                norm[key].append(enriched)
        layouts_by_id[pid] = norm
    return layouts_by_id, {'coord_sources': coord_sources, 'out_of_bounds': out_of_bounds, 'image_w': image_w, 'image_h': image_h, 'layouts_count': len(layouts_by_id)}


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    h, w = img.shape[:2]
    if (w, h) != (IMAGE_W, IMAGE_H):
        raise ValueError(f'Unexpected image size for {path}: {(w,h)} expected {(IMAGE_W, IMAGE_H)}')
    return img


def image_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def compute_hog_descriptor(gray: np.ndarray, resize_to: Tuple[int, int] = (384, 384)) -> np.ndarray:
    resized = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(resize_to, (32, 32), (16, 16), (16, 16), 9)
    feat = hog.compute(resized).reshape(-1).astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-8)


def extract_patch(gray: np.ndarray, center_xy: Tuple[int, int], patch_size: int) -> np.ndarray:
    x, y = center_xy
    half = patch_size // 2
    left, top = x - half, y - half
    right, bottom = left + patch_size, top + patch_size
    src_left, src_right = max(0, left), min(gray.shape[1], right)
    src_top, src_bottom = max(0, top), min(gray.shape[0], bottom)
    patch = np.zeros((patch_size, patch_size), dtype=gray.dtype)
    dst_left, dst_top = src_left - left, src_top - top
    patch[dst_top:dst_top + (src_bottom - src_top), dst_left:dst_left + (src_right - src_left)] = gray[src_top:src_bottom, src_left:src_right]
    return patch


def compute_patch_descriptor(gray: np.ndarray, center_xy: Tuple[int, int], patch_size: int = 72, descriptor_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    return compute_hog_descriptor(extract_patch(gray, center_xy, patch_size), resize_to=descriptor_size)


def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return query @ gallery.T


def build_gallery(images_dir: str, metadata_path: str) -> Dict[str, Any]:
    metadata = load_metadata(metadata_path)
    layouts_by_id, diagnostics = validate_and_collect_layout_points(metadata)
    records = list_images(images_dir)
    pattern_ids, image_paths = [], []
    image_features, slot_features, landmark_features = [], [], []
    missing = []
    for rec in records:
        layout = layouts_by_id.get(rec.pattern_id)
        if layout is None:
            missing.append(rec.pattern_id)
            continue
        gray = image_to_gray(read_image(rec.path))
        pattern_ids.append(rec.pattern_id)
        image_paths.append(rec.path)
        image_features.append(compute_hog_descriptor(gray))
        slot_features.append(np.stack([compute_patch_descriptor(gray, (int(p['x_px']), int(p['y_px'])), patch_size=72) for p in layout['slots']], axis=0))
        landmark_descs = [compute_patch_descriptor(gray, (int(p['x_px']), int(p['y_px'])), patch_size=64) for p in layout['landmarks']]
        landmark_features.append(np.stack(landmark_descs, axis=0) if landmark_descs else np.zeros((0, 576), dtype=np.float32))
    if missing:
        raise ValueError(f'Missing layouts in metadata for patternIds: {missing}')
    return {
        'pattern_ids': np.array(pattern_ids, dtype=np.int32),
        'image_paths': image_paths,
        'image_features': np.stack(image_features, axis=0),
        'slot_features': np.stack(slot_features, axis=0),
        'landmark_features': landmark_features,
        'layouts_by_id': layouts_by_id,
        'metadata_diagnostics': diagnostics,
        'config': {'image_size':[IMAGE_W, IMAGE_H], 'image_descriptor_resize':[384,384], 'slot_patch_size':72, 'landmark_patch_size':64},
    }


def standardize_scores(scores: np.ndarray) -> np.ndarray:
    mu, sigma = float(np.mean(scores)), float(np.std(scores))
    return scores - mu if sigma < 1e-8 else (scores - mu) / sigma


def compute_raw_scores(gray: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    image_feat = compute_hog_descriptor(gray)
    image_scores = cosine_similarity_matrix(image_feat[None, :], model['image_features'])[0]
    first_layout = model['layouts_by_id'][int(model['pattern_ids'][0])]
    slot_descs = np.stack([compute_patch_descriptor(gray, (int(p['x_px']), int(p['y_px'])), patch_size=72) for p in first_layout['slots']], axis=0)
    slot_scores = np.mean(np.sum(slot_descs[None,:,:] * model['slot_features'], axis=2), axis=1)
    landmark_scores = []
    for idx, pid in enumerate(model['pattern_ids']):
        layout = model['layouts_by_id'][int(pid)]
        q_descs = [compute_patch_descriptor(gray, (int(p['x_px']), int(p['y_px'])), patch_size=64) for p in layout['landmarks']]
        q_descs = np.stack(q_descs, axis=0) if q_descs else np.zeros((0, model['landmark_features'][idx].shape[1]), dtype=np.float32)
        if q_descs.shape[0] == 0 or model['landmark_features'][idx].shape[0] == 0:
            landmark_scores.append(0.0)
        else:
            landmark_scores.append(float(np.mean(np.sum(q_descs * model['landmark_features'][idx], axis=1))))
    landmark_scores = np.asarray(landmark_scores, dtype=np.float32)
    slot_aware_scores = 0.7 * slot_scores + 0.3 * landmark_scores
    return {'image_scores': image_scores, 'slot_scores': slot_aware_scores}


def finalize_scores(raw_scores: Dict[str, Any], model: Dict[str, Any], alpha: float = 0.60, temperature: float = 0.12) -> Dict[str, Any]:
    image_z = standardize_scores(raw_scores['image_scores'])
    slot_z = standardize_scores(raw_scores['slot_scores'])
    hybrid = alpha * image_z + (1-alpha) * slot_z
    logits = hybrid / max(temperature, 1e-6)
    logits = logits - np.max(logits)
    probs = np.exp(logits); probs = probs / np.sum(probs)
    order = np.argsort(-hybrid)
    top1 = order[0]; top2 = order[1] if len(order) > 1 else order[0]
    return {'pattern_ids': model['pattern_ids'], 'image_scores': raw_scores['image_scores'], 'slot_scores': raw_scores['slot_scores'], 'hybrid_scores': hybrid, 'probabilities': probs, 'ranking': order, 'best_pattern_id': int(model['pattern_ids'][top1]), 'margin': float(hybrid[top1] - hybrid[top2])}


def predict_scores(gray: np.ndarray, model: Dict[str, Any], alpha: float, temperature: float) -> Dict[str, Any]:
    return finalize_scores(compute_raw_scores(gray, model), model, alpha=alpha, temperature=temperature)


def calibrate_model(model: Dict[str, Any], images_dir: str, seed: int = 1337) -> Dict[str, Any]:
    del seed
    margins = []
    for rec in list_images(images_dir):
        gray = image_to_gray(read_image(rec.path))
        pred = predict_scores(gray, model, alpha=0.60, temperature=0.12)
        margins.append(pred['margin'])
    margins = np.asarray(margins, dtype=np.float32)
    threshold = float(np.quantile(margins, 0.10)) if len(margins) else 0.0
    return {'alpha':0.60, 'temperature':0.12, 'unknown_margin_threshold':threshold, 'calibration_note':'Heuristic calibration from self-retrieval margins; one template per class.'}


def classify_image(image_path: str, model: Dict[str, Any]) -> Dict[str, Any]:
    gray = image_to_gray(read_image(image_path))
    start = time.perf_counter()
    pred = predict_scores(gray, model, alpha=model['calibration']['alpha'], temperature=model['calibration']['temperature'])
    latency_ms = (time.perf_counter() - start) * 1000.0
    top5 = []
    for idx in pred['ranking'][:5]:
        top5.append({'patternId': int(model['pattern_ids'][idx]), 'score': float(pred['hybrid_scores'][idx]), 'probability': float(pred['probabilities'][idx]), 'image_score': float(pred['image_scores'][idx]), 'slot_score': float(pred['slot_scores'][idx])})
    unknown = pred['margin'] < model['calibration']['unknown_margin_threshold']
    return {'input': image_path, 'best_patternId': int(pred['best_pattern_id']), 'top5': top5, 'margin': float(pred['margin']), 'confidence': float(top5[0]['probability']), 'unknown': bool(unknown), 'decided_patternId': None if unknown else int(pred['best_pattern_id']), 'latency_ms': float(latency_ms)}


def compute_topk_accuracy(y_true: Sequence[int], ranked_predictions: Sequence[Sequence[int]], k: int) -> float:
    return sum(1 for t, r in zip(y_true, ranked_predictions) if t in list(r)[:k]) / max(len(y_true), 1)


def evaluate_images(images_dir: str, model: Dict[str, Any]) -> Dict[str, Any]:
    y_true=[]; y_pred=[]; ranked=[]; margins=[]; decided=[]; latencies=[]; rows=[]
    for rec in list_images(images_dir):
        out = classify_image(rec.path, model)
        rank_ids = [x['patternId'] for x in out['top5']] + [int(model['pattern_ids'][i]) for i in predict_scores(image_to_gray(read_image(rec.path)), model, alpha=model['calibration']['alpha'], temperature=model['calibration']['temperature'])['ranking'][5:]]
        y_true.append(rec.pattern_id); y_pred.append(out['best_patternId']); ranked.append(rank_ids); margins.append(out['margin']); decided.append(not out['unknown']); latencies.append(out['latency_ms'])
        rows.append({'image': os.path.basename(rec.path), 'truth': rec.pattern_id, 'pred': out['best_patternId'], 'margin': out['margin'], 'decided': not out['unknown'], 'latency_ms': out['latency_ms']})
    labels = [int(x) for x in model['pattern_ids']]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    decided_idx = [i for i, f in enumerate(decided) if f]
    decided_acc = float(np.mean([y_true[i] == y_pred[i] for i in decided_idx])) if decided_idx else None
    return {'count': len(y_true), 'top1_accuracy': compute_topk_accuracy(y_true, ranked, 1), 'top3_accuracy': compute_topk_accuracy(y_true, ranked, 3), 'top5_accuracy': compute_topk_accuracy(y_true, ranked, 5), 'unknown_rate': 1.0 - (sum(decided)/max(len(decided),1)), 'decided_accuracy': decided_acc, 'margin_distribution': {'min': float(np.min(margins)), 'p25': float(np.quantile(margins,0.25)), 'median': float(np.quantile(margins,0.5)), 'p75': float(np.quantile(margins,0.75)), 'max': float(np.max(margins))}, 'latency_ms': {'mean': float(np.mean(latencies)), 'median': float(np.median(latencies)), 'p95': float(np.quantile(latencies,0.95))}, 'confusion_matrix': {'labels': labels, 'matrix': cm.tolist()}, 'rows': rows}


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
