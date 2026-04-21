#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


BASE_ASSIGNABLE_TYPES = {
    "Church",
    "Fort",
    "Fortified Village",
    "Great Church",
    "Main Encampment",
    "Ruins",
    "Sorcerer's Rise",
    "Swamp",
    "Township",
    "Fortress",
    "Palace",
}
UNDER_ANCHOR_TYPE = "under_slot_gh_43_38"
STATUS_TYPES = {
    "Status_Poison",
    "Status_Bleed",
    "Status_Sleep",
    "Status_Lightning",
    "Status_Rot",
    "Status_Holy",
    "Status_Fire",
    "Status_Magic",
    "Status_Frost",
}

EXPAND_NORMAL: Dict[Tuple[str, str], str] = {
    ("Great Church", "Status_Sleep"): "GreatChurchTopRightSleep",
    ("Great Church", "Status_Lightning"): "GreatChurchTopRightLightning",
    ("Ruins", "Status_Poison"): "RuinsPoison",
    ("Ruins", "Status_Bleed"): "RuinsBleed",
    ("Fortress", "Status_Rot"): "FortressRot",
    ("Fortress", "Status_Holy"): "FortressHoly",
    ("Palace", "Status_Fire"): "PalaceFire",
    ("Palace", "Status_Magic"): "PalaceMagic",
}
EXPAND_UNDER: Dict[Tuple[str, str], str] = {
    ("Great Church", "Status_Holy"): "GreatChurchUnderHoly",
    ("Great Church", "Status_Frost"): "GreatChurchUnderFrost",
}

SLOT_CLUSTER_RADIUS_NORM = 0.018
SLOT_ASSIGN_TOLERANCE_FACTOR = 1.55
SLOT_MIN_RADIUS_NORM = 0.018
SLOT_MAX_RADIUS_NORM = 0.040


@dataclass
class SlotDef:
    slot_id: int
    x_norm: float
    y_norm: float
    radius_norm: float
    allowed_types: List[str]


@dataclass
class PatchSample:
    layout_number: int
    context: str
    slot_id: int
    label: str
    descriptor: np.ndarray


def load_annotations(pattern_dir: Path, suffix: str) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    pattern = f"pattern_*_Great Hollow{suffix}.json"
    for p in sorted(pattern_dir.glob(pattern)):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = int(obj.get("patternId", -1))
        if pid < 0:
            continue
        out[pid] = obj
    return out


def collect_slot_points(
    normal_ann: Dict[int, dict],
    under_ann: Dict[int, dict],
) -> List[dict]:
    points: List[dict] = []
    for context, ann_by_id in (("normal", normal_ann), ("under", under_ann)):
        for pid, ann in ann_by_id.items():
            for lm in ann.get("landmarks", []):
                t = str(lm.get("type", "")).strip()
                if t not in BASE_ASSIGNABLE_TYPES and t != UNDER_ANCHOR_TYPE:
                    continue
                x_norm = float(lm.get("x_norm", 0.0))
                y_norm = float(lm.get("y_norm", 0.0))
                points.append(
                    {
                        "layout_number": int(pid),
                        "context": context,
                        "type": t,
                        "x_norm": x_norm,
                        "y_norm": y_norm,
                    }
                )
    points.sort(key=lambda item: (float(item["y_norm"]), float(item["x_norm"])))
    return points


def _merge_close_slot_clusters(clusters: List[dict], radius: float) -> List[dict]:
    changed = True
    while changed:
        changed = False
        out: List[dict] = []
        used = [False] * len(clusters)
        for i, base in enumerate(clusters):
            if used[i]:
                continue
            used[i] = True
            points = list(base["points"])
            cx = float(base["x_norm"])
            cy = float(base["y_norm"])
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                other = clusters[j]
                dx = float(other["x_norm"]) - cx
                dy = float(other["y_norm"]) - cy
                if math.hypot(dx, dy) <= radius:
                    used[j] = True
                    points.extend(other["points"])
                    cx = float(sum(float(p["x_norm"]) for p in points) / len(points))
                    cy = float(sum(float(p["y_norm"]) for p in points) / len(points))
                    changed = True
            out.append({"x_norm": cx, "y_norm": cy, "points": points})
        clusters = out
    return clusters


def build_slots(points: Sequence[dict]) -> List[SlotDef]:
    clusters: List[dict] = []
    for pt in points:
        best_idx = None
        best_dist = None
        for idx, cluster in enumerate(clusters):
            dx = float(pt["x_norm"]) - float(cluster["x_norm"])
            dy = float(pt["y_norm"]) - float(cluster["y_norm"])
            dist = math.hypot(dx, dy)
            if dist <= SLOT_CLUSTER_RADIUS_NORM and (best_dist is None or dist < best_dist):
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            clusters.append(
                {
                    "x_norm": float(pt["x_norm"]),
                    "y_norm": float(pt["y_norm"]),
                    "points": [pt],
                }
            )
        else:
            cluster = clusters[best_idx]
            cluster["points"].append(pt)
            cluster["x_norm"] = float(
                sum(float(p["x_norm"]) for p in cluster["points"]) / len(cluster["points"])
            )
            cluster["y_norm"] = float(
                sum(float(p["y_norm"]) for p in cluster["points"]) / len(cluster["points"])
            )
    clusters = _merge_close_slot_clusters(clusters, SLOT_CLUSTER_RADIUS_NORM * 0.72)
    clusters.sort(key=lambda item: (float(item["y_norm"]), float(item["x_norm"])))

    slots: List[SlotDef] = []
    for idx, cluster in enumerate(clusters, start=1):
        cx = float(cluster["x_norm"])
        cy = float(cluster["y_norm"])
        dists = [
            math.hypot(float(p["x_norm"]) - cx, float(p["y_norm"]) - cy)
            for p in cluster["points"]
        ]
        max_dist = max(dists) if dists else 0.0
        radius_norm = min(SLOT_MAX_RADIUS_NORM, max(SLOT_MIN_RADIUS_NORM, max_dist * 1.35 + 0.004))
        allowed_types = sorted(
            {
                str(p["type"])
                for p in cluster["points"]
                if str(p["type"]) != UNDER_ANCHOR_TYPE
            }
        )
        slots.append(
            SlotDef(
                slot_id=idx,
                x_norm=cx,
                y_norm=cy,
                radius_norm=float(radius_norm),
                allowed_types=allowed_types,
            )
        )
    return slots


def assign_landmarks_to_slots(landmarks: Sequence[dict], slots: Sequence[SlotDef]) -> Dict[int, dict]:
    base_landmarks = []
    for lm in landmarks:
        t = str(lm.get("type", "")).strip()
        if t not in BASE_ASSIGNABLE_TYPES:
            continue
        base_landmarks.append(lm)

    candidates: List[Tuple[float, int, int]] = []
    for lm_idx, lm in enumerate(base_landmarks):
        lx = float(lm.get("x_norm", 0.0))
        ly = float(lm.get("y_norm", 0.0))
        for slot_idx, slot in enumerate(slots):
            dx = lx - float(slot.x_norm)
            dy = ly - float(slot.y_norm)
            dist = math.hypot(dx, dy)
            tol = float(slot.radius_norm) * SLOT_ASSIGN_TOLERANCE_FACTOR
            if dist <= tol:
                candidates.append((dist, lm_idx, slot_idx))
    candidates.sort(key=lambda item: item[0])

    used_landmarks = set()
    used_slots = set()
    out: Dict[int, dict] = {}
    for _, lm_idx, slot_idx in candidates:
        if lm_idx in used_landmarks or slot_idx in used_slots:
            continue
        used_landmarks.add(lm_idx)
        used_slots.add(slot_idx)
        out[int(slots[slot_idx].slot_id)] = base_landmarks[lm_idx]

    for lm_idx, lm in enumerate(base_landmarks):
        if lm_idx in used_landmarks:
            continue
        lx = float(lm.get("x_norm", 0.0))
        ly = float(lm.get("y_norm", 0.0))
        best_slot = None
        best_dist = None
        for slot_idx, slot in enumerate(slots):
            if slot_idx in used_slots:
                continue
            dist = math.hypot(lx - float(slot.x_norm), ly - float(slot.y_norm))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_slot = slot_idx
        if best_slot is not None:
            used_slots.add(best_slot)
            out[int(slots[best_slot].slot_id)] = lm
    return out


def status_by_source_id(landmarks: Sequence[dict]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for lm in landmarks:
        t = str(lm.get("type", "")).strip()
        if t not in STATUS_TYPES:
            continue
        source_id = lm.get("source_id")
        if source_id is None:
            continue
        out[int(source_id)] = t
    return out


def expand_label(base_type: str, status_type: str | None, context: str) -> str:
    if status_type:
        key = (str(base_type), str(status_type))
        if context == "normal" and key in EXPAND_NORMAL:
            return EXPAND_NORMAL[key]
        if context == "under" and key in EXPAND_UNDER:
            return EXPAND_UNDER[key]
    return str(base_type)


def build_slot_labels_for_annotation(annotation: dict, slots: Sequence[SlotDef], context: str) -> Dict[int, str]:
    labels: Dict[int, str] = {int(s.slot_id): "empty" for s in slots}
    landmarks = list(annotation.get("landmarks", []))
    assigned = assign_landmarks_to_slots(landmarks, slots)
    status_map = status_by_source_id(landmarks)

    for slot_id, lm in assigned.items():
        base_type = str(lm.get("type", "")).strip()
        source_id = lm.get("source_id")
        status_type = None
        if source_id is not None:
            status_type = status_map.get(int(source_id))
        labels[int(slot_id)] = expand_label(base_type, status_type, context)
    return labels


def extract_patch_reflect(img_bgr: np.ndarray, cx: int, cy: int, side: int) -> np.ndarray:
    half = side // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + side
    y1 = y0 + side
    h, w = img_bgr.shape[:2]
    xx0, yy0 = max(0, x0), max(0, y0)
    xx1, yy1 = min(w, x1), min(h, y1)
    patch = img_bgr[yy0:yy1, xx0:xx1].copy()
    top = yy0 - y0
    left = xx0 - x0
    bottom = y1 - yy1
    right = x1 - xx1
    if top or left or bottom or right:
        patch = cv2.copyMakeBorder(
            patch,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT_101,
        )
    return patch


def compute_descriptor(img_bgr: np.ndarray) -> np.ndarray:
    work = cv2.resize(img_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    thumb_size = 24
    small = cv2.resize(gray, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    small = (small - float(small.mean())) / (float(small.std()) + 1e-6)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge = cv2.resize(mag, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    edge = (edge - float(edge.mean())) / (float(edge.std()) + 1e-6)

    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten().astype(np.float32)
    hist /= float(hist.sum()) + 1e-6

    descriptor = np.concatenate([0.55 * small, 0.30 * edge, 3.00 * hist]).astype(np.float32)
    descriptor /= float(np.linalg.norm(descriptor)) + 1e-6
    return descriptor


def build_patch_samples(
    slots: Sequence[SlotDef],
    normal_ann: Dict[int, dict],
    under_ann: Dict[int, dict],
    normal_img_dir: Path,
    under_img_dir: Path,
) -> List[PatchSample]:
    samples: List[PatchSample] = []
    for context, ann_by_id, image_dir in (
        ("normal", normal_ann, normal_img_dir),
        ("under", under_ann, under_img_dir),
    ):
        for layout_number, ann in ann_by_id.items():
            image_path = image_dir / f"layout_{int(layout_number):04d}.png"
            if not image_path.exists():
                continue
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            side_min = min(h, w)
            slot_labels = build_slot_labels_for_annotation(ann, slots, context)
            for slot in slots:
                cx = int(round(float(slot.x_norm) * w))
                cy = int(round(float(slot.y_norm) * h))
                cx = max(0, min(w - 1, cx))
                cy = max(0, min(h - 1, cy))
                half = int(round(max(float(slot.radius_norm) * side_min * 1.2, 0.075 * side_min)))
                half = max(18, half)
                patch = extract_patch_reflect(img, cx, cy, 2 * half)
                patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)
                desc = compute_descriptor(patch)
                label = str(slot_labels.get(int(slot.slot_id), "empty"))
                samples.append(
                    PatchSample(
                        layout_number=int(layout_number),
                        context=context,
                        slot_id=int(slot.slot_id),
                        label=label,
                        descriptor=desc,
                    )
                )
    return samples


def split_layouts(
    layout_numbers: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    ids = sorted({int(x) for x in layout_numbers})
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * float(train_ratio))))
    n_val = max(1, int(round(n * float(val_ratio))))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
        n_train = max(1, n - n_val - 1)
    train = set(ids[:n_train])
    val = set(ids[n_train:n_train + n_val])
    test = set(ids[n_train + n_val:])
    if not test:
        last = max(train)
        train.remove(last)
        test.add(last)
    return train, val, test


def build_centroid_model(samples: Sequence[PatchSample]) -> Dict[int, Dict[str, np.ndarray]]:
    by_slot_label: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_slot_label[int(s.slot_id)][str(s.label)].append(s.descriptor.astype(np.float32))
    model: Dict[int, Dict[str, np.ndarray]] = {}
    for slot_id, by_label in by_slot_label.items():
        slot_model: Dict[str, np.ndarray] = {}
        for label, vecs in by_label.items():
            mat = np.stack(vecs, axis=0).astype(np.float32)
            centroid = mat.mean(axis=0).astype(np.float32)
            centroid /= float(np.linalg.norm(centroid)) + 1e-6
            slot_model[str(label)] = centroid
        model[int(slot_id)] = slot_model
    return model


def predict_one(
    model: Dict[int, Dict[str, np.ndarray]],
    sample: PatchSample,
    unknown_margin: float,
) -> Dict[str, object]:
    slot_model = model.get(int(sample.slot_id), {})
    if not slot_model:
        return {
            "pred_forced": "empty",
            "pred_unknown": "unknown",
            "best_score": -1.0,
            "second_score": -1.0,
            "margin": 0.0,
        }
    sims = []
    x = sample.descriptor.astype(np.float32)
    for label, centroid in slot_model.items():
        score = float(np.dot(centroid, x))
        sims.append((label, score))
    sims.sort(key=lambda kv: kv[1], reverse=True)
    best_label, best_score = sims[0]
    second_score = sims[1][1] if len(sims) > 1 else -1.0
    margin = float(best_score - second_score)
    pred_unknown = "unknown" if margin < float(unknown_margin) else str(best_label)
    return {
        "pred_forced": str(best_label),
        "pred_unknown": str(pred_unknown),
        "best_score": float(best_score),
        "second_score": float(second_score),
        "margin": float(margin),
    }


def evaluate_split(
    model: Dict[int, Dict[str, np.ndarray]],
    samples: Sequence[PatchSample],
    unknown_margin: float,
) -> Dict[str, object]:
    rows = []
    correct_forced = 0
    correct_unknown = 0
    decided = 0
    per_label = Counter()
    per_label_correct_forced = Counter()
    per_label_correct_unknown = Counter()
    per_label_decided = Counter()

    for s in samples:
        pred = predict_one(model, s, unknown_margin=unknown_margin)
        y = str(s.label)
        pf = str(pred["pred_forced"])
        pu = str(pred["pred_unknown"])
        correct_f = int(pf == y)
        correct_u = int(pu == y)
        is_decided = int(pu != "unknown")

        correct_forced += correct_f
        correct_unknown += correct_u
        decided += is_decided

        per_label[y] += 1
        per_label_correct_forced[y] += correct_f
        per_label_correct_unknown[y] += correct_u
        per_label_decided[y] += is_decided

        rows.append(
            {
                "layout_number": int(s.layout_number),
                "context": str(s.context),
                "slot_id": int(s.slot_id),
                "label_true": y,
                "label_pred_forced": pf,
                "label_pred_unknown": pu,
                "best_score": float(pred["best_score"]),
                "second_score": float(pred["second_score"]),
                "margin": float(pred["margin"]),
            }
        )

    n = max(1, len(samples))
    per_label_metrics = {}
    for label in sorted(per_label.keys()):
        count = int(per_label[label])
        per_label_metrics[label] = {
            "count": count,
            "forced_acc": float(per_label_correct_forced[label] / max(1, count)),
            "unknown_acc": float(per_label_correct_unknown[label] / max(1, count)),
            "coverage": float(per_label_decided[label] / max(1, count)),
        }

    return {
        "count": int(len(samples)),
        "forced_accuracy": float(correct_forced / n),
        "unknown_accuracy": float(correct_unknown / n),
        "decided_coverage": float(decided / n),
        "unknown_accuracy_on_decided": float(correct_unknown / max(1, decided)),
        "per_label": per_label_metrics,
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Experimental Great Hollow status-aware slot classifier test. "
            "Uses context-specific Great Church labels (TopRight Sleep/Lightning vs Under Holy/Frost)."
        )
    )
    p.add_argument("--normal-pattern-dir", type=Path, default=Path("renders_new/no_text"))
    p.add_argument("--normal-image-dir", type=Path, default=Path("renders/recognizer_no_text"))
    p.add_argument("--under-pattern-dir", type=Path, default=Path("renders_new_under/no_text"))
    p.add_argument("--under-image-dir", type=Path, default=Path("renders/recognizer_no_text_under"))
    p.add_argument("--out-dir", type=Path, default=Path("tmp_gh_status_aware_slot_classifier_test"))
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--unknown-margin", type=float, default=0.03)
    args = p.parse_args()

    normal_ann = load_annotations(Path(args.normal_pattern_dir), suffix="")
    under_ann = load_annotations(Path(args.under_pattern_dir), suffix="_under")
    common_layouts = sorted(set(normal_ann.keys()) & set(under_ann.keys()))
    normal_ann = {k: normal_ann[k] for k in common_layouts}
    under_ann = {k: under_ann[k] for k in common_layouts}
    if not common_layouts:
        raise RuntimeError("No overlapping Great Hollow layouts found between normal and under annotations.")

    points = collect_slot_points(normal_ann, under_ann)
    slots = build_slots(points)
    samples = build_patch_samples(
        slots=slots,
        normal_ann=normal_ann,
        under_ann=under_ann,
        normal_img_dir=Path(args.normal_image_dir),
        under_img_dir=Path(args.under_image_dir),
    )
    if not samples:
        raise RuntimeError("No patch samples built.")

    train_ids, val_ids, test_ids = split_layouts(
        layout_numbers=common_layouts,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    train_samples = [s for s in samples if int(s.layout_number) in train_ids]
    val_samples = [s for s in samples if int(s.layout_number) in val_ids]
    test_samples = [s for s in samples if int(s.layout_number) in test_ids]

    model = build_centroid_model(train_samples)
    train_metrics = evaluate_split(model, train_samples, unknown_margin=float(args.unknown_margin))
    val_metrics = evaluate_split(model, val_samples, unknown_margin=float(args.unknown_margin))
    test_metrics = evaluate_split(model, test_samples, unknown_margin=float(args.unknown_margin))

    label_counts = Counter(str(s.label) for s in samples)
    status_aware_labels = sorted(
        [
            k
            for k in label_counts.keys()
            if k.startswith("GreatChurchTopRight")
            or k.startswith("GreatChurchUnder")
            or k.startswith("Ruins")
            or k.startswith("Fortress")
            or k.startswith("Palace")
        ]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "normal_pattern_dir": str(args.normal_pattern_dir),
            "normal_image_dir": str(args.normal_image_dir),
            "under_pattern_dir": str(args.under_pattern_dir),
            "under_image_dir": str(args.under_image_dir),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "unknown_margin": float(args.unknown_margin),
        },
        "counts": {
            "layouts": int(len(common_layouts)),
            "slots": int(len(slots)),
            "samples_total": int(len(samples)),
            "samples_train": int(len(train_samples)),
            "samples_val": int(len(val_samples)),
            "samples_test": int(len(test_samples)),
            "labels_total": dict(sorted(label_counts.items())),
        },
        "status_aware_labels_in_use": status_aware_labels,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "slot_defs": [
            {
                "slot_id": int(s.slot_id),
                "x_norm": float(s.x_norm),
                "y_norm": float(s.y_norm),
                "radius_norm": float(s.radius_norm),
                "allowed_types": list(s.allowed_types),
            }
            for s in slots
        ],
    }
    report_path = out_dir / "gh_status_aware_slot_classifier_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"saved report: {report_path}")
    print(
        json.dumps(
            {
                "labels_status_aware": status_aware_labels,
                "test_forced_accuracy": float(test_metrics["forced_accuracy"]),
                "test_unknown_accuracy": float(test_metrics["unknown_accuracy"]),
                "test_coverage": float(test_metrics["decided_coverage"]),
                "test_unknown_accuracy_on_decided": float(test_metrics["unknown_accuracy_on_decided"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
