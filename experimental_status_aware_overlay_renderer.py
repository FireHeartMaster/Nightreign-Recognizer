#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import experimental_great_hollow_status_aware_slot_classifier_test as exp
import nightreign_map_recognizer_gui_ai_v37_completion_toast_overlay as gui


STATUS_AWARE_PREFIXES = (
    "GreatChurchTopRight",
    "GreatChurchUnder",
    "Ruins",
    "Fortress",
    "Palace",
)


def is_status_aware_label(label: str) -> bool:
    return str(label).startswith(STATUS_AWARE_PREFIXES)


def detect_map_circle_compat(backend, ctx, img_bgr: np.ndarray):
    region = getattr(backend, "DEFAULT_HOUGH_REGION", "right_half")
    scale = float(getattr(backend, "DEFAULT_HOUGH_SCALE", 1.0))
    try:
        return backend.detect_map_circle(img_bgr, region, scale, ctx)
    except TypeError:
        try:
            return backend.detect_map_circle(img_bgr, region, scale)
        except TypeError:
            return backend.detect_map_circle(img_bgr)


def refine_panel_box_compat(backend, ctx, img_bgr: np.ndarray, circle_obj):
    panel_scale = float(getattr(backend, "DEFAULT_PANEL_SCALE", 1.12))
    search_radius = int(getattr(backend, "DEFAULT_SEARCH_RADIUS", 80))
    pad = int(getattr(backend, "DEFAULT_PAD", 0))
    try:
        return backend.refine_panel_box(img_bgr, circle_obj, panel_scale, search_radius, pad, ctx)
    except TypeError:
        try:
            return backend.refine_panel_box(img_bgr, circle_obj, panel_scale, search_radius, pad)
        except TypeError:
            return backend.refine_panel_box(img_bgr, circle_obj)


def normalize_panel_for_slot_recognition_compat(backend, raw_panel_bgr: np.ndarray):
    fn = getattr(backend, "normalize_panel_for_slot_recognition", None)
    if fn is None:
        fn = getattr(backend, "normalize_panel_for_recognition", None)
    if fn is None:
        raise AttributeError(
            f"module '{backend.__name__}' has neither normalize_panel_for_slot_recognition "
            "nor normalize_panel_for_recognition"
        )
    return fn(raw_panel_bgr)


def refine_crop_box_by_earth_score_slot_compat(backend, ctx, full_img_bgr, initial_box, package_obj):
    fn = getattr(backend, "refine_crop_box_by_earth_score_slot", None)
    if fn is not None:
        try:
            return fn(full_img_bgr, initial_box, package_obj, ctx)
        except TypeError:
            return fn(full_img_bgr, initial_box, package_obj)
    fn = getattr(backend, "refine_crop_box_by_earth_score", None)
    if fn is not None:
        try:
            return fn(full_img_bgr, initial_box, package_obj, ctx)
        except TypeError:
            return fn(full_img_bgr, initial_box, package_obj)
    raise AttributeError(
        f"module '{backend.__name__}' has neither refine_crop_box_by_earth_score_slot nor refine_crop_box_by_earth_score"
    )


def slot_probs_for_patch(slot_model: Dict[str, np.ndarray], desc: np.ndarray, temp: float = 0.08) -> Dict[str, float]:
    if not slot_model:
        return {"empty": 1.0}
    labels = list(slot_model.keys())
    sims = np.array([float(np.dot(desc, slot_model[l])) for l in labels], dtype=np.float64)
    sims = sims / max(1e-6, float(temp))
    m = float(np.max(sims))
    ex = np.exp(sims - m)
    probs = ex / (float(np.sum(ex)) + 1e-12)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


def load_sparse_slot_model(path: Path) -> Dict[int, dict]:
    blob = np.load(str(path), allow_pickle=True)
    payload_json = str(blob["payload_json"].tolist())
    payload = json.loads(payload_json)
    out: Dict[int, dict] = {}
    for sid_str, row in payload.items():
        sid = int(sid_str)
        out[sid] = {
            "labels": [str(x) for x in row["labels"]],
            "mean": np.asarray(row["mean"], dtype=np.float32),
            "std": np.asarray(row["std"], dtype=np.float32),
            "weight": np.asarray(row["weight"], dtype=np.float32),
            "bias": np.asarray(row["bias"], dtype=np.float32),
        }
    return out


def sparse_slot_probs_for_patch(slot_model: dict, desc: np.ndarray, temp: float = 0.55) -> Dict[str, float]:
    if not slot_model:
        return {"empty": 1.0}
    labels: List[str] = list(slot_model["labels"])
    mean = slot_model["mean"]
    std = slot_model["std"]
    weight = slot_model["weight"]
    bias = slot_model["bias"]
    x = desc.astype(np.float32)
    xz = (x - mean) / np.maximum(std, 1e-6)
    logits = np.dot(weight, xz) + bias
    logits = logits / max(1e-6, float(temp))
    m = float(np.max(logits))
    ex = np.exp(logits - m)
    probs = ex / (float(np.sum(ex)) + 1e-12)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


def infer_slots_on_panel(
    panel_bgr: np.ndarray,
    slots: List[exp.SlotDef],
    model: Dict[int, Dict[str, np.ndarray]],
    sparse_model: Optional[Dict[int, dict]] = None,
    sparse_temperature: float = 0.55,
) -> Dict[int, dict]:
    h, w = panel_bgr.shape[:2]
    side_min = min(h, w)
    obs: Dict[int, dict] = {}
    for slot in slots:
        sid = int(slot.slot_id)
        cx = int(round(float(slot.x_norm) * w))
        cy = int(round(float(slot.y_norm) * h))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        half = int(round(max(float(slot.radius_norm) * side_min * 1.2, 0.075 * side_min)))
        half = max(18, half)
        patch = exp.extract_patch_reflect(panel_bgr, cx, cy, 2 * half)
        patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)
        desc = exp.compute_descriptor(patch)
        if sparse_model is not None:
            probs = sparse_slot_probs_for_patch(
                sparse_model.get(sid, {}),
                desc,
                temp=sparse_temperature,
            )
        else:
            probs = slot_probs_for_patch(model.get(sid, {}), desc, temp=0.08)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        obs[sid] = {
            "slot_id": sid,
            "x_px": cx,
            "y_px": cy,
            "radius_px": half,
            "best_label": str(ranked[0][0]),
            "best_prob": float(ranked[0][1]),
            "second_label": str(ranked[1][0]) if len(ranked) > 1 else "",
            "second_prob": float(ranked[1][1]) if len(ranked) > 1 else 0.0,
            "probs": probs,
        }
    return obs


def score_layout(obs: Dict[int, dict], expected: Dict[int, str]) -> float:
    eps = 1e-9
    total = 0.0
    count = 0
    for sid, item in obs.items():
        exp_label = str(expected.get(int(sid), "empty"))
        p = float(item["probs"].get(exp_label, eps))
        total += math.log(max(eps, p))
        count += 1
    return float(total / max(1, count))


def draw_status_aware_overlay(panel_bgr: np.ndarray, obs: Dict[int, dict], expected: Dict[int, str]) -> np.ndarray:
    out = panel_bgr.copy()
    for sid in sorted(obs.keys()):
        item = obs[sid]
        x = int(item["x_px"])
        y = int(item["y_px"])
        r = max(8, int(item["radius_px"]))
        pred = str(item["best_label"])
        exp_lbl = str(expected.get(int(sid), "empty"))
        p = float(item["best_prob"])
        p2 = float(item["second_prob"])

        if pred == "empty":
            color = (170, 170, 170)
        elif is_status_aware_label(pred):
            color = (0, 255, 0)
        else:
            color = (0, 200, 255)

        cv2.circle(out, (x, y), r, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 5, color, -1, cv2.LINE_AA)

        txt1 = f"S{sid:02d} pred={pred} {p:.2f}"
        txt2 = f"exp={exp_lbl} 2nd={item['second_label']} {p2:.2f}"
        tx = max(6, min(out.shape[1] - 640, x + 8))
        ty = max(20, min(out.shape[0] - 20, y - 10))
        cv2.putText(out, txt1, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt1, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(out, txt2, (tx, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt2, (tx, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render status-aware overlay annotations for Great Hollow experimental classifier.")
    ap.add_argument("--input", nargs="+", required=True, help="Screenshot PNG path(s).")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp_status_aware_overlay_renderer"))
    ap.add_argument("--normal-pattern-dir", type=Path, default=Path("renders_new/no_text"))
    ap.add_argument("--normal-image-dir", type=Path, default=Path("renders/recognizer_no_text"))
    ap.add_argument("--under-pattern-dir", type=Path, default=Path("renders_new_under/no_text"))
    ap.add_argument("--under-image-dir", type=Path, default=Path("renders/recognizer_no_text_under"))
    ap.add_argument("--sparse-model", type=Path, default=None, help="Optional sparse slot model .npz")
    ap.add_argument("--sparse-temperature", type=float, default=0.55, help="Softmax temperature for sparse model")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_ann = exp.load_annotations(Path(args.normal_pattern_dir), suffix="")
    under_ann = exp.load_annotations(Path(args.under_pattern_dir), suffix="_under")
    common_layouts = sorted(set(normal_ann.keys()) & set(under_ann.keys()))
    normal_ann = {k: normal_ann[k] for k in common_layouts}
    under_ann = {k: under_ann[k] for k in common_layouts}
    if not common_layouts:
        raise RuntimeError("No overlapping Great Hollow layouts found in normal/under annotation dirs.")

    points = exp.collect_slot_points(normal_ann, under_ann)
    slots = exp.build_slots(points)
    samples = exp.build_patch_samples(
        slots=slots,
        normal_ann=normal_ann,
        under_ann=under_ann,
        normal_img_dir=Path(args.normal_image_dir),
        under_img_dir=Path(args.under_image_dir),
    )
    model = exp.build_centroid_model(samples)
    sparse_model = load_sparse_slot_model(Path(args.sparse_model)) if args.sparse_model else None
    layout_expected = {
        int(pid): exp.build_slot_labels_for_annotation(normal_ann[pid], slots, context="normal")
        for pid in sorted(normal_ann.keys())
    }

    ctx = gui.RunContext(run_id=1, cancel_event=threading.Event(), log_func=lambda _rid, _msg: None)
    backend = gui.load_proto_backend_module(ctx)
    tuned_cfg = gui.load_tuned_config(ctx)
    gui.apply_config_to_module(backend, tuned_cfg)
    package_path = gui.find_slot_package_path(ctx)
    package_cls = getattr(backend, "SlotRecognitionPackage", None) or getattr(backend, "RecognitionPackage", None)
    package = package_cls(package_path)

    outputs = []
    for input_path in [Path(p) for p in args.input]:
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            outputs.append({"input": str(input_path), "error": "Could not read image"})
            continue

        circle, crop_method = detect_map_circle_compat(backend, ctx, img)
        if crop_method == "blue_ring":
            crop_box = backend.clip_box(
                backend.estimate_panel_box(circle, float(getattr(backend, "DEFAULT_PANEL_SCALE", 1.12))),
                img.shape[1],
                img.shape[0],
            )
            raw_panel = img[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
            panel = normalize_panel_for_slot_recognition_compat(backend, raw_panel)
        else:
            initial_box = refine_panel_box_compat(backend, ctx, img, circle)
            crop_box, raw_panel, panel, _metrics, _earth_rank, _crop_refine = refine_crop_box_by_earth_score_slot_compat(
                backend, ctx, img, initial_box, package
            )

        obs = infer_slots_on_panel(
            panel,
            slots,
            model,
            sparse_model=sparse_model,
            sparse_temperature=float(args.sparse_temperature),
        )

        ranked = []
        for pid, expected in layout_expected.items():
            ranked.append((int(pid), score_layout(obs, expected)))
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        best_layout = int(ranked[0][0]) if ranked else -1
        expected_best = layout_expected.get(best_layout, {})

        panel_overlay = draw_status_aware_overlay(panel, obs, expected_best)
        full_overlay = img.copy()
        cv2.rectangle(full_overlay, (int(crop_box.x0), int(crop_box.y0)), (int(crop_box.x1), int(crop_box.y1)), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            full_overlay,
            f"Status-aware best layout: {best_layout}",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        panel_path = out_dir / f"{input_path.stem}_status_aware_panel_overlay.png"
        full_path = out_dir / f"{input_path.stem}_status_aware_full_overlay.png"
        report_path = out_dir / f"{input_path.stem}_status_aware_overlay_report.json"
        cv2.imwrite(str(panel_path), panel_overlay)
        cv2.imwrite(str(full_path), full_overlay)

        top5 = [{"layout_number": int(pid), "score": float(sc)} for pid, sc in ranked[:5]]
        status_slot_rows = []
        for sid in sorted(obs.keys()):
            exp_lbl = str(expected_best.get(int(sid), "empty"))
            pred_lbl = str(obs[sid]["best_label"])
            if is_status_aware_label(exp_lbl) or is_status_aware_label(pred_lbl):
                status_slot_rows.append(
                    {
                        "slot_id": int(sid),
                        "expected": exp_lbl,
                        "predicted": pred_lbl,
                        "predicted_prob": float(obs[sid]["best_prob"]),
                        "second": str(obs[sid]["second_label"]),
                        "second_prob": float(obs[sid]["second_prob"]),
                    }
                )

        report = {
            "input": str(input_path),
            "crop_method": str(crop_method),
            "panel_shape": [int(panel.shape[1]), int(panel.shape[0])],
            "best_layout": int(best_layout),
            "model_type": "sparse" if sparse_model is not None else "centroid",
            "top5": top5,
            "status_slots": status_slot_rows,
            "panel_overlay_image": str(panel_path),
            "full_overlay_image": str(full_path),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        outputs.append(report)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
