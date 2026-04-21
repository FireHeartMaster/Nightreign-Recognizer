#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import experimental_great_hollow_empty_vs_churchrise_model_test as base


def split_layouts(samples: Sequence[base.Sample], train_ratio: float, val_ratio: float, seed: int):
    return base.split_by_layout(samples, train_ratio, val_ratio, seed)


def fit_prototypes(samples: Sequence[base.Sample]) -> Dict[str, np.ndarray]:
    x_empty = np.stack([s.patch_bgr.astype(np.float32).reshape(-1) for s in samples if s.binary_label == 0], axis=0)
    x_pos = np.stack([s.patch_bgr.astype(np.float32).reshape(-1) for s in samples if s.binary_label == 1], axis=0)
    return {
        "centroid_empty": x_empty.mean(axis=0).astype(np.float32),
        "centroid_pos": x_pos.mean(axis=0).astype(np.float32),
    }


def infer_prob_empty(x_vec: np.ndarray, cent_empty: np.ndarray, cent_pos: np.ndarray, temp: float) -> Tuple[float, float, float]:
    de = float(np.mean((x_vec - cent_empty) ** 2))
    dp = float(np.mean((x_vec - cent_pos) ** 2))
    t = max(1e-6, float(temp))
    a = -de / t
    b = -dp / t
    m = max(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    p_empty = float(ea / (ea + eb + 1e-12))
    return p_empty, de, dp


def eval_set(samples: Sequence[base.Sample], cent_empty: np.ndarray, cent_pos: np.ndarray, temp: float) -> Dict:
    rows = []
    tp = tn = fp = fn = 0
    empty_probs = []
    pos_probs = []
    for s in samples:
        x = s.patch_bgr.astype(np.float32).reshape(-1)
        p_empty, de, dp = infer_prob_empty(x, cent_empty, cent_pos, temp)
        pred = 0 if p_empty >= 0.5 else 1
        y = int(s.binary_label)
        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 1:
            tp += 1
        else:
            fn += 1
        if y == 0:
            empty_probs.append(p_empty)
        else:
            pos_probs.append(p_empty)
        rows.append(
            {
                "layout_number": int(s.layout_number),
                "slot_id": int(s.slot_id),
                "gt_label": str(s.gt_label),
                "y_true": int(y),
                "y_pred": int(pred),
                "p_empty": float(p_empty),
                "d_empty": float(de),
                "d_pos": float(dp),
            }
        )
    n = max(1, len(samples))
    acc = float((tp + tn) / n)
    empty_recall = float(tn / max(1, (tn + fp)))
    empty_precision = float(tn / max(1, (tn + fn)))
    pos_recall = float(tp / max(1, (tp + fn)))
    separation = float(np.mean(empty_probs) - np.mean(pos_probs)) if empty_probs and pos_probs else 0.0
    return {
        "count": int(len(samples)),
        "accuracy": acc,
        "empty_recall": empty_recall,
        "empty_precision": empty_precision,
        "church_like_recall": pos_recall,
        "confusion_true_empty_pos_pred_empty_pos": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "mean_p_empty_when_true_empty": float(np.mean(empty_probs)) if empty_probs else 0.0,
        "mean_p_empty_when_true_church_like": float(np.mean(pos_probs)) if pos_probs else 0.0,
        "empty_confidence_separation": separation,
        "rows": rows,
    }


def choose_temperature(val_samples: Sequence[base.Sample], cent_empty: np.ndarray, cent_pos: np.ndarray) -> float:
    candidates = [0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0]
    best_t = 0.1
    best_score = -1.0
    for t in candidates:
        metrics = eval_set(val_samples, cent_empty, cent_pos, t)
        score = (
            2.0 * float(metrics["accuracy"])
            + 3.0 * float(metrics["empty_recall"])
            + 1.0 * float(metrics["empty_confidence_separation"])
        )
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)


def leave_one_layout_out(samples: Sequence[base.Sample], temp: float) -> Dict:
    layouts = sorted({int(s.layout_number) for s in samples})
    all_rows: List[dict] = []
    for hold in layouts:
        tr = [s for s in samples if int(s.layout_number) != hold]
        te = [s for s in samples if int(s.layout_number) == hold]
        proto = fit_prototypes(tr)
        fold = eval_set(te, proto["centroid_empty"], proto["centroid_pos"], temp)
        all_rows.extend(fold["rows"])
    n = max(1, len(all_rows))
    acc = float(sum(int(r["y_true"] == r["y_pred"]) for r in all_rows) / n)
    empty_rows = [r for r in all_rows if int(r["y_true"]) == 0]
    pos_rows = [r for r in all_rows if int(r["y_true"]) == 1]
    empty_recall = float(sum(int(r["y_pred"] == 0) for r in empty_rows) / max(1, len(empty_rows)))
    sep = float(np.mean([r["p_empty"] for r in empty_rows]) - np.mean([r["p_empty"] for r in pos_rows]))
    return {
        "count": int(len(all_rows)),
        "accuracy": acc,
        "empty_recall": empty_recall,
        "mean_p_empty_when_true_empty": float(np.mean([r["p_empty"] for r in empty_rows])) if empty_rows else 0.0,
        "mean_p_empty_when_true_church_like": float(np.mean([r["p_empty"] for r in pos_rows])) if pos_rows else 0.0,
        "empty_confidence_separation": sep,
    }


def train(args: argparse.Namespace) -> None:
    meta = base.load_metadata(Path(args.package))
    samples = base.build_samples(
        metadata=meta,
        images_dir=Path(args.images_dir),
        patch_size=int(args.patch_size),
        context_scale=float(args.context_scale),
        min_half_frac=float(args.min_half_frac),
        target_slot_ids=[int(args.slot_id)],
    )
    tr, val, te = split_layouts(samples, args.train_ratio, args.val_ratio, args.seed)
    proto = fit_prototypes(tr)
    temp = choose_temperature(val, proto["centroid_empty"], proto["centroid_pos"])

    train_metrics = eval_set(tr, proto["centroid_empty"], proto["centroid_pos"], temp)
    val_metrics = eval_set(val, proto["centroid_empty"], proto["centroid_pos"], temp)
    test_metrics = eval_set(te, proto["centroid_empty"], proto["centroid_pos"], temp)
    loo_metrics = leave_one_layout_out(samples, temp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "gh_slot4_empty_vs_churchrise_prototype_model.pkl"
    model_blob = {
        "model_type": "prototype_distance_binary",
        "slot_id": int(args.slot_id),
        "patch_size": int(args.patch_size),
        "context_scale": float(args.context_scale),
        "min_half_frac": float(args.min_half_frac),
        "temperature": float(temp),
        "centroid_empty": proto["centroid_empty"],
        "centroid_pos": proto["centroid_pos"],
        "label_map": {"empty": 0, "church_like": 1},
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "counts": {
            "all": len(samples),
            "train": len(tr),
            "val": len(val),
            "test": len(te),
            "labels_all": dict(Counter(int(s.binary_label) for s in samples)),
        },
        "slot_id": int(args.slot_id),
        "config": {
            "package": str(args.package),
            "images_dir": str(args.images_dir),
            "patch_size": int(args.patch_size),
            "context_scale": float(args.context_scale),
            "min_half_frac": float(args.min_half_frac),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "temperature": float(temp),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "leave_one_layout_out_metrics": loo_metrics,
        "model_path": str(model_path),
    }
    report_path = out_dir / "gh_slot4_empty_vs_churchrise_prototype_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved model: {model_path}")
    print(f"saved report: {report_path}")
    print(
        json.dumps(
            {
                "val": {
                    "accuracy": val_metrics["accuracy"],
                    "empty_recall": val_metrics["empty_recall"],
                    "empty_confidence_separation": val_metrics["empty_confidence_separation"],
                },
                "test": {
                    "accuracy": test_metrics["accuracy"],
                    "empty_recall": test_metrics["empty_recall"],
                    "empty_confidence_separation": test_metrics["empty_confidence_separation"],
                },
                "loo": loo_metrics,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Great Hollow slot-specific prototype model: empty vs church_like.")
    p.add_argument("--package", type=Path, default=Path("nightreign_geom_proto_v4_slots.npz"))
    p.add_argument("--images-dir", type=Path, default=Path("renders/recognizer_no_text"))
    p.add_argument("--out-dir", type=Path, default=Path("tmp_gh_slot4_prototype_model_test"))
    p.add_argument("--slot-id", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--context-scale", type=float, default=1.2)
    p.add_argument("--min-half-frac", type=float, default=0.075)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
