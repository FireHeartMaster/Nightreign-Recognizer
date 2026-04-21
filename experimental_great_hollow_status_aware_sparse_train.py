#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import experimental_great_hollow_status_aware_slot_classifier_test as exp


@dataclass
class SlotModelBundle:
    labels: List[str]
    mean: np.ndarray
    std: np.ndarray
    weight: np.ndarray
    bias: np.ndarray


def split_samples_by_layout(
    samples: Sequence[exp.PatchSample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[exp.PatchSample], List[exp.PatchSample], List[exp.PatchSample]]:
    layout_ids = sorted({int(s.layout_number) for s in samples})
    tr_ids, va_ids, te_ids = exp.split_layouts(layout_ids, train_ratio, val_ratio, seed)
    tr = [s for s in samples if int(s.layout_number) in tr_ids]
    va = [s for s in samples if int(s.layout_number) in va_ids]
    te = [s for s in samples if int(s.layout_number) in te_ids]
    return tr, va, te


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    logp = torch.log(probs.clamp(min=1e-9))
    return -(probs * logp).sum(dim=1).mean()


def _train_one_slot(
    slot_samples: Sequence[exp.PatchSample],
    epochs: int,
    lr: float,
    weight_decay: float,
    entropy_penalty: float,
    device: torch.device,
) -> SlotModelBundle:
    labels = sorted({str(s.label) for s in slot_samples})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    x = np.stack([s.descriptor.astype(np.float32) for s in slot_samples], axis=0).astype(np.float32)
    y = np.array([label_to_idx[str(s.label)] for s in slot_samples], dtype=np.int64)

    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    xz = ((x - mean) / std).astype(np.float32)

    x_t = torch.from_numpy(xz).to(device)
    y_t = torch.from_numpy(y).to(device)

    model = nn.Linear(x_t.shape[1], len(labels), bias=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    ce = nn.CrossEntropyLoss()

    for _ in range(int(epochs)):
        logits = model(x_t)
        loss = ce(logits, y_t)
        if float(entropy_penalty) > 0.0:
            loss = loss + float(entropy_penalty) * _entropy_from_logits(logits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    weight = model.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.bias.detach().cpu().numpy().astype(np.float32)
    return SlotModelBundle(
        labels=labels,
        mean=mean,
        std=std,
        weight=weight,
        bias=bias,
    )


def train_models(
    train_samples: Sequence[exp.PatchSample],
    epochs: int,
    lr: float,
    weight_decay: float,
    entropy_penalty: float,
    device: torch.device,
) -> Dict[int, SlotModelBundle]:
    by_slot: Dict[int, List[exp.PatchSample]] = defaultdict(list)
    for s in train_samples:
        by_slot[int(s.slot_id)].append(s)
    out: Dict[int, SlotModelBundle] = {}
    for slot_id, samples in by_slot.items():
        out[int(slot_id)] = _train_one_slot(
            slot_samples=samples,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            entropy_penalty=entropy_penalty,
            device=device,
        )
    return out


def predict_probs(bundle: SlotModelBundle, descriptor: np.ndarray, temperature: float) -> Dict[str, float]:
    x = descriptor.astype(np.float32)
    xz = (x - bundle.mean) / bundle.std
    logits = np.dot(bundle.weight, xz) + bundle.bias
    t = max(1e-6, float(temperature))
    logits = logits / t
    m = float(np.max(logits))
    ex = np.exp(logits - m)
    probs = ex / (float(np.sum(ex)) + 1e-12)
    return {bundle.labels[i]: float(probs[i]) for i in range(len(bundle.labels))}


def evaluate(
    models: Dict[int, SlotModelBundle],
    samples: Sequence[exp.PatchSample],
    unknown_margin: float,
    temperature: float,
) -> Dict[str, object]:
    rows = []
    correct_forced = 0
    correct_unknown = 0
    decided = 0
    margins: List[float] = []
    per_label = Counter()
    per_label_correct_forced = Counter()
    per_label_correct_unknown = Counter()
    per_label_decided = Counter()

    for s in samples:
        bundle = models.get(int(s.slot_id))
        if bundle is None:
            pred_forced = "empty"
            pred_unknown = "unknown"
            best_p = 0.0
            second_p = 0.0
            margin = 0.0
            probs = {"empty": 1.0}
        else:
            probs = predict_probs(bundle, s.descriptor, temperature=temperature)
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            pred_forced = str(ranked[0][0])
            best_p = float(ranked[0][1])
            second_p = float(ranked[1][1]) if len(ranked) > 1 else 0.0
            margin = float(best_p - second_p)
            pred_unknown = "unknown" if margin < float(unknown_margin) else pred_forced

        y = str(s.label)
        cf = int(pred_forced == y)
        cu = int(pred_unknown == y)
        is_decided = int(pred_unknown != "unknown")

        correct_forced += cf
        correct_unknown += cu
        decided += is_decided
        margins.append(float(margin))

        per_label[y] += 1
        per_label_correct_forced[y] += cf
        per_label_correct_unknown[y] += cu
        per_label_decided[y] += is_decided

        rows.append(
            {
                "layout_number": int(s.layout_number),
                "context": str(s.context),
                "slot_id": int(s.slot_id),
                "label_true": y,
                "label_pred_forced": pred_forced,
                "label_pred_unknown": pred_unknown,
                "best_prob": best_p,
                "second_prob": second_p,
                "margin": margin,
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

    margins_np = np.asarray(margins, dtype=np.float64) if margins else np.asarray([0.0], dtype=np.float64)
    return {
        "count": int(len(samples)),
        "forced_accuracy": float(correct_forced / n),
        "unknown_accuracy": float(correct_unknown / n),
        "decided_coverage": float(decided / n),
        "unknown_accuracy_on_decided": float(correct_unknown / max(1, decided)),
        "mean_margin": float(np.mean(margins_np)),
        "median_margin": float(np.median(margins_np)),
        "p10_margin": float(np.percentile(margins_np, 10)),
        "p90_margin": float(np.percentile(margins_np, 90)),
        "per_label": per_label_metrics,
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Great Hollow status-aware sparse trainer. "
            "Trains per-slot linear classifiers with entropy-penalized objective "
            "to make probabilities more separated."
        )
    )
    p.add_argument("--normal-pattern-dir", type=Path, default=Path("renders_new/no_text"))
    p.add_argument("--normal-image-dir", type=Path, default=Path("renders/recognizer_no_text"))
    p.add_argument("--under-pattern-dir", type=Path, default=Path("renders_new_under/no_text"))
    p.add_argument("--under-image-dir", type=Path, default=Path("renders/recognizer_no_text_under"))
    p.add_argument("--out-dir", type=Path, default=Path("tmp_gh_status_aware_sparse_train"))
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--entropy-penalty", type=float, default=0.08)
    p.add_argument("--unknown-margin", type=float, default=0.03)
    p.add_argument("--temperature", type=float, default=0.55)
    args = p.parse_args()

    normal_ann = exp.load_annotations(Path(args.normal_pattern_dir), suffix="")
    under_ann = exp.load_annotations(Path(args.under_pattern_dir), suffix="_under")
    common_layouts = sorted(set(normal_ann.keys()) & set(under_ann.keys()))
    normal_ann = {k: normal_ann[k] for k in common_layouts}
    under_ann = {k: under_ann[k] for k in common_layouts}
    if not common_layouts:
        raise RuntimeError("No overlapping Great Hollow layouts found.")

    points = exp.collect_slot_points(normal_ann, under_ann)
    slots = exp.build_slots(points)
    samples = exp.build_patch_samples(
        slots=slots,
        normal_ann=normal_ann,
        under_ann=under_ann,
        normal_img_dir=Path(args.normal_image_dir),
        under_img_dir=Path(args.under_image_dir),
    )
    if not samples:
        raise RuntimeError("No patch samples built.")

    train_samples, val_samples, test_samples = split_samples_by_layout(
        samples=samples,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = train_models(
        train_samples=train_samples,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        entropy_penalty=float(args.entropy_penalty),
        device=device,
    )

    train_metrics = evaluate(models, train_samples, float(args.unknown_margin), float(args.temperature))
    val_metrics = evaluate(models, val_samples, float(args.unknown_margin), float(args.temperature))
    test_metrics = evaluate(models, test_samples, float(args.unknown_margin), float(args.temperature))

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
    model_path = out_dir / "gh_status_aware_sparse_slot_models.npz"
    model_blob = {
        "slot_ids": np.array(sorted(models.keys()), dtype=np.int32),
        "payload_json": json.dumps(
            {
                int(sid): {
                    "labels": bundle.labels,
                    "mean": bundle.mean.tolist(),
                    "std": bundle.std.tolist(),
                    "weight": bundle.weight.tolist(),
                    "bias": bundle.bias.tolist(),
                }
                for sid, bundle in models.items()
            }
        ),
    }
    np.savez_compressed(model_path, **model_blob)

    report = {
        "config": {
            "normal_pattern_dir": str(args.normal_pattern_dir),
            "normal_image_dir": str(args.normal_image_dir),
            "under_pattern_dir": str(args.under_pattern_dir),
            "under_image_dir": str(args.under_image_dir),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "entropy_penalty": float(args.entropy_penalty),
            "unknown_margin": float(args.unknown_margin),
            "temperature": float(args.temperature),
            "device": str(device),
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
        "model_path": str(model_path),
    }
    report_path = out_dir / "gh_status_aware_sparse_train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"saved model: {model_path}")
    print(f"saved report: {report_path}")
    print(
        json.dumps(
            {
                "test_forced_accuracy": float(test_metrics["forced_accuracy"]),
                "test_unknown_accuracy": float(test_metrics["unknown_accuracy"]),
                "test_coverage": float(test_metrics["decided_coverage"]),
                "test_mean_margin": float(test_metrics["mean_margin"]),
                "test_p10_margin": float(test_metrics["p10_margin"]),
                "test_p90_margin": float(test_metrics["p90_margin"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
