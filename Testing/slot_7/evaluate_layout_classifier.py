import argparse
from pathlib import Path
from typing import Dict, List

import torch

import layout_classifier_lib as lib


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Slot 7 CNN on class subfolders.")
    parser.add_argument("--model", default="slot7_status_cnn.pt", help="Checkpoint path.")
    parser.add_argument("--data-dir", default=".", help="Dataset folder containing class subfolders.")
    parser.add_argument("--output-json", default="evaluation_report.json", help="Evaluation report JSON.")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size, meta = lib.load_checkpoint(model_path, device)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = lib.collect_samples(data_dir, class_names)
    if not samples:
        raise SystemExit("No evaluation images found.")

    confusion: List[List[int]] = [[0 for _ in class_names] for _ in class_names]
    per_class_total = [0 for _ in class_names]
    per_class_correct = [0 for _ in class_names]
    failures: List[Dict[str, object]] = []
    total = 0
    correct = 0

    for image_path, true_idx in samples:
        pred = lib.infer_proba(
            model=model,
            image_path=image_path,
            class_names=class_names,
            image_size=image_size,
            device=device,
        )
        pred_idx = class_to_idx[pred["predicted_label"]]
        confusion[true_idx][pred_idx] += 1
        per_class_total[true_idx] += 1
        total += 1
        if pred_idx == true_idx:
            per_class_correct[true_idx] += 1
            correct += 1
        else:
            failures.append(
                {
                    "file": str(image_path),
                    "true_label": class_names[true_idx],
                    "predicted_label": class_names[pred_idx],
                    "confidence": float(pred["confidence"]),
                }
            )

    class_metrics = []
    for i, name in enumerate(class_names):
        n = per_class_total[i]
        acc = (per_class_correct[i] / n) if n > 0 else 0.0
        class_metrics.append(
            {
                "label": name,
                "count": n,
                "correct": per_class_correct[i],
                "accuracy": acc,
            }
        )

    report = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "class_names": class_names,
        "image_size": int(image_size),
        "meta": meta,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
        "failures": failures,
    }

    out_path = Path(args.output_json).resolve()
    lib.write_json(out_path, report)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
