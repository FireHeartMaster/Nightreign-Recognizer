import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import layout_classifier_lib as lib


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Slot 2 CNN on a labeled folder.")
    parser.add_argument("--model", default="slot2_status_cnn.pt", help="Checkpoint path.")
    parser.add_argument("--data-dir", default=".", help="Folder containing class subfolders.")
    parser.add_argument("--output-json", default="evaluation_report.json", help="Output JSON path.")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_json = Path(args.output_json).resolve()

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size, meta = lib.load_checkpoint(model_path, device)
    samples = lib.collect_samples(data_dir, class_names)
    if not samples:
        raise SystemExit("No samples found.")

    conf = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    failures: List[Dict[str, object]] = []
    class_totals = np.zeros(len(class_names), dtype=np.int64)
    class_correct = np.zeros(len(class_names), dtype=np.int64)

    for path, gt in samples:
        out = lib.infer_proba(
            model=model,
            image_path=path,
            class_names=class_names,
            image_size=image_size,
            device=device,
        )
        pred = class_names.index(out["predicted_label"])
        conf[gt, pred] += 1
        class_totals[gt] += 1
        if pred == gt:
            class_correct[gt] += 1
        else:
            failures.append(
                {
                    "file": str(path),
                    "expected": class_names[gt],
                    "predicted": class_names[pred],
                    "confidence": float(out["confidence"]),
                    "ranking": out["ranking"],
                }
            )

    total = int(class_totals.sum())
    correct = int(class_correct.sum())
    class_metrics = []
    for idx, name in enumerate(class_names):
        c_total = int(class_totals[idx])
        c_correct = int(class_correct[idx])
        class_metrics.append(
            {
                "label": name,
                "count": c_total,
                "correct": c_correct,
                "accuracy": float(c_correct / c_total) if c_total else 0.0,
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
        "accuracy": float(correct / total) if total else 0.0,
        "class_metrics": class_metrics,
        "confusion_matrix": conf.tolist(),
        "failures": failures,
    }
    lib.write_json(out_json, report)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Saved report to: {out_json}")


if __name__ == "__main__":
    main()
