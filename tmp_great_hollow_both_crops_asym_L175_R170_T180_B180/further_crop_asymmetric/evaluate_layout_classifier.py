#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from layout_classifier_lib import classify_image, configure_logging, list_images, load_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Great Hollow layout classifier")
    parser.add_argument("--images", required=True, help="Directory with map PNGs")
    parser.add_argument("--metadata", required=False, help="Accepted for CLI compatibility; metadata is embedded in the model")
    parser.add_argument("--model", required=True, help="Path to trained model pickle")
    parser.add_argument("--output", required=True, help="Path to evaluation report JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    model = load_pickle(args.model)
    records = list_images(args.images)

    y_true = []
    rankings = []
    margins = []
    latencies = []
    decided = []
    rows = []
    for rec in records:
        result = classify_image(rec.path, model)
        y_true.append(rec.pattern_id)
        rankings.append([item["patternId"] for item in result["top5"]])
        margins.append(result["margin"])
        latencies.append(result["latency_ms"])
        decided.append(not result["unknown"])
        rows.append({
            "file": Path(rec.path).name,
            "truth": rec.pattern_id,
            "best_patternId": result["best_patternId"],
            "unknown": result["unknown"],
            "margin": result["margin"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
            "top5": result["top5"],
        })

    def topk(k: int) -> float:
        hits = 0
        for truth, ranked in zip(y_true, rankings):
            if truth in ranked[:k]:
                hits += 1
        return hits / max(len(y_true), 1)

    labels = [int(x) for x in model["pattern_ids"]]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for row in rows:
        pred = row["best_patternId"]
        cm[label_to_index[row["truth"]]][label_to_index[pred]] += 1

    decided_rows = [row for row in rows if not row["unknown"]]
    decided_acc = (
        sum(1 for row in decided_rows if row["truth"] == row["best_patternId"]) / len(decided_rows)
        if decided_rows else None
    )

    report = {
        "model_path": args.model,
        "num_images": len(rows),
        "top1_accuracy": topk(1),
        "top3_accuracy": topk(3),
        "top5_accuracy": topk(5),
        "unknown_rate": 1.0 - (sum(decided) / max(len(decided), 1)),
        "accuracy_on_decided_samples": decided_acc,
        "margin_distribution": {
            "min": min(margins) if margins else None,
            "p25": sorted(margins)[max(0, int(0.25 * (len(margins)-1)))] if margins else None,
            "median": sorted(margins)[len(margins)//2] if margins else None,
            "p75": sorted(margins)[max(0, int(0.75 * (len(margins)-1)))] if margins else None,
            "max": max(margins) if margins else None,
        },
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "median": sorted(latencies)[len(latencies)//2] if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,
        },
        "rows": rows,
        "model_metadata_diagnostics": model.get("metadata_diagnostics", {}),
        "calibration": model.get("calibration", {}),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ["num_images", "top1_accuracy", "top3_accuracy", "top5_accuracy", "unknown_rate", "accuracy_on_decided_samples"]}, indent=2))


if __name__ == "__main__":
    main()
