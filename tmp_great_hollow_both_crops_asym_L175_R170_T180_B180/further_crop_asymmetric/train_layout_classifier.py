#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from layout_classifier_lib import (
    build_gallery,
    configure_logging,
    evaluate_on_queries,
    image_to_gray,
    list_images,
    read_image,
    save_pickle,
    tune_hyperparameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Great Hollow layout classifier")
    parser.add_argument("--images", required=True, help="Directory with map PNGs")
    parser.add_argument("--metadata", required=True, help="Metadata JSON path")
    parser.add_argument("--output", default="models/layout_model.pkl", help="Output pickle path")
    parser.add_argument("--report", default="models/training_report.json", help="Training report path")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    model = build_gallery(args.images, args.metadata)
    calibration = tune_hyperparameters(model, args.images, seed=args.seed)
    model["calibration"] = calibration
    model["train_args"] = vars(args)

    records = list_images(args.images)
    original_queries = []
    for rec in records:
        gray = image_to_gray(read_image(rec.path))
        original_queries.append((rec.pattern_id, Path(rec.path).name, gray))
    original_report = evaluate_on_queries(original_queries, model)

    report = {
        "model_summary": {
            "patterns": [int(x) for x in model["pattern_ids"]],
            "num_patterns": int(len(model["pattern_ids"])),
            "num_images": int(len(model["image_paths"])),
            "metadata_diagnostics": model["metadata_diagnostics"],
            "calibration": model["calibration"],
        },
        "split_reports": {},
        "original_gallery_report": original_report,
        "limitations": [
            "Dataset contains one image per patternId, so the saved model is a gallery matcher rather than a conventional class-generalizing classifier.",
            "The uploaded JSON did not contain *_px_bl fields; code therefore used x_norm/y_norm fallback with explicit logging and documentation.",
        ],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_pickle(model, args.output)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({
        "saved_model": args.output,
        "saved_report": args.report,
        "num_patterns": len(model["pattern_ids"]),
        "calibration": model["calibration"],
        "original_top1": original_report["top1_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
