#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run nightreign_run_with_tuned_config_v1.py over all numbered reference images, "
            "summarize performance, list failures, and analyze likely failure reasons."
        )
    )
    p.add_argument("--runner", type=Path, required=True, help="Path to nightreign_run_with_tuned_config_v1.py")
    p.add_argument("--backend", type=Path, required=True, help="Path to the recognizer backend script used by the runner")
    p.add_argument("--config-json", type=Path, required=True, help="Path to the tuned config JSON")
    p.add_argument("--package", type=Path, required=True, help="Path to the .npz recognition package")
    p.add_argument("--images-dir", type=Path, required=True, help="Directory containing numbered reference images")
    p.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable to use")
    p.add_argument("--top", type=int, default=20, help="How many layouts the recognizer should print/rank")
    p.add_argument("--max-images", type=int, default=0, help="Optional cap on number of images. 0 means all")
    p.add_argument("--keep-per-image-json", action="store_true", help="Keep per-image JSON reports in the temp/output folder")
    p.add_argument("--output-dir", type=Path, default=Path("tuned_eval_all_images_output"), help="Directory for summary outputs")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional override path for summary JSON")
    p.add_argument("--summary-csv", type=Path, default=None, help="Optional override path for per-image CSV")
    return p.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_numbered_images(images_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for path in sorted(images_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.search(r"(\d+)$", path.stem)
        if not m:
            continue
        items.append((int(m.group(1)), path))
    if not items:
        raise FileNotFoundError(f"No numbered reference images found in {images_dir}")
    return items


def load_package_metadata(package_path: Path) -> Dict[str, Any]:
    with np.load(package_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    return metadata


def build_truth_maps(metadata: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, List[dict]]]:
    layouts = metadata["layouts"]
    slots_by_earth = metadata["slots_by_earth"]
    by_number = {int(item["layout_number"]): item for item in layouts}
    return by_number, slots_by_earth


def run_one_image(
    python_exe: str,
    runner: Path,
    backend: Path,
    config_json: Path,
    package: Path,
    image_path: Path,
    json_out: Path,
    top: int,
) -> Tuple[int, str, str]:
    cmd = [
        python_exe,
        str(runner),
        "--backend", str(backend),
        "--config-json", str(config_json),
        "--package", str(package),
        "--image", str(image_path),
        "--top", str(top),
        "--json-out", str(json_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def find_rank(results: Sequence[Dict[str, Any]], layout_number: int) -> Optional[int]:
    for idx, item in enumerate(results, start=1):
        if int(item["layout_number"]) == int(layout_number):
            return idx
    return None


def safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def slot_quality_vs_truth(obs_slots: Sequence[Dict[str, Any]], gt_labels: Sequence[str]) -> Dict[str, Any]:
    out = {
        "slot_count": 0,
        "correct": 0,
        "unknown": 0,
        "wrong_label": 0,
        "false_empty": 0,
        "false_positive_empty": 0,
        "per_slot_status": [],
        "label_confusions": Counter(),
    }
    if len(obs_slots) != len(gt_labels):
        return out
    for idx, (obs, gt) in enumerate(zip(obs_slots, gt_labels), start=1):
        pred = str(obs["label"])
        gt = str(gt)
        status = "correct"
        if pred == gt:
            out["correct"] += 1
        elif pred == "unknown":
            out["unknown"] += 1
            status = "unknown"
        elif pred == "empty" and gt != "empty":
            out["false_empty"] += 1
            status = "false_empty"
            out["label_confusions"][f"{gt} -> empty"] += 1
        elif pred != "empty" and gt == "empty":
            out["false_positive_empty"] += 1
            status = "false_positive_empty"
            out["label_confusions"][f"empty -> {pred}"] += 1
        else:
            out["wrong_label"] += 1
            status = "wrong_label"
            out["label_confusions"][f"{gt} -> {pred}"] += 1

        out["slot_count"] += 1
        out["per_slot_status"].append({
            "slot_index": idx,
            "slot_name": obs.get("slot_name", f"slot_{idx:02d}"),
            "pred": pred,
            "gt": gt,
            "status": status,
            "best_score": obs.get("best_score"),
        })
    return out


def score_margin(results: Sequence[Dict[str, Any]]) -> Optional[float]:
    if len(results) < 2:
        return None
    s1 = float(results[0]["total_score"])
    s2 = float(results[1]["total_score"])
    if not (math.isfinite(s1) and math.isfinite(s2)):
        return None
    return s1 - s2


def earth_margin(earth_ranking: Sequence[Dict[str, Any]]) -> Optional[float]:
    if len(earth_ranking) < 2:
        return None
    s1 = float(earth_ranking[0]["score"])
    s2 = float(earth_ranking[1]["score"])
    if not (math.isfinite(s1) and math.isfinite(s2)):
        return None
    return s1 - s2


def summarize_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "earth_top1_acc": safe_mean([1.0 if r["earth_correct"] else 0.0 for r in rows]),
        "layout_top1_acc": safe_mean([1.0 if r["layout_correct"] else 0.0 for r in rows]),
        "layout_top3_acc": safe_mean([1.0 if (r["true_rank"] is not None and r["true_rank"] <= 3) else 0.0 for r in rows]),
        "avg_true_rank": safe_mean([r["true_rank"] for r in rows if r["true_rank"] is not None]),
        "avg_top1_margin": safe_mean([r["top1_margin"] for r in rows if r["top1_margin"] is not None]),
        "avg_earth_margin": safe_mean([r["earth_margin"] for r in rows if r["earth_margin"] is not None]),
        "avg_slot_correct_rate": safe_mean([r["slot_correct_rate"] for r in rows if r["slot_correct_rate"] is not None]),
        "avg_slot_unknown_rate": safe_mean([r["slot_unknown_rate"] for r in rows if r["slot_unknown_rate"] is not None]),
        "avg_slot_wrong_label_rate": safe_mean([r["slot_wrong_label_rate"] for r in rows if r["slot_wrong_label_rate"] is not None]),
        "avg_slot_false_empty_rate": safe_mean([r["slot_false_empty_rate"] for r in rows if r["slot_false_empty_rate"] is not None]),
        "avg_slot_false_positive_empty_rate": safe_mean([r["slot_false_positive_empty_rate"] for r in rows if r["slot_false_positive_empty_rate"] is not None]),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = args.summary_json or (output_dir / "tuned_eval_summary.json")
    summary_csv_path = args.summary_csv or (output_dir / "tuned_eval_per_image.csv")

    metadata = load_package_metadata(args.package)
    truth_layouts, slots_by_earth = build_truth_maps(metadata)

    numbered_images = list_numbered_images(args.images_dir)
    if args.max_images > 0:
        numbered_images = numbered_images[:args.max_images]

    wrong_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    failure_reason_counter = Counter()
    wrong_slot_counter = Counter()
    unknown_slot_counter = Counter()
    confusion_counter = Counter()
    wrong_top1_layout_counter = Counter()
    wrong_earth_counter = Counter()

    per_image_json_dir = output_dir / "per_image_json"
    per_image_json_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(numbered_images)} images...")
    print()

    for idx, (layout_number, image_path) in enumerate(numbered_images, start=1):
        gt_layout = truth_layouts.get(layout_number)
        if gt_layout is None:
            print(f"[{idx:03d}/{len(numbered_images)}] Skipping {image_path.name}: no ground truth layout in package")
            continue

        json_out = per_image_json_dir / f"{layout_number}.json"
        rc, stdout, stderr = run_one_image(
            python_exe=args.python_exe,
            runner=args.runner,
            backend=args.backend,
            config_json=args.config_json,
            package=args.package,
            image_path=image_path,
            json_out=json_out,
            top=args.top,
        )

        if rc != 0:
            row = {
                "layout_number": layout_number,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "error": True,
                "error_returncode": rc,
                "error_stdout": stdout,
                "error_stderr": stderr,
                "truth_earth": gt_layout["shifting_earth"],
                "layout_correct": False,
                "earth_correct": False,
                "true_rank": None,
            }
            all_rows.append(row)
            wrong_rows.append(row)
            failure_reason_counter["runner_error"] += 1
            print(f"[{idx:03d}/{len(numbered_images)}] {image_path.name}: ERROR rc={rc}")
            continue

        payload = json.loads(json_out.read_text(encoding="utf-8"))
        truth_earth = str(gt_layout["shifting_earth"])
        earth_ranking = payload.get("earth_ranking", [])
        pred_earth = earth_ranking[0]["name"] if earth_ranking else None
        earth_correct = (pred_earth == truth_earth)

        results = payload.get("results", [])
        top1_layout = int(results[0]["layout_number"]) if results else None
        true_rank = find_rank(results, layout_number)
        layout_correct = (top1_layout == layout_number)

        obs_slots = payload.get("observations_by_earth", {}).get(truth_earth, {}).get("slots", [])
        slot_eval = slot_quality_vs_truth(obs_slots, gt_layout["slot_labels"])
        slot_count = slot_eval["slot_count"] or None
        slot_correct_rate = (slot_eval["correct"] / slot_count) if slot_count else None
        slot_unknown_rate = (slot_eval["unknown"] / slot_count) if slot_count else None
        slot_wrong_label_rate = (slot_eval["wrong_label"] / slot_count) if slot_count else None
        slot_false_empty_rate = (slot_eval["false_empty"] / slot_count) if slot_count else None
        slot_false_positive_empty_rate = (slot_eval["false_positive_empty"] / slot_count) if slot_count else None

        row = {
            "layout_number": layout_number,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "truth_earth": truth_earth,
            "pred_earth": pred_earth,
            "earth_correct": earth_correct,
            "top1_layout": top1_layout,
            "layout_correct": layout_correct,
            "true_rank": true_rank,
            "top1_margin": score_margin(results),
            "earth_margin": earth_margin(earth_ranking),
            "crop_method": payload.get("crop_method"),
            "slot_count": slot_count,
            "slot_correct": slot_eval["correct"],
            "slot_unknown": slot_eval["unknown"],
            "slot_wrong_label": slot_eval["wrong_label"],
            "slot_false_empty": slot_eval["false_empty"],
            "slot_false_positive_empty": slot_eval["false_positive_empty"],
            "slot_correct_rate": slot_correct_rate,
            "slot_unknown_rate": slot_unknown_rate,
            "slot_wrong_label_rate": slot_wrong_label_rate,
            "slot_false_empty_rate": slot_false_empty_rate,
            "slot_false_positive_empty_rate": slot_false_positive_empty_rate,
            "context_refinement": payload.get("observations_by_earth", {}).get(truth_earth, {}).get("context_refinement", {}),
            "error": False,
        }
        all_rows.append(row)

        if not layout_correct:
            wrong_rows.append(row)
            if not earth_correct:
                failure_reason_counter["wrong_earth"] += 1
                wrong_earth_counter[f"{truth_earth} -> {pred_earth}"] += 1
            else:
                failure_reason_counter["same_earth_wrong_layout"] += 1

            if slot_unknown_rate is not None and slot_unknown_rate >= 0.45:
                failure_reason_counter["high_unknown_rate"] += 1
            if slot_wrong_label_rate is not None and slot_wrong_label_rate >= 0.12:
                failure_reason_counter["high_wrong_label_rate"] += 1
            if slot_false_positive_empty_rate is not None and slot_false_positive_empty_rate >= 0.08:
                failure_reason_counter["high_false_positive_empty_rate"] += 1
            if row["top1_margin"] is not None and row["top1_margin"] < 1.0:
                failure_reason_counter["low_top1_margin"] += 1
            if row["earth_margin"] is not None and row["earth_margin"] < 0.03:
                failure_reason_counter["low_earth_margin"] += 1

            if top1_layout is not None:
                wrong_top1_layout_counter[top1_layout] += 1

            for item in slot_eval["per_slot_status"]:
                if item["status"] == "unknown":
                    unknown_slot_counter[item["slot_name"]] += 1
                elif item["status"] != "correct":
                    wrong_slot_counter[item["slot_name"]] += 1
            confusion_counter.update(slot_eval["label_confusions"])

        status = "OK" if layout_correct else f"WRONG (got #{top1_layout}, true #{layout_number}, rank={true_rank})"
        print(f"[{idx:03d}/{len(numbered_images)}] {image_path.name}: {status}")

        if not args.keep_per_image_json:
            try:
                json_out.unlink(missing_ok=True)
            except Exception:
                pass

    correct_rows = [r for r in all_rows if not r.get("error") and r["layout_correct"]]
    incorrect_rows = [r for r in all_rows if not r.get("error") and not r["layout_correct"]]

    overall = summarize_group([r for r in all_rows if not r.get("error")])
    correct_summary = summarize_group(correct_rows)
    incorrect_summary = summarize_group(incorrect_rows)

    analysis = {
        "counts": {
            "total_images": len(numbered_images),
            "evaluated_without_runner_error": len([r for r in all_rows if not r.get("error")]),
            "correct_top1": len(correct_rows),
            "incorrect_top1": len(incorrect_rows),
            "runner_errors": len([r for r in all_rows if r.get("error")]),
        },
        "overall": overall,
        "correct_cases": correct_summary,
        "incorrect_cases": incorrect_summary,
        "wrong_vs_right_differences": {
            "slot_unknown_rate_difference": (
                None if correct_summary["avg_slot_unknown_rate"] is None or incorrect_summary["avg_slot_unknown_rate"] is None
                else incorrect_summary["avg_slot_unknown_rate"] - correct_summary["avg_slot_unknown_rate"]
            ),
            "slot_wrong_label_rate_difference": (
                None if correct_summary["avg_slot_wrong_label_rate"] is None or incorrect_summary["avg_slot_wrong_label_rate"] is None
                else incorrect_summary["avg_slot_wrong_label_rate"] - correct_summary["avg_slot_wrong_label_rate"]
            ),
            "top1_margin_difference": (
                None if correct_summary["avg_top1_margin"] is None or incorrect_summary["avg_top1_margin"] is None
                else incorrect_summary["avg_top1_margin"] - correct_summary["avg_top1_margin"]
            ),
            "earth_margin_difference": (
                None if correct_summary["avg_earth_margin"] is None or incorrect_summary["avg_earth_margin"] is None
                else incorrect_summary["avg_earth_margin"] - correct_summary["avg_earth_margin"]
            ),
        },
        "failure_reason_counts": dict(failure_reason_counter.most_common()),
        "most_frequent_wrong_top1_layouts": wrong_top1_layout_counter.most_common(15),
        "most_frequent_wrong_earths": wrong_earth_counter.most_common(15),
        "slots_most_often_unknown_in_failures": unknown_slot_counter.most_common(20),
        "slots_most_often_wrong_in_failures": wrong_slot_counter.most_common(20),
        "most_common_label_confusions_in_failures": confusion_counter.most_common(25),
        "wrong_cases": wrong_rows,
    }

    ensure_parent(summary_json_path)
    summary_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    ensure_parent(summary_csv_path)
    csv_fields = [
        "layout_number", "image_name", "image_path", "truth_earth", "pred_earth",
        "earth_correct", "top1_layout", "layout_correct", "true_rank",
        "top1_margin", "earth_margin", "crop_method", "slot_count", "slot_correct",
        "slot_unknown", "slot_wrong_label", "slot_false_empty", "slot_false_positive_empty",
        "slot_correct_rate", "slot_unknown_rate", "slot_wrong_label_rate",
        "slot_false_empty_rate", "slot_false_positive_empty_rate", "error",
    ]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    print("\n========== FINAL SUMMARY ==========")
    print(f"Total images:            {analysis['counts']['total_images']}")
    print(f"Evaluated successfully:  {analysis['counts']['evaluated_without_runner_error']}")
    print(f"Top-1 correct:           {analysis['counts']['correct_top1']}")
    print(f"Top-1 wrong:             {analysis['counts']['incorrect_top1']}")
    print(f"Runner errors:           {analysis['counts']['runner_errors']}")
    print()
    print(f"Overall layout top-1 acc: {overall['layout_top1_acc']:.4f}" if overall["layout_top1_acc"] is not None else "Overall layout top-1 acc: n/a")
    print(f"Overall layout top-3 acc: {overall['layout_top3_acc']:.4f}" if overall["layout_top3_acc"] is not None else "Overall layout top-3 acc: n/a")
    print(f"Overall avg true rank:    {overall['avg_true_rank']:.3f}" if overall["avg_true_rank"] is not None else "Overall avg true rank: n/a")
    print(f"Overall earth top-1 acc:  {overall['earth_top1_acc']:.4f}" if overall["earth_top1_acc"] is not None else "Overall earth top-1 acc: n/a")
    print(f"Overall slot correct rate:{overall['avg_slot_correct_rate']:.4f}" if overall["avg_slot_correct_rate"] is not None else "Overall slot correct rate: n/a")
    print(f"Overall slot unknown rate:{overall['avg_slot_unknown_rate']:.4f}" if overall["avg_slot_unknown_rate"] is not None else "Overall slot unknown rate: n/a")
    print(f"Overall slot wrong-label rate:{overall['avg_slot_wrong_label_rate']:.4f}" if overall["avg_slot_wrong_label_rate"] is not None else "Overall slot wrong-label rate: n/a")

    print("\n========== WHY WRONG CASES DIFFER ==========")
    diffs = analysis["wrong_vs_right_differences"]
    for key, val in diffs.items():
        if val is None:
            print(f"{key}: n/a")
        else:
            print(f"{key}: {val:+.4f}")

    print("\n========== TOP FAILURE REASONS ==========")
    for reason, count in failure_reason_counter.most_common(10):
        print(f"{reason}: {count}")

    print("\n========== WRONG IMAGES ==========")
    if not wrong_rows:
        print("None.")
    else:
        for row in wrong_rows:
            if row.get("error"):
                print(f"{row['image_name']}: RUNNER ERROR rc={row['error_returncode']}")
            else:
                print(
                    f"{row['image_name']}: got #{row['top1_layout']}, true #{row['layout_number']}, "
                    f"rank={row['true_rank']}, earth={row['pred_earth']} (truth={row['truth_earth']}), "
                    f"unknown_rate={row['slot_unknown_rate']:.3f}, wrong_label_rate={row['slot_wrong_label_rate']:.3f}"
                )

    print(f"\nSaved summary JSON to: {summary_json_path}")
    print(f"Saved per-image CSV to: {summary_csv_path}")
    if args.keep_per_image_json:
        print(f"Kept per-image JSON files in: {per_image_json_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
