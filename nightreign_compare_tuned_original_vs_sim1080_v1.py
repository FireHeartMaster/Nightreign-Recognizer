#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the final tuned config separately on original references and simulated 1080p references, "
            "then print and save a side-by-side comparison."
        )
    )
    p.add_argument("--runner", type=Path, required=True, help="Path to nightreign_run_with_tuned_config_v1.py")
    p.add_argument("--backend", type=Path, required=True, help="Path to the recognizer backend script")
    p.add_argument("--config-json", type=Path, required=True, help="Path to the final tuned config JSON")
    p.add_argument("--package", type=Path, required=True, help="Path to the .npz recognition package")
    p.add_argument("--original-dir", type=Path, required=True, help="Directory containing the original numbered reference panels")
    p.add_argument("--sim1080-dir", type=Path, required=True, help="Directory containing the simulated 1080p numbered reference panels")
    p.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable to use")
    p.add_argument("--top", type=int, default=20, help="How many layouts the recognizer should rank")
    p.add_argument("--max-images", type=int, default=0, help="Optional cap per directory. 0 means all")
    p.add_argument("--output-dir", type=Path, default=Path("tuned_compare_output"), help="Directory for summary outputs")
    p.add_argument("--keep-per-image-json", action="store_true", help="Keep per-image JSON reports")
    return p.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_numbered_images(images_dir: Path, max_images: int = 0) -> List[Tuple[int, Path]]:
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
    if max_images > 0:
        items = items[:max_images]
    return items


def load_package_metadata(package_path: Path) -> Dict[str, Any]:
    with np.load(package_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    return metadata


def build_truth_layouts(metadata: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["layout_number"]): item for item in metadata["layouts"]}


def run_one_image(
    python_exe: str,
    runner: Path,
    backend: Path,
    config_json: Path,
    package: Path,
    panel_image: Path,
    json_out: Path,
    top: int,
) -> Tuple[int, str, str]:
    cmd = [
        python_exe,
        str(runner),
        "--backend", str(backend),
        "--config-json", str(config_json),
        "--package", str(package),
        "--panel-image", str(panel_image),
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


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def earth_margin(earth_ranking: Sequence[Dict[str, Any]]) -> Optional[float]:
    if len(earth_ranking) < 2:
        return None
    s1 = float(earth_ranking[0]["score"])
    s2 = float(earth_ranking[1]["score"])
    if not (math.isfinite(s1) and math.isfinite(s2)):
        return None
    return s1 - s2


def top1_margin(results: Sequence[Dict[str, Any]]) -> Optional[float]:
    if len(results) < 2:
        return None
    s1 = float(results[0]["total_score"])
    s2 = float(results[1]["total_score"])
    if not (math.isfinite(s1) and math.isfinite(s2)):
        return None
    return s1 - s2


def slot_quality_vs_truth(obs_slots: Sequence[Dict[str, Any]], gt_labels: Sequence[str]) -> Dict[str, Any]:
    out = {
        "slot_count": 0,
        "correct": 0,
        "unknown": 0,
        "wrong_label": 0,
        "false_empty": 0,
        "false_positive_empty": 0,
        "confusions": Counter(),
    }
    if len(obs_slots) != len(gt_labels):
        return out
    for obs, gt in zip(obs_slots, gt_labels):
        pred = str(obs["label"])
        gt = str(gt)
        out["slot_count"] += 1
        if pred == gt:
            out["correct"] += 1
        elif pred == "unknown":
            out["unknown"] += 1
        elif pred == "empty" and gt != "empty":
            out["false_empty"] += 1
            out["confusions"][f"{gt} -> empty"] += 1
        elif pred != "empty" and gt == "empty":
            out["false_positive_empty"] += 1
            out["confusions"][f"empty -> {pred}"] += 1
        else:
            out["wrong_label"] += 1
            out["confusions"][f"{gt} -> {pred}"] += 1
    return out


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    ok_rows = [r for r in rows if not r.get("error")]
    n = len(ok_rows)
    if n == 0:
        return {
            "cases": 0,
            "earth_top1_acc": None,
            "layout_top1_acc": None,
            "layout_top3_acc": None,
            "layout_top5_acc": None,
            "mrr": None,
            "avg_true_rank": None,
            "avg_earth_margin": None,
            "avg_top1_margin": None,
            "avg_slot_correct_rate": None,
            "avg_slot_unknown_rate": None,
            "avg_slot_wrong_label_rate": None,
            "avg_slot_false_empty_rate": None,
            "avg_slot_false_positive_empty_rate": None,
        }
    return {
        "cases": float(n),
        "earth_top1_acc": safe_mean([1.0 if r["earth_correct"] else 0.0 for r in ok_rows]),
        "layout_top1_acc": safe_mean([1.0 if r["layout_correct"] else 0.0 for r in ok_rows]),
        "layout_top3_acc": safe_mean([1.0 if (r["true_rank"] is not None and r["true_rank"] <= 3) else 0.0 for r in ok_rows]),
        "layout_top5_acc": safe_mean([1.0 if (r["true_rank"] is not None and r["true_rank"] <= 5) else 0.0 for r in ok_rows]),
        "mrr": safe_mean([(1.0 / r["true_rank"]) if r["true_rank"] else None for r in ok_rows]),
        "avg_true_rank": safe_mean([r["true_rank"] for r in ok_rows]),
        "avg_earth_margin": safe_mean([r["earth_margin"] for r in ok_rows]),
        "avg_top1_margin": safe_mean([r["top1_margin"] for r in ok_rows]),
        "avg_slot_correct_rate": safe_mean([r["slot_correct_rate"] for r in ok_rows]),
        "avg_slot_unknown_rate": safe_mean([r["slot_unknown_rate"] for r in ok_rows]),
        "avg_slot_wrong_label_rate": safe_mean([r["slot_wrong_label_rate"] for r in ok_rows]),
        "avg_slot_false_empty_rate": safe_mean([r["slot_false_empty_rate"] for r in ok_rows]),
        "avg_slot_false_positive_empty_rate": safe_mean([r["slot_false_positive_empty_rate"] for r in ok_rows]),
    }


def fmt_num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_side_by_side(original_summary: Dict[str, Optional[float]], sim_summary: Dict[str, Optional[float]]) -> None:
    metrics = [
        ("cases", 0),
        ("earth_top1_acc", 4),
        ("layout_top1_acc", 4),
        ("layout_top3_acc", 4),
        ("layout_top5_acc", 4),
        ("mrr", 4),
        ("avg_true_rank", 3),
        ("avg_earth_margin", 4),
        ("avg_top1_margin", 4),
        ("avg_slot_correct_rate", 4),
        ("avg_slot_unknown_rate", 4),
        ("avg_slot_wrong_label_rate", 4),
        ("avg_slot_false_empty_rate", 4),
        ("avg_slot_false_positive_empty_rate", 4),
    ]
    left_title = "Original refs"
    right_title = "Simulated 1080p refs"
    print("\n========== SIDE-BY-SIDE COMPARISON ==========")
    print(f"{'metric':32} {left_title:>18} {right_title:>22} {'sim-original':>16}")
    print("-" * 94)
    for key, digits in metrics:
        lv = original_summary.get(key)
        rv = sim_summary.get(key)
        if key == "cases":
            left = "n/a" if lv is None else str(int(lv))
            right = "n/a" if rv is None else str(int(rv))
            diff = "n/a" if lv is None or rv is None else str(int(rv - lv))
        else:
            left = fmt_num(lv, digits)
            right = fmt_num(rv, digits)
            diff = "n/a" if lv is None or rv is None else f"{(rv - lv):+.{digits}f}"
        print(f"{key:32} {left:>18} {right:>22} {diff:>16}")
    print()


def evaluate_directory(
    source_tag: str,
    images_dir: Path,
    args: argparse.Namespace,
    truth_layouts: Dict[int, Dict[str, Any]],
    per_image_json_dir: Path,
) -> Tuple[List[Dict[str, Any]], Counter, Counter]:
    numbered_images = list_numbered_images(images_dir, args.max_images)
    rows: List[Dict[str, Any]] = []
    confusion_counter = Counter()
    wrong_top1_counter = Counter()

    print(f"Evaluating {source_tag}: {len(numbered_images)} images from {images_dir}")
    for idx, (layout_number, image_path) in enumerate(numbered_images, start=1):
        gt_layout = truth_layouts.get(layout_number)
        if gt_layout is None:
            print(f"[{source_tag} {idx:03d}/{len(numbered_images)}] Skipping {image_path.name}: no ground truth layout in package")
            continue

        json_out = per_image_json_dir / source_tag / f"{layout_number}.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)

        rc, stdout, stderr = run_one_image(
            python_exe=args.python_exe,
            runner=args.runner,
            backend=args.backend,
            config_json=args.config_json,
            package=args.package,
            panel_image=image_path,
            json_out=json_out,
            top=args.top,
        )

        if rc != 0:
            row = {
                "source_tag": source_tag,
                "layout_number": layout_number,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "truth_earth": gt_layout["shifting_earth"],
                "pred_earth": None,
                "earth_correct": False,
                "top1_layout": None,
                "layout_correct": False,
                "true_rank": None,
                "earth_margin": None,
                "top1_margin": None,
                "slot_correct_rate": None,
                "slot_unknown_rate": None,
                "slot_wrong_label_rate": None,
                "slot_false_empty_rate": None,
                "slot_false_positive_empty_rate": None,
                "error": True,
                "error_returncode": rc,
                "error_stdout": stdout,
                "error_stderr": stderr,
            }
            rows.append(row)
            print(f"[{source_tag} {idx:03d}/{len(numbered_images)}] {image_path.name}: ERROR rc={rc}")
            continue

        payload = json.loads(json_out.read_text(encoding="utf-8"))
        truth_earth = str(gt_layout["shifting_earth"])
        earth_ranking = payload.get("earth_ranking", [])
        pred_earth = earth_ranking[0]["name"] if earth_ranking else None
        earth_ok = (pred_earth == truth_earth)

        results = payload.get("results", [])
        top1_layout_num = int(results[0]["layout_number"]) if results else None
        rank = find_rank(results, layout_number)
        layout_ok = (top1_layout_num == layout_number)
        if not layout_ok and top1_layout_num is not None:
            wrong_top1_counter[top1_layout_num] += 1

        obs_slots = payload.get("observations_by_earth", {}).get(truth_earth, {}).get("slots", [])
        slot_eval = slot_quality_vs_truth(obs_slots, gt_layout["slot_labels"])
        confusion_counter.update(slot_eval["confusions"])
        slot_n = slot_eval["slot_count"] if slot_eval["slot_count"] > 0 else None

        row = {
            "source_tag": source_tag,
            "layout_number": layout_number,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "truth_earth": truth_earth,
            "pred_earth": pred_earth,
            "earth_correct": earth_ok,
            "top1_layout": top1_layout_num,
            "layout_correct": layout_ok,
            "true_rank": rank,
            "earth_margin": earth_margin(earth_ranking),
            "top1_margin": top1_margin(results),
            "slot_correct_rate": (slot_eval["correct"] / slot_n) if slot_n else None,
            "slot_unknown_rate": (slot_eval["unknown"] / slot_n) if slot_n else None,
            "slot_wrong_label_rate": (slot_eval["wrong_label"] / slot_n) if slot_n else None,
            "slot_false_empty_rate": (slot_eval["false_empty"] / slot_n) if slot_n else None,
            "slot_false_positive_empty_rate": (slot_eval["false_positive_empty"] / slot_n) if slot_n else None,
            "error": False,
        }
        rows.append(row)

        status = "OK" if layout_ok else f"WRONG (got #{top1_layout_num}, true #{layout_number}, rank={rank})"
        print(f"[{source_tag} {idx:03d}/{len(numbered_images)}] {image_path.name}: {status}")

        if not args.keep_per_image_json:
            try:
                json_out.unlink(missing_ok=True)
            except Exception:
                pass

    return rows, confusion_counter, wrong_top1_counter


def write_per_image_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    fieldnames = [
        "source_tag", "layout_number", "image_name", "image_path",
        "truth_earth", "pred_earth", "earth_correct", "top1_layout",
        "layout_correct", "true_rank", "earth_margin", "top1_margin",
        "slot_correct_rate", "slot_unknown_rate", "slot_wrong_label_rate",
        "slot_false_empty_rate", "slot_false_positive_empty_rate", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_package_metadata(args.package)
    truth_layouts = build_truth_layouts(metadata)

    per_image_json_dir = output_dir / "per_image_json"
    per_image_json_dir.mkdir(parents=True, exist_ok=True)

    original_rows, original_confusions, original_wrong_top1 = evaluate_directory(
        source_tag="original",
        images_dir=args.original_dir,
        args=args,
        truth_layouts=truth_layouts,
        per_image_json_dir=per_image_json_dir,
    )

    print()
    sim_rows, sim_confusions, sim_wrong_top1 = evaluate_directory(
        source_tag="sim1080",
        images_dir=args.sim1080_dir,
        args=args,
        truth_layouts=truth_layouts,
        per_image_json_dir=per_image_json_dir,
    )

    original_summary = summarize_rows(original_rows)
    sim_summary = summarize_rows(sim_rows)

    print_side_by_side(original_summary, sim_summary)

    print("Top wrong top-1 predicted layouts (original):")
    for k, v in original_wrong_top1.most_common(10):
        print(f"  #{k}: {v}")
    print("Top wrong top-1 predicted layouts (sim1080):")
    for k, v in sim_wrong_top1.most_common(10):
        print(f"  #{k}: {v}")

    print("\nMost common label confusions (original):")
    for k, v in original_confusions.most_common(15):
        print(f"  {k}: {v}")
    print("Most common label confusions (sim1080):")
    for k, v in sim_confusions.most_common(15):
        print(f"  {k}: {v}")

    summary = {
        "original_dir": str(args.original_dir),
        "sim1080_dir": str(args.sim1080_dir),
        "original_summary": original_summary,
        "sim1080_summary": sim_summary,
        "difference_sim_minus_original": {
            key: (
                None
                if original_summary.get(key) is None or sim_summary.get(key) is None
                else float(sim_summary[key] - original_summary[key])
            )
            for key in original_summary.keys()
            if key != "cases"
        },
        "original_wrong_top1_layouts": original_wrong_top1.most_common(20),
        "sim1080_wrong_top1_layouts": sim_wrong_top1.most_common(20),
        "original_confusions": original_confusions.most_common(30),
        "sim1080_confusions": sim_confusions.most_common(30),
    }

    summary_json = output_dir / "compare_original_vs_sim1080_summary.json"
    per_image_csv = output_dir / "compare_original_vs_sim1080_per_image.csv"
    ensure_parent(summary_json)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_per_image_csv(per_image_csv, original_rows + sim_rows)

    print(f"\nSaved summary JSON to: {summary_json}")
    print(f"Saved per-image CSV to: {per_image_csv}")
    if args.keep_per_image_json:
        print(f"Kept per-image JSON in: {per_image_json_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
