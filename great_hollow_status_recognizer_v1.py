#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from icon_recognizer_v4 import iter_images, recognize_markers

DEFAULT_REF_FIRE_ROT = (
    Path("layout_1007.png"),
    Path("renders") / "recognizer_no_text" / "layout_1007.png",
)
DEFAULT_REF_MAGIC_HOLY = (
    Path("layout_1005.png"),
    Path("renders") / "recognizer_no_text" / "layout_1005.png",
)


def find_first_existing(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the reference paths exist: {', '.join(str(p) for p in candidates)}")


def remap_result(raw: dict) -> dict:
    top_left = raw["top_left"]
    bottom_right = raw["bottom_right"]
    return {
        "Palace": {
            "label": top_left["label"],
            "score": top_left["score"],
            "runner_up": top_left["runner_up"],
            "runner_up_score": top_left["runner_up_score"],
            "margin": top_left["margin"],
            "confidence": top_left["confidence"],
            "box": top_left["box"],
            "ranked": top_left["ranked"],
        },
        "Fortress": {
            "label": bottom_right["label"],
            "score": bottom_right["score"],
            "runner_up": bottom_right["runner_up"],
            "runner_up_score": bottom_right["runner_up_score"],
            "margin": bottom_right["margin"],
            "confidence": bottom_right["confidence"],
            "box": bottom_right["box"],
            "ranked": bottom_right["ranked"],
        },
    }


def recognize_full_map(query_path: Path, save_debug: Path | None = None, save_annotated: Path | None = None) -> dict:
    ref_fire_rot = find_first_existing(DEFAULT_REF_FIRE_ROT)
    ref_magic_holy = find_first_existing(DEFAULT_REF_MAGIC_HOLY)
    raw = recognize_markers(
        str(ref_fire_rot),
        str(ref_magic_holy),
        str(query_path),
        save_debug=None if save_debug is None else str(save_debug),
        save_annotated=None if save_annotated is None else str(save_annotated),
    )
    return remap_result(raw)


def run_folder_mode(input_folder: Path, output_csv: Path, output_json: Path | None, save_debug: Path | None, save_annotated: Path | None, recursive: bool) -> None:
    images = list(iter_images(input_folder, recursive=recursive))
    if not images:
        raise SystemExit("No image files found in the input folder.")

    rows = []
    detailed = []
    for img_path in images:
        try:
            debug_dir = save_debug / img_path.relative_to(input_folder).with_suffix("") if save_debug else None
            annotated_path = save_annotated / img_path.relative_to(input_folder) if save_annotated else None
            result = recognize_full_map(img_path, save_debug=debug_dir, save_annotated=annotated_path)
            palace = result["Palace"]
            fortress = result["Fortress"]
            row = {
                "file": str(img_path.relative_to(input_folder)),
                "Palace": palace["label"],
                "Palace_score": palace["score"],
                "Palace_margin": palace["margin"],
                "Palace_confidence": palace["confidence"],
                "Fortress": fortress["label"],
                "Fortress_score": fortress["score"],
                "Fortress_margin": fortress["margin"],
                "Fortress_confidence": fortress["confidence"],
                "status": "ok",
                "error": "",
            }
            rows.append(row)
            detailed.append({"file": row["file"], "status": "ok", "result": result})
            print(f"[OK] {row['file']} -> Palace={row['Palace']} ({row['Palace_confidence']}), Fortress={row['Fortress']} ({row['Fortress_confidence']})")
        except Exception as exc:
            row = {
                "file": str(img_path.relative_to(input_folder)),
                "Palace": "",
                "Palace_score": "",
                "Palace_margin": "",
                "Palace_confidence": "",
                "Fortress": "",
                "Fortress_score": "",
                "Fortress_margin": "",
                "Fortress_confidence": "",
                "status": "error",
                "error": str(exc),
            }
            rows.append(row)
            detailed.append({"file": row["file"], "status": "error", "error": str(exc)})
            print(f"[ERROR] {row['file']} -> {exc}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "Palace",
                "Palace_score",
                "Palace_margin",
                "Palace_confidence",
                "Fortress",
                "Fortress_score",
                "Fortress_margin",
                "Fortress_confidence",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    err_count = len(rows) - ok_count
    print(f"\nDone. Processed {len(rows)} image(s): {ok_count} ok, {err_count} error(s).")
    print(f"CSV written to: {output_csv}")
    if output_json is not None:
        print(f"JSON written to: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize Great Hollow Palace/Fortress status icons from a full map image.")
    parser.add_argument("input_path", help="Full map image to classify, or a folder of images")
    parser.add_argument("--save-debug", default=None, help="Optional folder for aligned-image debug output")
    parser.add_argument("--save-annotated", default=None, help="Optional output image or folder for annotated results")
    parser.add_argument("--output-csv", default="great_hollow_status_results.csv", help="In folder mode, write summary CSV here")
    parser.add_argument("--output-json", default=None, help="In folder mode, optionally write detailed JSON here")
    parser.add_argument("--recursive", action="store_true", help="In folder mode, also scan subfolders")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    save_debug = None if args.save_debug is None else Path(args.save_debug)
    save_annotated = None if args.save_annotated is None else Path(args.save_annotated)

    if input_path.is_dir():
        run_folder_mode(
            input_path,
            Path(args.output_csv),
            None if args.output_json is None else Path(args.output_json),
            save_debug,
            save_annotated,
            args.recursive,
        )
        return

    result = recognize_full_map(input_path, save_debug=save_debug, save_annotated=save_annotated)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
