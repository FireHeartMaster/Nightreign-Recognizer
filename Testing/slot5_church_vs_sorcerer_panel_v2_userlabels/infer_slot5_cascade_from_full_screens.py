import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


def _load_modules(repo_root: Path):
    # Map/panel detector backend
    sys.path.insert(0, str(repo_root))
    import recognize_nightreign_map_proto_v12_debugplus_radiusscale_fix as rec  # type: ignore

    # Slot CNN lib (same architecture used for both checkpoints)
    sys.path.insert(0, str(repo_root / "Testing" / "slot_5"))
    import layout_classifier_lib as lib  # type: ignore

    return rec, lib


def _slot5_crop_from_panel(panel_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    # Corrected slot-5 center in 1536-reference panel coordinates
    slot5_center_x_ref = 499.0
    slot5_center_y_ref = 809.0
    ref_panel = 1536.0
    model_side_ref = 200.0

    ph, pw = panel_bgr.shape[:2]
    native_side = max(8, int(round((model_side_ref / ref_panel) * pw)))
    cx = int(round((slot5_center_x_ref / ref_panel) * pw))
    cy = int(round((slot5_center_y_ref / ref_panel) * ph))

    x0 = int(round(cx - native_side / 2.0))
    y0 = int(round(cy - native_side / 2.0))
    x0 = max(0, min(pw - native_side, x0))
    y0 = max(0, min(ph - native_side, y0))
    x1 = x0 + native_side
    y1 = y0 + native_side

    crop = panel_bgr[y0:y1, x0:x1]
    return crop, {"x": x0, "y": y0, "size": native_side}


def _predict_on_crop(
    crop_bgr: np.ndarray,
    model: torch.nn.Module,
    class_names: List[str],
    image_size: int,
    device: torch.device,
    lib,
    interpolation: int = cv2.INTER_CUBIC,
) -> Dict[str, object]:
    model_input = cv2.resize(crop_bgr, (image_size, image_size), interpolation=interpolation)
    x = lib.preprocess_bgr_200(model_input, image_size=image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu()
        probs = torch.softmax(logits, dim=0).numpy()

    ranking = sorted(
        [{"label": class_names[i], "probability": float(probs[i])} for i in range(len(class_names))],
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "predicted_label": ranking[0]["label"],
        "confidence": float(ranking[0]["probability"]),
        "ranking": ranking,
        "model_input": model_input,
    }


def run(
    input_path: Path,
    output_dir: Path,
    empty_model_path: Path,
    occupied_model_path: Path,
    save_artifacts: bool,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    rec, lib = _load_modules(repo_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    empty_model, empty_classes, empty_size, empty_meta = lib.load_checkpoint(empty_model_path, device)
    occ_model, occ_classes, occ_size, occ_meta = lib.load_checkpoint(occupied_model_path, device)

    if input_path.is_dir():
        images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}])
    else:
        images = [input_path]

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for image_path in images:
        full = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if full is None:
            continue

        circle, method = rec.detect_map_circle(full)
        if method == "blue_hough":
            panel_box = rec.clip_box(rec.estimate_panel_box(circle, panel_scale=1.0), full.shape[1], full.shape[0])
        else:
            panel_box = rec.refine_panel_box(full, circle)

        panel = full[panel_box.y0:panel_box.y1, panel_box.x0:panel_box.x1]
        slot_crop, slot_rect = _slot5_crop_from_panel(panel)

        # Stage 1: empty vs occupied
        # Stage-1 must match the previous occupied/empty run behavior:
        # resize 194->200 with INTER_AREA.
        s1 = _predict_on_crop(
            slot_crop,
            empty_model,
            empty_classes,
            empty_size,
            device,
            lib,
            interpolation=cv2.INTER_AREA,
        )
        stage1_label = str(s1["predicted_label"])

        # Stage 2 only when occupied
        stage2 = None
        if stage1_label.lower() == "occupied":
            s2 = _predict_on_crop(
                slot_crop,
                occ_model,
                occ_classes,
                occ_size,
                device,
                lib,
                interpolation=cv2.INTER_CUBIC,
            )
            stage2 = {
                "predicted_label": s2["predicted_label"],
                "confidence": s2["confidence"],
                "ranking": s2["ranking"],
            }
            final_label = str(s2["predicted_label"])
            final_conf = float(s2["confidence"])
            final_source = "occupied_model"
            occ_input = s2["model_input"]
        else:
            final_label = "empty"
            final_conf = float(s1["confidence"])
            final_source = "empty_model"
            occ_input = None

        row = {
            "input_image": str(image_path),
            "panel_crop_box": {
                "x0": int(panel_box.x0),
                "y0": int(panel_box.y0),
                "x1": int(panel_box.x1),
                "y1": int(panel_box.y1),
            },
            "panel_shape": {"w": int(panel.shape[1]), "h": int(panel.shape[0])},
            "slot5_crop_rect": slot_rect,
            "stage1_empty_vs_occupied": {
                "predicted_label": s1["predicted_label"],
                "confidence": s1["confidence"],
                "ranking": s1["ranking"],
            },
            "stage2_church_vs_sorcerer": stage2,
            "final_label": final_label,
            "final_confidence": final_conf,
            "final_source": final_source,
        }

        if save_artifacts:
            stem = image_path.stem
            panel_path = output_dir / f"{stem}__panel_crop.png"
            slot_native_path = output_dir / f"{stem}__slot5_native_{slot_rect['size']}.png"
            s1_input_path = output_dir / f"{stem}__slot5_input_stage1_{empty_size}.png"
            overlay_path = output_dir / f"{stem}__panel_with_slot5_box.png"
            cv2.imwrite(str(panel_path), panel)
            cv2.imwrite(str(slot_native_path), slot_crop)
            cv2.imwrite(str(s1_input_path), s1["model_input"])

            if occ_input is not None:
                s2_input_path = output_dir / f"{stem}__slot5_input_stage2_{occ_size}.png"
                cv2.imwrite(str(s2_input_path), occ_input)
                row["stage2_model_input"] = str(s2_input_path)

            overlay = panel.copy()
            x0 = slot_rect["x"]
            y0 = slot_rect["y"]
            sz = slot_rect["size"]
            cv2.rectangle(overlay, (x0, y0), (x0 + sz - 1, y0 + sz - 1), (80, 220, 120), 2)
            cv2.putText(
                overlay,
                f"slot5 {final_label} {final_conf:.4f}",
                (max(8, x0 - 4), max(20, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 220, 120),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(overlay_path), overlay)

            row["panel_path"] = str(panel_path)
            row["slot5_native_crop"] = str(slot_native_path)
            row["stage1_model_input"] = str(s1_input_path)
            row["slot5_overlay"] = str(overlay_path)

        results.append(row)

    report = {
        "mode": "slot5_two_stage_cascade",
        "rule": "if stage1 predicts empty -> final empty; if occupied -> use stage2 Church/Sorcerers_Rise",
        "empty_model_path": str(empty_model_path),
        "occupied_model_path": str(occupied_model_path),
        "empty_model_classes": empty_classes,
        "occupied_model_classes": occ_classes,
        "empty_model_meta": empty_meta,
        "occupied_model_meta": occ_meta,
        "results": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage slot 5 cascade from full screenshots.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input screenshot file or folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parent / "cascade_test_outputs"
        ),
        help="Where to save report and optional artifacts.",
    )
    parser.add_argument(
        "--empty-model",
        default=str(
            Path(__file__).resolve().parents[1]
            / "slot5_authoritative_binary_panel_v1"
            / "slot5_occupied_empty_authoritative_v1.pt"
        ),
        help="Stage-1 empty/occupied model.",
    )
    parser.add_argument(
        "--occupied-model",
        default=str(
            Path(__file__).resolve().parent / "slot5_church_vs_sorcerer_v2.pt"
        ),
        help="Stage-2 Church/Sorcerers_Rise model.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not save panel/crop/overlay images.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    empty_model_path = Path(args.empty_model).resolve()
    occupied_model_path = Path(args.occupied_model).resolve()

    report = run(
        input_path=input_path,
        output_dir=output_dir,
        empty_model_path=empty_model_path,
        occupied_model_path=occupied_model_path,
        save_artifacts=not bool(args.no_artifacts),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "slot5_cascade_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for row in report["results"]:
        name = Path(row["input_image"]).name
        print(
            f"{name}\tfinal={row['final_label']}\t"
            f"stage1={row['stage1_empty_vs_occupied']['predicted_label']}"
            f"({row['stage1_empty_vs_occupied']['confidence']:.4f})"
        )
    print(f"REPORT\t{report_path}")


if __name__ == "__main__":
    main()
