import argparse
import importlib.util
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch


PANEL_REF = 1536.0
GH_ASYM_INPUT = 1536
GH_ASYM_CROP = {"left": 175, "right": 170, "top": 180, "bottom": 180}

# In-code model switches (enabled by default):
# Prefer new GH-variation retrained models for Palace/Fortress/slot2.
USE_SLOT2_GH_VARIATION_MODEL = True
USE_PALACE_GH_VARIATION_MODEL = True
USE_FORTRESS_GH_VARIATION_MODEL = True
# Slot5 model policy:
# - Default to the userlabels single-stage model because it better matches recent real-screen behavior.
# - Keep the previous cascade path available behind a flag.
USE_SLOT5_SINGLE_STAGE_USERLABELS_MODEL = True
# Slot5 single-stage model selection policy:
# - When enabled, prefer the newer GH retrained checkpoint first.
# - Keep older single-stage checkpoints as fallback.
USE_SLOT5_GH_RETRAINED_MODEL = True
# Slot2 crop horizontal offset policy:
# - New GH-variation model was trained on fixed slot-2 crops without +50px horizontal shift.
# - Legacy slot2_status_cnn_offset50_v1 expects +50px shift.
SLOT2_CROP_DX_WHEN_GH_VARIATION_MODEL = 0
SLOT2_CROP_DX_WHEN_LEGACY_MODEL = 50


@dataclass(frozen=True)
class ModelSpec:
    name: str
    lib_path: Path
    model_path: Path


def _load_lib_module(lib_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(lib_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {lib_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_existing_path(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


def _crop_from_panel_ref(
    panel_bgr: np.ndarray,
    center_x_ref: float,
    center_y_ref: float,
    model_side_ref: float,
    dx_px: int = 0,
    dy_px: int = 0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    ph, pw = panel_bgr.shape[:2]
    native_side = max(8, int(round((float(model_side_ref) / PANEL_REF) * pw)))
    cx = int(round((float(center_x_ref) / PANEL_REF) * pw)) + int(dx_px)
    cy = int(round((float(center_y_ref) / PANEL_REF) * ph)) + int(dy_px)
    return _crop_center(panel_bgr, cx, cy, native_side, extra={"dx": int(dx_px), "dy": int(dy_px)})


def _crop_from_panel_norm(
    panel_bgr: np.ndarray,
    center_x_norm: float,
    center_y_norm: float,
    model_side_ref: float,
    dx_px: int = 0,
    dy_px: int = 0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    ph, pw = panel_bgr.shape[:2]
    native_side = max(8, int(round((float(model_side_ref) / PANEL_REF) * pw)))
    cx = int(round(float(center_x_norm) * pw)) + int(dx_px)
    cy = int(round(float(center_y_norm) * ph)) + int(dy_px)
    return _crop_center(panel_bgr, cx, cy, native_side, extra={"dx": int(dx_px), "dy": int(dy_px)})


def _crop_center(
    img_bgr: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    extra: Dict[str, int] | None = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    h, w = img_bgr.shape[:2]
    side = max(8, int(size))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = max(0, min(w - side, x0))
    y0 = max(0, min(h - side, y0))
    x1 = x0 + side
    y1 = y0 + side
    crop = img_bgr[y0:y1, x0:x1]
    rect = {"x": int(x0), "y": int(y0), "size": int(side), "cx": int(cx), "cy": int(cy)}
    if extra:
        rect.update({k: int(v) for k, v in extra.items()})
    return crop, rect


def _preprocess_gh_asym(panel_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(panel_bgr, (GH_ASYM_INPUT, GH_ASYM_INPUT), interpolation=cv2.INTER_CUBIC)
    l = int(GH_ASYM_CROP["left"])
    r = int(GH_ASYM_CROP["right"])
    t = int(GH_ASYM_CROP["top"])
    b = int(GH_ASYM_CROP["bottom"])
    y0 = max(0, t)
    y1 = min(GH_ASYM_INPUT, GH_ASYM_INPUT - b)
    x0 = max(0, l)
    x1 = min(GH_ASYM_INPUT, GH_ASYM_INPUT - r)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Invalid GH asym crop configuration.")
    return resized[y0:y1, x0:x1].copy()


def _predict_from_crop(
    crop_bgr: np.ndarray,
    lib: Any,
    model: torch.nn.Module,
    class_names: List[str],
    image_size: int,
    device: torch.device,
    interpolation: int,
) -> Dict[str, Any]:
    model_input = cv2.resize(crop_bgr, (image_size, image_size), interpolation=interpolation)
    x = lib.preprocess_bgr_200(model_input, image_size=image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu()
        probs = torch.softmax(logits, dim=0).numpy()
    ranking = sorted(
        [{"label": class_names[i], "probability": float(probs[i])} for i in range(len(class_names))],
        key=lambda it: it["probability"],
        reverse=True,
    )
    return {
        "predicted_label": str(ranking[0]["label"]),
        "confidence": float(ranking[0]["probability"]),
        "ranking": ranking,
    }


def _load_bundle(spec: ModelSpec, device: torch.device, module_tag: str) -> Tuple[Any, torch.nn.Module, List[str], int, Dict[str, Any]]:
    lib = _load_lib_module(spec.lib_path, module_tag)
    model, class_names, image_size, meta = lib.load_checkpoint(spec.model_path, device)
    return lib, model, list(class_names), int(image_size), dict(meta or {})


def _task_slot2(
    panel_bgr: np.ndarray,
    repo_root: Path,
    device: torch.device,
    model_path: Path,
    crop_dx_px: int,
) -> Dict[str, Any]:
    spec = ModelSpec(
        name="slot2",
        lib_path=repo_root / "Testing" / "slot_2" / "layout_classifier_lib.py",
        model_path=model_path,
    )
    lib, model, class_names, image_size, meta = _load_bundle(spec, device, "slot2_lib_parallel")
    crop, rect = _crop_from_panel_norm(
        panel_bgr,
        0.82890625,
        0.38046875,
        model_side_ref=260.0,
        dx_px=int(crop_dx_px),
        dy_px=0,
    )
    pred = _predict_from_crop(crop, lib, model, class_names, image_size, device, interpolation=cv2.INTER_AREA)
    return {
        "classifier": "slot2_great_church_status",
        "type": "single_stage",
        "crop_source": "panel_raw",
        "model_path": str(spec.model_path),
        "crop_dx_px": int(crop_dx_px),
        "model_meta": meta,
        "crop_rect": rect,
        **pred,
    }


def _task_panel_single(
    panel_bgr: np.ndarray,
    device: torch.device,
    classifier_key: str,
    lib_path: Path,
    model_path: Path,
    center_ref_x: float,
    center_ref_y: float,
    model_side_ref: float,
    dx_px: int = 0,
    dy_px: int = 0,
    interp: int = cv2.INTER_CUBIC,
) -> Dict[str, Any]:
    spec = ModelSpec(
        name=classifier_key,
        lib_path=lib_path,
        model_path=model_path,
    )
    lib, model, class_names, image_size, meta = _load_bundle(spec, device, f"{classifier_key}_panel_single_lib_parallel")
    crop, rect = _crop_from_panel_ref(
        panel_bgr,
        center_x_ref=center_ref_x,
        center_y_ref=center_ref_y,
        model_side_ref=model_side_ref,
        dx_px=dx_px,
        dy_px=dy_px,
    )
    pred = _predict_from_crop(crop, lib, model, class_names, image_size, device, interpolation=interp)
    return {
        "classifier": classifier_key,
        "type": "single_stage",
        "crop_source": "panel_raw",
        "model_path": str(spec.model_path),
        "model_meta": meta,
        "crop_rect": rect,
        **pred,
    }


def _task_cascade(
    panel_bgr: np.ndarray,
    device: torch.device,
    slot_name: str,
    lib_path: Path,
    stage1_model: Path,
    stage2_model: Path,
    center_ref_x: float,
    center_ref_y: float,
    model_side_ref: float,
    dx_px: int = 0,
    dy_px: int = 0,
    stage1_interp: int = cv2.INTER_AREA,
    stage2_interp: int = cv2.INTER_AREA,
    empty_accept_confidence: float = 0.5,
) -> Dict[str, Any]:
    stage1_spec = ModelSpec(name=f"{slot_name}_stage1", lib_path=lib_path, model_path=stage1_model)
    stage2_spec = ModelSpec(name=f"{slot_name}_stage2", lib_path=lib_path, model_path=stage2_model)

    lib1, model1, classes1, size1, meta1 = _load_bundle(stage1_spec, device, f"{slot_name}_lib_stage1_parallel")
    lib2, model2, classes2, size2, meta2 = _load_bundle(stage2_spec, device, f"{slot_name}_lib_stage2_parallel")

    crop, rect = _crop_from_panel_ref(
        panel_bgr,
        center_x_ref=center_ref_x,
        center_y_ref=center_ref_y,
        model_side_ref=model_side_ref,
        dx_px=dx_px,
        dy_px=dy_px,
    )

    s1 = _predict_from_crop(crop, lib1, model1, classes1, size1, device, interpolation=stage1_interp)
    s1_label = str(s1["predicted_label"]).lower()
    s1_conf = float(s1["confidence"])
    empty_gate = float(empty_accept_confidence)

    stage2_payload: Dict[str, Any] | None = None
    if s1_label == "occupied" or (s1_label == "empty" and s1_conf < empty_gate):
        s2 = _predict_from_crop(crop, lib2, model2, classes2, size2, device, interpolation=stage2_interp)
        stage2_payload = s2
        final_label = str(s2["predicted_label"])
        final_conf = float(s2["confidence"])
        final_source = "stage2" if s1_label == "occupied" else "stage2_after_empty_override"
    else:
        final_label = "empty"
        final_conf = s1_conf
        final_source = "stage1"

    return {
        "classifier": slot_name,
        "type": "two_stage_cascade",
        "crop_source": "panel_raw",
        "crop_rect": rect,
        "stage1_model_path": str(stage1_model),
        "stage1_model_meta": meta1,
        "empty_accept_confidence": empty_gate,
        "stage1": s1,
        "stage2_model_path": str(stage2_model),
        "stage2_model_meta": meta2,
        "stage2": stage2_payload,
        "final_label": final_label,
        "final_confidence": final_conf,
        "final_source": final_source,
    }


def _task_status_single(
    gh_asym_bgr: np.ndarray,
    device: torch.device,
    classifier_key: str,
    lib_path: Path,
    model_path: Path,
    center_x_asym: int,
    center_y_asym: int,
    crop_size: int = 200,
) -> Dict[str, Any]:
    spec = ModelSpec(name=classifier_key, lib_path=lib_path, model_path=model_path)
    lib, model, class_names, image_size, meta = _load_bundle(spec, device, f"{classifier_key}_lib_parallel")
    crop, rect = _crop_center(gh_asym_bgr, center_x_asym, center_y_asym, crop_size)
    pred = _predict_from_crop(crop, lib, model, class_names, image_size, device, interpolation=cv2.INTER_AREA)
    return {
        "classifier": classifier_key,
        "type": "single_stage",
        "crop_source": "gh_asym_legacy",
        "model_path": str(model_path),
        "model_meta": meta,
        "crop_rect": rect,
        **pred,
    }


def run_all_from_panel_bgr(
    panel_bgr: np.ndarray,
    panel_source: str,
    output_json: Path | None,
    device_name: str,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if panel_bgr is None or panel_bgr.size == 0:
        raise ValueError("panel_bgr is empty")
    gh_asym = _preprocess_gh_asym(panel_bgr)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    slot2_model = _resolve_existing_path(
        (
            [
                repo_root / "Testing" / "slot_2_fixed_geometry_panel_bases260" / "slot2_status_cnn_gh_variations_retrain_latest.pt",
                repo_root / "Testing" / "slot_2_fixed_geometry_panel_bases260" / "slot2_status_cnn_offset50_v1.pt",
                repo_root / "Testing" / "slot_2" / "slot2_status_cnn.pt",
            ]
            if USE_SLOT2_GH_VARIATION_MODEL
            else [
                repo_root / "Testing" / "slot_2_fixed_geometry_panel_bases260" / "slot2_status_cnn_offset50_v1.pt",
                repo_root / "Testing" / "slot_2" / "slot2_status_cnn.pt",
                repo_root / "Testing" / "slot_2_fixed_geometry_panel_bases260" / "slot2_status_cnn_gh_variations_retrain_latest.pt",
            ]
        )
    )
    slot2_model_name = slot2_model.name.lower()
    if "gh_variations" in slot2_model_name:
        slot2_crop_dx_px = int(SLOT2_CROP_DX_WHEN_GH_VARIATION_MODEL)
    else:
        slot2_crop_dx_px = int(SLOT2_CROP_DX_WHEN_LEGACY_MODEL)

    palace_model = _resolve_existing_path(
        (
            [
                repo_root / "Testing" / "palace" / "palace_status_cnn_gh_variations_retrain_latest.pt",
                repo_root / "Testing" / "palace" / "palace_status_cnn_retrain_latest.pt",
                repo_root / "Testing" / "palace" / "palace_status_cnn.pt",
            ]
            if USE_PALACE_GH_VARIATION_MODEL
            else [
                repo_root / "Testing" / "palace" / "palace_status_cnn_retrain_latest.pt",
                repo_root / "Testing" / "palace" / "palace_status_cnn.pt",
                repo_root / "Testing" / "palace" / "palace_status_cnn_gh_variations_retrain_latest.pt",
            ]
        )
    )
    fortress_model = _resolve_existing_path(
        (
            [
                repo_root / "Testing" / "fortress" / "fortress_status_cnn_gh_variations_retrain_latest.pt",
                repo_root / "Testing" / "fortress" / "fortress_status_cnn_retrain_latest.pt",
                repo_root / "Testing" / "fortress" / "fortress_status_cnn.pt",
            ]
            if USE_FORTRESS_GH_VARIATION_MODEL
            else [
                repo_root / "Testing" / "fortress" / "fortress_status_cnn_retrain_latest.pt",
                repo_root / "Testing" / "fortress" / "fortress_status_cnn.pt",
                repo_root / "Testing" / "fortress" / "fortress_status_cnn_gh_variations_retrain_latest.pt",
            ]
        )
    )
    slot5_single_stage_model_candidates = (
        [
            repo_root / "Testing" / "slot_5_userlabels_panel6_iso" / "slot5_status_cnn_userlabels_panel6_iso_ghslot5_3000x3_20260422.pt",
            repo_root / "Testing" / "slot_5_userlabels_panel6_iso" / "slot5_status_cnn_userlabels_panel6_iso.pt",
            repo_root / "Testing" / "slot_5_plus_real_heavy" / "slot5_status_cnn_plus_real_heavy.pt",
            repo_root / "Testing" / "slot_5" / "slot5_status_cnn.pt",
        ]
        if USE_SLOT5_GH_RETRAINED_MODEL
        else [
            repo_root / "Testing" / "slot_5_userlabels_panel6_iso" / "slot5_status_cnn_userlabels_panel6_iso.pt",
            repo_root / "Testing" / "slot_5_plus_real_heavy" / "slot5_status_cnn_plus_real_heavy.pt",
            repo_root / "Testing" / "slot_5" / "slot5_status_cnn.pt",
            repo_root / "Testing" / "slot_5_userlabels_panel6_iso" / "slot5_status_cnn_userlabels_panel6_iso_ghslot5_3000x3_20260422.pt",
        ]
    )
    slot5_single_stage_model = _resolve_existing_path(slot5_single_stage_model_candidates)

    if USE_SLOT5_SINGLE_STAGE_USERLABELS_MODEL:
        slot5_task = lambda: _task_panel_single(
            panel_bgr=panel_bgr,
            device=device,
            classifier_key="slot5_minor",
            lib_path=repo_root / "Testing" / "slot_5" / "layout_classifier_lib.py",
            model_path=slot5_single_stage_model,
            center_ref_x=499.0,
            center_ref_y=809.0,
            model_side_ref=200.0,
            interp=cv2.INTER_CUBIC,
        )
    else:
        slot5_task = lambda: _task_cascade(
            panel_bgr=panel_bgr,
            device=device,
            slot_name="slot5_minor",
            lib_path=repo_root / "Testing" / "slot_5" / "layout_classifier_lib.py",
            stage1_model=repo_root / "Testing" / "slot5_authoritative_binary_panel_v1" / "slot5_occupied_empty_authoritative_v1.pt",
            stage2_model=repo_root / "Testing" / "slot5_church_vs_sorcerer_panel_v2_userlabels" / "slot5_church_vs_sorcerer_v2.pt",
            center_ref_x=499.0,
            center_ref_y=809.0,
            model_side_ref=200.0,
            stage1_interp=cv2.INTER_AREA,
            stage2_interp=cv2.INTER_CUBIC,
        )

    tasks = {
        "slot2": lambda: _task_slot2(panel_bgr, repo_root, device, slot2_model, slot2_crop_dx_px),
        "slot4": lambda: _task_cascade(
            panel_bgr=panel_bgr,
            device=device,
            slot_name="slot4_minor",
            lib_path=repo_root / "Testing" / "slot_4" / "layout_classifier_lib.py",
            stage1_model=repo_root / "Testing" / "slot_4_fixed_geometry_panel_bases260" / "separated_occupied_empty" / "slot4_occupied_empty_from_panel_fixed_v1.pt",
            stage2_model=repo_root / "Testing" / "slot_4_fixed_geometry_panel_bases260" / "slot4_church_vs_sorcerer_bases260" / "slot4_church_vs_sorcerer_v1.pt",
            # Fixed slot-4 center in panel-1536 coordinates (matches bases manifest).
            center_ref_x=984.0,
            center_ref_y=805.0,
            model_side_ref=260.0,
            # Keep slot-4 on fixed panel geometry without legacy ad-hoc offset.
            dx_px=0,
            dy_px=0,
            stage1_interp=cv2.INTER_AREA,
            stage2_interp=cv2.INTER_AREA,
            # Slot 4 stage-1 tends to over-predict empty on real screenshots.
            # Accept empty only when stage-1 confidence is very high.
            empty_accept_confidence=0.995,
        ),
        "slot5": slot5_task,
        "slot7": lambda: _task_cascade(
            panel_bgr=panel_bgr,
            device=device,
            slot_name="slot7_minor",
            lib_path=repo_root / "Testing" / "slot_7" / "layout_classifier_lib.py",
            stage1_model=repo_root / "Testing" / "slot_7_fixed_geometry_panel_bases260" / "separated_occupied_empty" / "slot7_occupied_empty_from_panel_fixed_v1.pt",
            stage2_model=repo_root / "Testing" / "slot_7_fixed_geometry_panel_bases260" / "slot7_church_vs_sorcerer_bases260" / "slot7_church_vs_sorcerer_v1.pt",
            center_ref_x=514.0,
            center_ref_y=1265.0,
            model_side_ref=260.0,
            stage1_interp=cv2.INTER_CUBIC,
            stage2_interp=cv2.INTER_CUBIC,
        ),
        "slot6": lambda: _task_status_single(
            gh_asym_bgr=gh_asym,
            device=device,
            classifier_key="slot6_ruins_status",
            lib_path=repo_root / "Testing" / "slot_6" / "layout_classifier_lib.py",
            model_path=repo_root / "Testing" / "slot_6" / "slot6_status_cnn.pt",
            center_x_asym=144,
            center_y_asym=731,
            crop_size=200,
        ),
        "palace": lambda: _task_status_single(
            gh_asym_bgr=gh_asym,
            device=device,
            classifier_key="palace_status",
            lib_path=repo_root / "Testing" / "palace" / "layout_classifier_lib.py",
            model_path=palace_model,
            center_x_asym=329,
            center_y_asym=251,
            crop_size=200,
        ),
        "fortress": lambda: _task_status_single(
            gh_asym_bgr=gh_asym,
            device=device,
            classifier_key="fortress_status",
            lib_path=repo_root / "Testing" / "fortress" / "layout_classifier_lib.py",
            model_path=fortress_model,
            center_x_asym=875,
            center_y_asym=897,
            crop_size=200,
        ),
    }

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        fut_to_key = {ex.submit(fn): key for key, fn in tasks.items()}
        for fut in as_completed(fut_to_key):
            key = fut_to_key[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:  # pragma: no cover
                errors[key] = f"{exc}\n{traceback.format_exc()}"

    ordered = {k: results[k] for k in ["slot2", "slot4", "slot5", "slot6", "slot7", "palace", "fortress"] if k in results}
    payload = {
        "mode": "parallel_panel_classifiers",
        "input_panel": str(panel_source),
        "panel_shape": {"w": int(panel_bgr.shape[1]), "h": int(panel_bgr.shape[0])},
        "gh_asym_shape": {"w": int(gh_asym.shape[1]), "h": int(gh_asym.shape[0])},
        "device": str(device),
        "model_selection_flags": {
            "USE_SLOT2_GH_VARIATION_MODEL": bool(USE_SLOT2_GH_VARIATION_MODEL),
            "USE_PALACE_GH_VARIATION_MODEL": bool(USE_PALACE_GH_VARIATION_MODEL),
            "USE_FORTRESS_GH_VARIATION_MODEL": bool(USE_FORTRESS_GH_VARIATION_MODEL),
            "USE_SLOT5_SINGLE_STAGE_USERLABELS_MODEL": bool(USE_SLOT5_SINGLE_STAGE_USERLABELS_MODEL),
            "USE_SLOT5_GH_RETRAINED_MODEL": bool(USE_SLOT5_GH_RETRAINED_MODEL),
            "SLOT2_CROP_DX_WHEN_GH_VARIATION_MODEL": int(SLOT2_CROP_DX_WHEN_GH_VARIATION_MODEL),
            "SLOT2_CROP_DX_WHEN_LEGACY_MODEL": int(SLOT2_CROP_DX_WHEN_LEGACY_MODEL),
        },
        "resolved_models": {
            "slot2": str(slot2_model),
            "palace": str(palace_model),
            "fortress": str(fortress_model),
            "slot5_single_stage": str(slot5_single_stage_model),
        },
        "resolved_slot2_crop_dx_px": int(slot2_crop_dx_px),
        "classifiers": ordered,
        "errors": errors,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def run_all(panel_path: Path, output_json: Path | None, device_name: str) -> Dict[str, Any]:
    panel_bgr = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
    if panel_bgr is None:
        raise FileNotFoundError(f"Could not read panel image: {panel_path}")
    return run_all_from_panel_bgr(
        panel_bgr=panel_bgr,
        panel_source=str(panel_path),
        output_json=output_json,
        device_name=device_name,
    )


def _print_summary(payload: Dict[str, Any]) -> None:
    print(f"INPUT\t{payload['input_panel']}")
    print(f"PANEL\t{payload['panel_shape']['w']}x{payload['panel_shape']['h']}")
    print(f"GH_ASYM\t{payload['gh_asym_shape']['w']}x{payload['gh_asym_shape']['h']}")
    print(f"DEVICE\t{payload['device']}")
    print("")
    order = ["slot2", "slot4", "slot5", "slot6", "slot7", "palace", "fortress"]
    for key in order:
        row = payload["classifiers"].get(key)
        if not row:
            continue
        if row["type"] == "two_stage_cascade":
            s1 = row["stage1"]
            s2 = row.get("stage2")
            s2_text = "n/a"
            if s2:
                s2_text = f"{s2['predicted_label']} ({float(s2['confidence']):.4f})"
            print(
                f"{key}\tfinal={row['final_label']} ({float(row['final_confidence']):.4f})\t"
                f"stage1={s1['predicted_label']} ({float(s1['confidence']):.4f})\tstage2={s2_text}"
            )
        else:
            print(f"{key}\t{row['predicted_label']}\t{float(row['confidence']):.4f}")
    if payload["errors"]:
        print("")
        print("ERRORS")
        for key, err in payload["errors"].items():
            print(f"{key}\t{err.splitlines()[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run slot/Palace/Fortress classifiers in parallel on a panel crop (Great Hollow setup)."
    )
    parser.add_argument("--panel", required=True, help="Path to panel crop image.")
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "parallel_panel_classifiers_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device selection.",
    )
    args = parser.parse_args()

    panel_path = Path(args.panel).resolve()
    if not panel_path.exists():
        raise SystemExit(f"Panel image not found: {panel_path}")
    output_json = Path(args.output_json).resolve() if args.output_json else None

    payload = run_all(panel_path=panel_path, output_json=output_json, device_name=args.device)
    _print_summary(payload)
    if output_json is not None:
        print("")
        print(f"REPORT\t{output_json}")


if __name__ == "__main__":
    main()
