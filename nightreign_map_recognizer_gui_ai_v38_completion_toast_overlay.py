#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import math
import json
import os
import pickle
from collections import defaultdict
import queue
import re
import sys
import threading
import time
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Set default environment variables if not already set
if "GH_USE_PARALLEL_PANEL_CLASSIFIER" not in os.environ:
    os.environ["GH_USE_PARALLEL_PANEL_CLASSIFIER"] = "1"
if "GH_LAYOUT_MODEL_MATCH_MODE" not in os.environ:
    os.environ["GH_LAYOUT_MODEL_MATCH_MODE"] = "models"

IMAGE_W = 1536.0
IMAGE_H = 1536.0
WORLD_MIN_X = -256.0
WORLD_MIN_Z = -256.0
WORLD_MAX_X = 1280.0
WORLD_MAX_Z = 1280.0
EVERGAOL_ICON_PATH = Path(resource_path("POI_icons")) / "Evergaol.png"
EVERGAOL_WORLD_POINTS: Tuple[Tuple[float, float], ...] = (
    (2.6, 793.03),
    (276.65, 309.65),
    (428.37, 693.18),
    (483.79, 978.09),
    (697.3, 101.45),
    (713.22, 722.15),
    (957.87, 530.38),
)
EVERGAOL_EXPECTED_BY_EARTH = {
    "Limveld": {0, 1, 2, 3, 4, 5, 6},
    "Mountaintop": {1, 4, 5, 6},
    "The Crater": {0, 1, 4, 6},
    "Rotted Woods": {0, 1, 2, 3, 5, 6},
    "Noklateo": {0, 2, 3, 4, 5, 6},
    "Great Hollow": set(),
}
EVERGAOL_TEMPLATE_CACHE: Optional[List[dict]] = None
GREAT_HOLLOW_STATUS_ICON_PATHS = {
    "Status_Poison": Path(resource_path("POI_icons")) / "Status_Poison.png",
    "Status_Bleed": Path(resource_path("POI_icons")) / "Status_Bleed.png",
    "Status_Sleep": Path(resource_path("POI_icons")) / "Status_Sleep.png",
    "Status_Lightning": Path(resource_path("POI_icons")) / "Status_Lightning.png",
    "Status_Rot": Path(resource_path("POI_icons")) / "Status_Rot.png",
    "Status_Holy": Path(resource_path("POI_icons")) / "Status_Holy.png",
    "Status_Fire": Path(resource_path("POI_icons")) / "Status_Fire.png",
    "Status_Magic": Path(resource_path("POI_icons")) / "Status_Magic.png",
}
GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE = {
    "Ruins": ("Status_Poison", "Status_Bleed"),
    "Great Church": ("Status_Sleep", "Status_Lightning"),
    "Fortress": ("Status_Rot", "Status_Holy"),
    "Palace": ("Status_Fire", "Status_Magic"),
}
GREAT_HOLLOW_STATUS_PATTERN_DIRS = (
    Path(resource_path("renders_new")) / "no_text",
    Path(resource_path("renders_new")) / "with_text",
)
GREAT_HOLLOW_FIXED_STATUS_REFS = {
    "ref1": (
        Path(resource_path("layout_1007.png")),
        Path(resource_path("renders")) / "recognizer_no_text" / "layout_1007.png",
    ),
    "ref2": (
        Path(resource_path("layout_1005.png")),
        Path(resource_path("renders")) / "recognizer_no_text" / "layout_1005.png",
    ),
}
GREAT_HOLLOW_FIXED_STATUS_CONFIG = {
    "Palace": {
        "xy_norm": (0.328125, 0.28046875),
        "allowed": ("Status_Fire", "Status_Magic"),
        "ref1_label": "Status_Fire",
        "ref2_label": "Status_Magic",
    },
    "Fortress": {
        "xy_norm": (0.68359375, 0.7015625),
        "allowed": ("Status_Rot", "Status_Holy"),
        "ref1_label": "Status_Rot",
        "ref2_label": "Status_Holy",
    },
}
GREAT_HOLLOW_PATTERNSET_DIR = Path(resource_path("great_hollow_pattern_sets"))
GREAT_HOLLOW_PATTERNSET_STATUS_BY_FIELD = {
    "greatHollowEastFortMainModifier1": ("Fortress", "Status_Rot"),
    "greatHollowEastFortMainModifier2": ("Fortress", "Status_Holy"),
    "greatHollowWestFortMainModifier1": ("Palace", "Status_Fire"),
    "greatHollowWestFortMainModifier2": ("Palace", "Status_Magic"),
    # Great Hollow-specific status anchors beyond Palace/Fortress.
    # Mapping convention follows modifier index 1->first allowed status, 2->second allowed status.
    "greatHollowWestFortUnderModifier1": ("Ruins", "Status_Poison"),
    "greatHollowWestFortUnderModifier2": ("Ruins", "Status_Bleed"),
    "greatHollowWestFortLibraryModifier1": ("Great Church", "Status_Sleep"),
    "greatHollowWestFortLibraryModifier2": ("Great Church", "Status_Lightning"),
}
GREAT_HOLLOW_SYNTHETIC_WORLD_POINTS = {
    "Fortress": (734.399, 267.268),
    "Palace": (278.752, 805.636),
    "Ruins": (138.250, 392.670),
    "Great Church": (933.460, 548.590),
}
GREAT_HOLLOW_CROP_BORDER = 128.0
GREAT_HOLLOW_STATUS_INPUT_SIZE = 1536
GREAT_HOLLOW_STATUS_CROP_LEFT_PX = 175
GREAT_HOLLOW_STATUS_CROP_RIGHT_PX = 170
GREAT_HOLLOW_STATUS_CROP_TOP_PX = 180
GREAT_HOLLOW_STATUS_CROP_BOTTOM_PX = 180
GREAT_HOLLOW_STATUS_ICON_SOURCES = None
GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE = None
GREAT_HOLLOW_LAYOUT_STATUS_CACHE = {}
GREAT_HOLLOW_STATUS_ANCHOR_CACHE = None
GREAT_HOLLOW_STATUS_CLASSIFIER_MODE_DEFAULT = "models"  # classic|models
GREAT_HOLLOW_STATUS_MODEL_PATHS = {
    "Palace": (
        Path(resource_path("tmp_palace_magic_fire_test")) / "palace_magic_fire_test_model.pt",
    ),
    "Fortress": (
        Path(resource_path("tmp_fortress_rot_holy_test")) / "fortress_rot_holy_test_model.pt",
    ),
    "Ruins": (
        Path(resource_path("tmp_ruins_poison_bleed_test")) / "ruins_poison_bleed_test_model.pt",
    ),
    "Great Church": (
        Path(resource_path("tmp_gc_sleep_lightning_test")) / "gc_sleep_lightning_test_model.pt",
    ),
}
GREAT_HOLLOW_STATUS_MODEL_CACHE: dict = {}
ICON_RECOGNIZER_V4_MODULE = None
ICON_RECOGNIZER_V4_LOAD_ATTEMPTED = False
GREAT_HOLLOW_UNDER_PATTERN_DIRS = (
    Path(resource_path("renders_new_under")) / "no_text",
    Path(resource_path("renders_new_under")) / "with_text",
)
GREAT_HOLLOW_UNDER_IMAGE_DIRS = (
    Path(resource_path("renders")) / "recognizer_no_text_under",
    Path(resource_path("renders")) / "recognizer_no_text",
)
GREAT_HOLLOW_UNDER_MODEL_PATHS = (
    Path(resource_path("tmp_gc_frost_holy_test_newbox")) / "gc_frost_holy_test_model_newbox.pt",
)
GREAT_HOLLOW_UNDER_HYBRID_ALPHA_MODEL = 0.30
GREAT_HOLLOW_UNDER_HYBRID_SHARP_SCALE = 3.8
GREAT_HOLLOW_UNDER_HYBRID_BUNDLE = None
GREAT_HOLLOW_UNDER_HYBRID_LOAD_ATTEMPTED = False
GREAT_HOLLOW_UNDER_EXPECTED_STATUS_CACHE: dict = {}
GH_SLOT4_CLASSIFIER_MODE_DEFAULT = "prototype"  # classic|prototype
GH_SLOT4_TARGET_SLOT_ID = 4
GH_SLOT4_EMPTY_GATE_THRESHOLD_DEFAULT = 0.55
GH_SLOT4_PROTOTYPE_MODEL_PATHS = (
    Path(resource_path("tmp_gh_slot4_prototype_model_test")) / "gh_slot4_empty_vs_churchrise_prototype_model.pkl",
)
GH_SLOT4_PROTOTYPE_CACHE = None
GREAT_HOLLOW_LAYOUT_MODEL_MATCH_MODE_DEFAULT = "models"  # classic|models
GREAT_HOLLOW_LAYOUT_MODEL_ASYM_INPUT_SIZE = 1536
GREAT_HOLLOW_LAYOUT_MODEL_CROP_LEFT_PX = 175
GREAT_HOLLOW_LAYOUT_MODEL_CROP_RIGHT_PX = 170
GREAT_HOLLOW_LAYOUT_MODEL_CROP_TOP_PX = 180
GREAT_HOLLOW_LAYOUT_MODEL_CROP_BOTTOM_PX = 180
GREAT_HOLLOW_LAYOUT_MODEL_SPECS = {
    "slot2_great_church_status": {
        "kind": "status",
        "anchor_type": "Great Church",
        "source": "panel_raw",
        "crop_size": 200,
        "center_norm": (0.82890625, 0.38046875),  # panel crop normalized (top-left)
        "dx_px": 50,  # keep aligned with slot_2 offset50 training/infer path
        "dy_px": 0,
        "weight": 0.14,
        "module_paths": (
            Path(resource_path("Testing")) / "slot_2" / "layout_classifier_lib.py",
        ),
        "model_paths": (
            Path(resource_path("Testing")) / "slot_2_fixed_geometry_panel_bases260" / "slot2_status_cnn_offset50_v1.pt",
            Path(resource_path("Testing")) / "slot_2" / "slot2_status_cnn.pt",
        ),
    },
    "palace_status": {
        "kind": "status",
        "anchor_type": "Palace",
        "source": "gh_asym_legacy",
        "crop_size": 200,
        "center_norm": (329.0 / 1191.0, 251.0 / 1176.0),
        "weight": 0.12,
        "module_paths": (
            Path(resource_path("Testing")) / "palace" / "layout_classifier_lib.py",
        ),
        "model_paths": (
            Path(resource_path("Testing")) / "palace" / "palace_status_cnn.pt",
        ),
    },
    "fortress_status": {
        "kind": "status",
        "anchor_type": "Fortress",
        "source": "gh_asym_legacy",
        "crop_size": 200,
        "center_norm": (875.0 / 1191.0, 897.0 / 1176.0),
        "weight": 0.12,
        "module_paths": (
            Path(resource_path("Testing")) / "fortress" / "layout_classifier_lib.py",
        ),
        "model_paths": (
            Path(resource_path("Testing")) / "fortress" / "fortress_status_cnn.pt",
        ),
    },
    "slot6_ruins_status": {
        "kind": "status",
        "anchor_type": "Ruins",
        "source": "gh_asym_legacy",
        "crop_size": 200,
        "center_norm": (144.0 / 1191.0, 731.0 / 1176.0),
        "weight": 0.10,
        "module_paths": (
            Path(resource_path("Testing")) / "slot_6" / "layout_classifier_lib.py",
        ),
        "model_paths": (
            Path(resource_path("Testing")) / "slot_6" / "slot6_status_cnn.pt",
        ),
    },
    "slot4_minor": {
        "kind": "slot_label",
        "slot_id": 4,
        "source": "gh_asym_legacy",
        "crop_size": 200,
        "center_norm": (809.0 / 1191.0, 625.0 / 1176.0),
        "weight": 0.08,
        "module_paths": (
            Path(resource_path("Testing")) / "slot_4" / "layout_classifier_lib.py",
        ),
        "model_paths": (
            Path(resource_path("Testing")) / "slot_4_plus_real_heavy" / "slot4_status_cnn_plus_real_heavy.pt",
            Path(resource_path("Testing")) / "slot_4" / "slot4_status_cnn.pt",
        ),
    },
    "slot5_minor": {
        "kind": "slot_label",
        "slot_id": 5,
        "source": "gh_asym_legacy",
        "crop_size": 200,
        "center_norm": (324.0 / 1191.0, 629.0 / 1176.0),
        "weight": 0.08,
        "module_paths": (
            Path(resource_path("Testing")) / "slot_5" / "layout_classifier_lib.py",
        ),
        "model_paths": (
            Path(resource_path("Testing")) / "slot_5_plus_real_heavy" / "slot5_status_cnn_plus_real_heavy.pt",
            Path(resource_path("Testing")) / "slot_5" / "slot5_status_cnn.pt",
        ),
    },
    "slot7_minor": {
        "kind": "slot_label",
        "slot_id": 7,
        "source": "gh_asym_legacy",
        "crop_size": 200,
        "center_norm": (339.0 / 1191.0, 1085.0 / 1176.0),
        "weight": 0.08,
        "module_paths": (
            Path(resource_path("Testing")) / "slot_7" / "layout_classifier_lib.py",
        ),
        "model_paths": (
            Path(resource_path("Testing")) / "slot_7_plus_real_heavy" / "slot7_status_cnn_plus_real_heavy.pt",
            Path(resource_path("Testing")) / "slot_7" / "slot7_status_cnn.pt",
        ),
    },
}
GREAT_HOLLOW_LAYOUT_MODEL_CACHE: dict = {}
GH_PARALLEL_PANEL_CLASSIFIER_MODULE = None
GH_PARALLEL_PANEL_CLASSIFIER_LOAD_ATTEMPTED = False
GH_USE_PARALLEL_PANEL_CLASSIFIER_DEFAULT = True
EVERGAOL_MODEL_FILENAMES = [
    "evergaol_ai_test_output_no_text/evergaol_patch_model_v1.pt",
    "evergaol_ai_test_output/evergaol_patch_model_v1.pt",
]
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk

import render_bosses_on_backgrounds as renderer

renderer.GREAT_HOLLOW_PATTERNSET_DIR = Path(resource_path("great_hollow_pattern_sets"))
renderer.icons_dir = Path(resource_path("POI_icons"))
renderer.backgrounds_dir = Path(resource_path("maps"))

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    ctypes = None  # type: ignore[assignment]
    wintypes = None  # type: ignore[assignment]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
CMP_SIZE = 512
CACHE_VERSION = 6
DEFAULT_HOUGH_REGION = "right_half"
DEFAULT_HOUGH_SCALE = 0.5
DEFAULT_PANEL_SCALE = 1.12
DEFAULT_SEARCH_RADIUS = 80
DEFAULT_SQUARE_RATIO = 0.69
DEFAULT_PAD = 0
TOP_RESULTS_TO_LOG = 20
THUMB_COUNT = 5

# Step 1 preprocessing contract (definition only, no behavior switch yet):
# - resize panel to 1536x1536
# - crop 256 px from each border
# - resulting canonical earth-selector view is 1024x1024
EARTH_SELECTOR_INPUT_SIZE = 1536
EARTH_SELECTOR_BORDER_CROP_PX = 256
EARTH_SELECTOR_CROPPED_SIZE = EARTH_SELECTOR_INPUT_SIZE - (2 * EARTH_SELECTOR_BORDER_CROP_PX)

CACHE_FILENAME = "nightreign_ref_cache_square_gui.npz"
DISPLAY_SUBDIRS = [
    Path("renders/recognizer_with_text"),
    Path("renders/recognizer_no_text"),
    Path("renders/recognizer"),
    Path("renders/with_text"),
    Path("all_images/with_text"),
]
LEGACY_CACHE_SUBDIR = Path("all_images/no_text")
LOG_FILE_NAME = "recognition.log"
SETTINGS_FILE_NAME = "nightreign_settings.json"
DEFAULT_GLOBAL_HOTKEY = "Ctrl + Alt + M"
LARGE_EXE_ICON_FILE = "nightreign_sword_large_transparent_256.ico"
SMALL_WINDOW_ICON_FILE = "nightreign_sword_small_transparent_48.ico"
WINDOWS_APP_USER_MODEL_ID = "nightreign.recognizer.slots"

LOG_PANEL_EXPANDED_WIDTH = 390
LOG_PANEL_TOGGLE_WIDTH = 26
LOG_PANEL_ANIMATION_DURATION_MS = 180
LOG_PANEL_ANIMATION_FRAME_MS = 16


@dataclass
class Circle:
    cx: float
    cy: float
    r: float


@dataclass
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class RankedResult:
    number: int
    filename: str
    score: float
    details: dict


class CancelledError(RuntimeError):
    pass


class RunContext:
    def __init__(self, run_id: int, cancel_event: threading.Event, log_func: Callable[[int, str], None]):
        self.run_id = run_id
        self.cancel_event = cancel_event
        self._log_func = log_func

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("Recognition run was cancelled.")

    def log(self, message: str) -> None:
        self._log_func(self.run_id, message)


def timestamp_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def runtime_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        parent = exe_path.parent
        if parent.name.lower() == "_internal" and parent.parent.exists():
            return parent.parent
        return parent
    return Path(__file__).resolve().parent


def resource_base_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        return base
    return Path(__file__).resolve().parent


def log_file_path() -> Path:
    return runtime_dir() / LOG_FILE_NAME


def display_search_paths() -> List[Path]:
    rb = resource_base_dir()
    rd = runtime_dir()
    candidates: List[Path] = []
    seen = set()
    for base in (rd, rb):
        for subdir in DISPLAY_SUBDIRS:
            path = base / subdir
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)
    return candidates


def choose_existing_display_dir() -> Optional[Path]:
    for path in display_search_paths():
        if path.exists():
            return path
    return None


def settings_file_path() -> Path:
    return runtime_dir() / SETTINGS_FILE_NAME

def resource_path(name: str) -> Path:
    rb = resource_base_dir()
    candidate = rb / name
    if candidate.exists():
        return candidate
    return runtime_dir() / name

_PROTO_BACKEND_MODULE = None

def load_proto_backend_module(ctx: Optional[RunContext] = None):
    global _PROTO_BACKEND_MODULE
    if _PROTO_BACKEND_MODULE is not None:
        return _PROTO_BACKEND_MODULE
    backend_name = "recognize_nightreign_map_proto_v12_authoritative.py"
    candidates = [runtime_dir() / backend_name, resource_base_dir() / backend_name]
    backend_path = None
    for cand in candidates:
        if cand.exists():
            backend_path = cand
            break
    if backend_path is None:
        searched = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Authoritative proto backend not found. Looked in:\n{searched}")
    try:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("nightreign_proto_backend_authoritative", backend_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for {backend_path}")
        module = importlib.util.module_from_spec(spec)
        # dataclasses inside the backend expect the module to be registered in sys.modules
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _PROTO_BACKEND_MODULE = module
        if ctx is not None:
            ctx.log(f"Loaded authoritative proto backend: {backend_path.resolve()}")
        return module
    except Exception as exc:
        raise RuntimeError(f"Failed to load authoritative proto backend from {backend_path}: {exc}") from exc


def find_slot_package_path(ctx: Optional[RunContext] = None) -> Path:
    candidates = [
        runtime_dir() / SLOT_PACKAGE_FILENAME,
        resource_base_dir() / SLOT_PACKAGE_FILENAME,
        runtime_dir() / "nightreign_geom_proto_v4_slots.npz",
        resource_base_dir() / "nightreign_geom_proto_v4_slots.npz",
    ]
    seen = set()
    unique = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    for cand in unique:
        if cand.exists():
            if ctx is not None:
                ctx.log(f"Using slot package path: {cand.resolve()}")
            return cand
    searched = "\n".join(str(p) for p in unique)
    raise FileNotFoundError(f"Slot package not found. Looked in:\n{searched}")


def set_windows_app_user_model_id() -> None:
    if sys.platform != "win32" or ctypes is None:
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_USER_MODEL_ID)
    except Exception:
        pass


def apply_window_icons(root: tk.Tk) -> None:
    if sys.platform != "win32":
        return
    icon_candidates = [resource_path(SMALL_WINDOW_ICON_FILE), resource_path(LARGE_EXE_ICON_FILE)]
    for icon_path in icon_candidates:
        if not icon_path.exists():
            continue
        try:
            root.iconbitmap(default=str(icon_path))
            return
        except Exception:
            try:
                img = Image.open(icon_path)
                photo = ImageTk.PhotoImage(img)
                root._app_icon_photo = photo  # type: ignore[attr-defined]
                root.iconphoto(True, photo)
                return
            except Exception:
                continue


def load_settings() -> dict:
    path = settings_file_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    path = settings_file_path()
    ensure_parent_dir(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


HOTKEY_MODIFIER_ORDER = ("Ctrl", "Alt", "Shift", "Win")
HOTKEY_MODIFIER_SET = set(HOTKEY_MODIFIER_ORDER)
HOTKEY_NAME_ALIASES = {
    "CONTROL": "Ctrl",
    "CTRL": "Ctrl",
    "ALT": "Alt",
    "SHIFT": "Shift",
    "WIN": "Win",
    "WINDOWS": "Win",
    "SUPER": "Win",
    "RETURN": "Enter",
    "ENTER": "Enter",
    "ESCAPE": "Esc",
    "ESC": "Esc",
    "SPACE": "Space",
    "PRIOR": "PageUp",
    "NEXT": "PageDown",
    "PGUP": "PageUp",
    "PGDN": "PageDown",
    "PAGEUP": "PageUp",
    "PAGEDOWN": "PageDown",
    "LEFT": "Left",
    "RIGHT": "Right",
    "UP": "Up",
    "DOWN": "Down",
    "TAB": "Tab",
    "BACKSPACE": "Backspace",
    "DELETE": "Delete",
    "DEL": "Delete",
    "INSERT": "Insert",
    "HOME": "Home",
    "END": "End",
}
HOTKEY_EVENT_KEYSYM_ALIASES = {
    "Control_L": "Ctrl",
    "Control_R": "Ctrl",
    "Alt_L": "Alt",
    "Alt_R": "Alt",
    "Shift_L": "Shift",
    "Shift_R": "Shift",
    "Super_L": "Win",
    "Super_R": "Win",
    "Win_L": "Win",
    "Win_R": "Win",
    "Return": "Enter",
    "Escape": "Esc",
    "space": "Space",
    "Prior": "PageUp",
    "Next": "PageDown",
    "Left": "Left",
    "Right": "Right",
    "Up": "Up",
    "Down": "Down",
    "Tab": "Tab",
    "BackSpace": "Backspace",
    "Delete": "Delete",
    "Insert": "Insert",
    "Home": "Home",
    "End": "End",
}
HOTKEY_VK_CODES = {
    "Backspace": 0x08,
    "Tab": 0x09,
    "Enter": 0x0D,
    "Esc": 0x1B,
    "Space": 0x20,
    "PageUp": 0x21,
    "PageDown": 0x22,
    "End": 0x23,
    "Home": 0x24,
    "Left": 0x25,
    "Up": 0x26,
    "Right": 0x27,
    "Down": 0x28,
    "Insert": 0x2D,
    "Delete": 0x2E,
}
for _ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    HOTKEY_VK_CODES[_ch] = ord(_ch)
for _digit in "0123456789":
    HOTKEY_VK_CODES[_digit] = ord(_digit)
for _i in range(1, 25):
    HOTKEY_VK_CODES[f"F{_i}"] = 0x6F + _i


def canonicalize_hotkey_part(part: str) -> Optional[str]:
    cleaned = str(part).strip()
    if not cleaned:
        return None
    mapped = HOTKEY_NAME_ALIASES.get(cleaned.replace(" ", "").upper())
    if mapped is not None:
        return mapped
    if len(cleaned) == 1 and cleaned.isalpha():
        return cleaned.upper()
    if len(cleaned) == 1 and cleaned.isdigit():
        return cleaned
    upper = cleaned.upper()
    if re.fullmatch(r"F\d{1,2}", upper):
        return upper
    if cleaned in HOTKEY_VK_CODES:
        return cleaned
    titled = cleaned[:1].upper() + cleaned[1:]
    if titled in HOTKEY_VK_CODES:
        return titled
    return None


def hotkey_from_event_keysym(keysym: str) -> Optional[str]:
    if not keysym:
        return None
    if keysym in HOTKEY_EVENT_KEYSYM_ALIASES:
        return HOTKEY_EVENT_KEYSYM_ALIASES[keysym]
    return canonicalize_hotkey_part(keysym)


def format_hotkey_combo(modifiers: Sequence[str], key: str) -> str:
    ordered_mods = [m for m in HOTKEY_MODIFIER_ORDER if m in set(modifiers)]
    return " + ".join(ordered_mods + [key])


def normalize_hotkey_string(hotkey: str) -> Optional[str]:
    parts = [canonicalize_hotkey_part(part) for part in str(hotkey).split("+")]
    parts = [part for part in parts if part]
    if not parts:
        return None
    modifiers = {part for part in parts if part in HOTKEY_MODIFIER_SET}
    keys = [part for part in parts if part not in HOTKEY_MODIFIER_SET]
    if len(keys) != 1 or not modifiers:
        return None
    return format_hotkey_combo(modifiers, keys[0])


def hotkey_string_to_win_parts(hotkey: str) -> Optional[Tuple[int, int]]:
    normalized = normalize_hotkey_string(hotkey)
    if normalized is None:
        return None
    parts = normalized.split(" + ")
    modifiers = parts[:-1]
    key = parts[-1]
    mods = 0
    if "Alt" in modifiers:
        mods |= 0x0001
    if "Ctrl" in modifiers:
        mods |= 0x0002
    if "Shift" in modifiers:
        mods |= 0x0004
    if "Win" in modifiers:
        mods |= 0x0008
    vk = HOTKEY_VK_CODES.get(key)
    if vk is None:
        return None
    return mods, vk


class GlobalHotkeyManager:
    HOTKEY_ID = 0xB123
    WM_HOTKEY = 0x0312
    PM_REMOVE = 0x0001

    def __init__(self, trigger_callback: Callable[[], None]):
        self.trigger_callback = trigger_callback
        self._command_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker_main, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=2.0)

    def configure(self, hotkey: str) -> Tuple[bool, str]:
        if sys.platform != "win32":
            return False, "Global hotkeys are only implemented on Windows in this build."
        normalized = normalize_hotkey_string(hotkey)
        if normalized is None:
            return False, "Use at least one modifier and one main key."
        parts = hotkey_string_to_win_parts(normalized)
        if parts is None:
            return False, f"Unsupported hotkey: {normalized}"
        self.start()
        reply_queue: queue.Queue = queue.Queue()
        self._command_queue.put(("set", normalized, parts, reply_queue))
        try:
            return reply_queue.get(timeout=2.0)
        except queue.Empty:
            return False, "Timed out while applying the hotkey."

    def stop(self) -> None:
        if self._thread is None:
            return
        self._command_queue.put(("stop",))
        self._thread.join(timeout=2.0)
        self._thread = None

    def _worker_main(self) -> None:
        if sys.platform != "win32":
            self._ready_event.set()
            return
        user32 = ctypes.windll.user32
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        current_registered = False
        self._ready_event.set()
        try:
            while True:
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, self.PM_REMOVE):
                    if msg.message == self.WM_HOTKEY and msg.wParam == self.HOTKEY_ID:
                        try:
                            self.trigger_callback()
                        except Exception:
                            pass
                try:
                    command = self._command_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                kind = command[0]
                if kind == "stop":
                    break
                if kind == "set":
                    _, normalized, parts, reply_queue = command
                    mods, vk = parts
                    if current_registered:
                        user32.UnregisterHotKey(None, self.HOTKEY_ID)
                        current_registered = False
                    if not user32.RegisterHotKey(None, self.HOTKEY_ID, mods, vk):
                        err = ctypes.GetLastError()
                        try:
                            message = ctypes.FormatError(err).strip() or f"Windows error {err}"
                        except Exception:
                            message = f"Windows error {err}"
                        reply_queue.put((False, f"Could not register {normalized}: {message}"))
                    else:
                        current_registered = True
                        reply_queue.put((True, f"Hotkey set to {normalized}"))
        finally:
            if current_registered:
                user32.UnregisterHotKey(None, self.HOTKEY_ID)


def cache_search_paths() -> List[Path]:
    rb = resource_base_dir()
    rd = runtime_dir()
    candidates = [
        rd / CACHE_FILENAME,
        rb / CACHE_FILENAME,
        rb / LEGACY_CACHE_SUBDIR / CACHE_FILENAME,
        rd / LEGACY_CACHE_SUBDIR / CACHE_FILENAME,
    ]
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def choose_existing_cache_path() -> Optional[Path]:
    for path in cache_search_paths():
        if path.exists():
            return path
    return None


def load_cache_only(square_ratio: float, ctx: RunContext) -> Tuple[dict, Path]:
    checked: List[str] = []
    for path in cache_search_paths():
        checked.append(str(path.resolve() if path.exists() else path))
        if not path.exists():
            continue
        ctx.check_cancel()
        ctx.log(f"Trying cache file: {path.resolve()}")
        cache = load_reference_cache(path)
        meta = cache.get("metadata", {})
        if meta.get("cache_version") != CACHE_VERSION:
            raise RuntimeError(
                f"Cache file has version {meta.get('cache_version')}, but this app expects version {CACHE_VERSION}."
            )
        if int(meta.get("cmp_size", -1)) != CMP_SIZE:
            raise RuntimeError(
                f"Cache file has cmp_size {meta.get('cmp_size')}, but this app expects {CMP_SIZE}."
            )
        if abs(float(meta.get("square_ratio", -1.0)) - float(square_ratio)) >= 1e-12:
            raise RuntimeError(
                f"Cache file has square_ratio {meta.get('square_ratio')}, but this app expects {square_ratio}."
            )
        if "hashes" not in cache:
            raise RuntimeError("Cache file is missing the 'hashes' array required by this app.")
        return cache, path

    checked_block = "\n- ".join(checked) if checked else "(no cache paths configured)"
    raise FileNotFoundError(
        "Cache file not found. Checked these locations:\n- "
        + checked_block
        + "\n\nGenerate the GUI cache first with the image-based version, then either leave it in all_images/no_text or copy it next to this script or bundled app."
    )


def step_start(ctx: RunContext, name: str) -> float:
    ctx.check_cancel()
    ctx.log(f"[START] {name}")
    return time.perf_counter()


def step_end(ctx: RunContext, name: str, start_time: float) -> None:
    ctx.check_cancel()
    ctx.log(f"[DONE] {name} ({fmt_seconds(time.perf_counter() - start_time)})")


def ensure_parent_dir(path: Optional[Path]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def save_bgr(path: Optional[Path], img_bgr: np.ndarray) -> None:
    if path is None:
        return
    ensure_parent_dir(path)
    cv2.imwrite(str(path), img_bgr)


def get_available_monitors() -> List[Tuple[int, dict]]:
    with mss.mss() as sct:
        return [(idx, dict(monitor)) for idx, monitor in enumerate(sct.monitors[1:], start=1)]


def monitor_label(monitor_index: int, monitor: dict) -> str:
    width = int(monitor.get("width", 0))
    height = int(monitor.get("height", 0))
    left = int(monitor.get("left", 0))
    top = int(monitor.get("top", 0))
    return f"Display {monitor_index} — {width}x{height} @ ({left},{top})"


def capture_monitor_bgr(monitor_index: int) -> np.ndarray:
    with mss.mss() as sct:
        if monitor_index < 1 or monitor_index >= len(sct.monitors):
            available = max(0, len(sct.monitors) - 1)
            raise ValueError(f"Monitor index {monitor_index} is not available. Detected {available} screen(s).")
        monitor = sct.monitors[monitor_index]
        shot = np.array(sct.grab(monitor), dtype=np.uint8)
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def capture_primary_monitor_bgr() -> np.ndarray:
    return capture_monitor_bgr(1)


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def list_reference_images(refs_dir: Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for path in refs_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.search(r"(\d+)$", path.stem)
        if not m:
            continue
        files.append((int(m.group(1)), path))
    if not files:
        raise FileNotFoundError(f"No suitable reference images found in {refs_dir}.")
    files.sort(key=lambda x: (x[0], x[1].name.lower()))
    return files


def find_image_by_number(images_dir: Path, number: int) -> Path:
    matches: List[Path] = []
    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.search(r"(\d+)$", path.stem)
        if not m:
            continue
        if int(m.group(1)) == int(number):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No image ending with number {number} was found in {images_dir}.")
    matches.sort(key=lambda p: p.name.lower())
    return matches[0]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def reference_signature(refs: Sequence[Tuple[int, Path]]) -> List[dict]:
    sig = []
    for number, path in refs:
        stat = path.stat()
        sig.append(
            {
                "number": int(number),
                "name": path.name,
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
                "sha256": file_sha256(path),
            }
        )
    return sig


def detect_map_circle(img_bgr: np.ndarray, region_name: str, scale: float, ctx: RunContext) -> Tuple[Circle, str]:
    h, w = img_bgr.shape[:2]
    fallback = Circle(w * 0.76, h * 0.50, min(h, w) * 0.32)

    if region_name == "right_half":
        x0, y0, x1, y1 = w // 2, 0, w, h
    elif region_name == "right_40":
        x0, y0, x1, y1 = int(round(w * 0.60)), 0, w, h
    elif region_name == "right_third":
        x0, y0, x1, y1 = (2 * w) // 3, 0, w, h
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    roi = img_bgr[y0:y1, x0:x1]

    ctx.log(
        f"Hough search region '{region_name}': x0={x0}, y0={y0}, x1={x1}, y1={y1}, width={x1-x0}, height={y1-y0}"
    )
    ctx.check_cancel()

    if scale < 1.0:
        small_w = max(1, int(round((x1 - x0) * scale)))
        small_h = max(1, int(round((y1 - y0) * scale)))
        roi_small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        roi_small = roi
        small_h, small_w = roi_small.shape[:2]
    ctx.log(f"Hough scale={scale:.3f}; ROI resized to {small_w}x{small_h}")

    min_radius = int(min(small_h, small_w) * 0.18)
    max_radius = int(min(small_h, small_w) * 0.42)

    # First try to detect the always-blue outer circle directly.
    hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([88, 20, 70], dtype=np.uint8), np.array([132, 255, 255], dtype=np.uint8))
    blue_mask = cv2.GaussianBlur(blue_mask, (9, 9), 2.0)
    blue_circles = cv2.HoughCircles(
        blue_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, min(small_h, small_w) // 3),
        param1=80,
        param2=22,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if blue_circles is not None and len(blue_circles[0]) > 0:
        ctx.log(f"Blue-ring Hough found {len(blue_circles[0])} candidates.")
        best_blue = None
        best_blue_score = None
        for idx, c in enumerate(blue_circles[0], start=1):
            ctx.check_cancel()
            cx, cy, r = map(float, c)
            full_cx = x0 + (cx / scale if scale < 1.0 else cx)
            full_cy = y0 + (cy / scale if scale < 1.0 else cy)
            full_r = (r / scale) if scale < 1.0 else r
            # Prefer circles close to the known panel anchor, but strongly favor larger circles.
            score = (
                abs(full_cx - fallback.cx) / w
                + abs(full_cy - fallback.cy) / h
                - 0.40 * (full_r / min(h, w))
            )
            if idx <= 8:
                ctx.log(
                    f"Blue candidate {idx}: full_cx={full_cx:.2f}, full_cy={full_cy:.2f}, full_r={full_r:.2f}, score={score:.6f}"
                )
            if best_blue_score is None or score < best_blue_score:
                best_blue_score = score
                best_blue = Circle(full_cx, full_cy, full_r)

        if best_blue is not None:
            center_dx = best_blue.cx - fallback.cx
            center_dy = best_blue.cy - fallback.cy
            radius_ratio = best_blue.r / fallback.r if fallback.r > 1e-6 else 1.0
            if abs(center_dx) <= w * 0.03 and abs(center_dy) <= h * 0.08 and 0.92 <= radius_ratio <= 1.10:
                ctx.log(
                    f"Using blue-ring circle: cx={best_blue.cx:.2f}, cy={best_blue.cy:.2f}, r={best_blue.r:.2f}"
                )
                return best_blue, "blue_ring"
            ctx.log(
                f"Best blue-ring circle drifted too far from the stable anchor "
                f"(dx={center_dx:.2f}, dy={center_dy:.2f}, radius_ratio={radius_ratio:.3f}). Ignoring it."
            )
    else:
        ctx.log("Blue-ring Hough found 0 candidates.")

    gray = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

    hough_params = {
        "dp": 1.2,
        "minDist": max(1, min(small_h, small_w) // 3),
        "param1": 120,
        "param2": 28,
        "minRadius": min_radius,
        "maxRadius": max_radius,
    }
    ctx.log(
        "Grayscale Hough params: "
        f"dp={hough_params['dp']}, minDist={hough_params['minDist']}, "
        f"param1={hough_params['param1']}, param2={hough_params['param2']}, "
        f"minRadius={hough_params['minRadius']}, maxRadius={hough_params['maxRadius']}"
    )
    ctx.check_cancel()

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=hough_params["dp"],
        minDist=hough_params["minDist"],
        param1=hough_params["param1"],
        param2=hough_params["param2"],
        minRadius=hough_params["minRadius"],
        maxRadius=hough_params["maxRadius"],
    )

    if circles is None or len(circles[0]) == 0:
        ctx.log(
            f"Grayscale Hough found 0 candidates. Using stable fallback circle cx={fallback.cx:.2f}, cy={fallback.cy:.2f}, r={fallback.r:.2f}"
        )
        return fallback, "fallback"

    ctx.log(f"Grayscale Hough found {len(circles[0])} candidates.")
    best = None
    best_score = None
    preview_count = min(10, len(circles[0]))
    for idx, c in enumerate(circles[0], start=1):
        ctx.check_cancel()
        cx, cy, r = map(float, c)
        full_cx = x0 + (cx / scale if scale < 1.0 else cx)
        full_cy = y0 + (cy / scale if scale < 1.0 else cy)
        full_r = (r / scale) if scale < 1.0 else r
        size_bonus = 0.30 * (full_r / min(h, w))
        score = abs(full_cx - fallback.cx) / w + abs(full_cy - fallback.cy) / h - size_bonus
        if idx <= preview_count:
            ctx.log(
                f"Gray candidate {idx}: full_cx={full_cx:.2f}, full_cy={full_cy:.2f}, full_r={full_r:.2f}, score={score:.6f}"
            )
        if best_score is None or score < best_score:
            best_score = score
            best = Circle(full_cx, full_cy, full_r)

    chosen = best if best is not None else fallback

    center_dx = chosen.cx - fallback.cx
    center_dy = chosen.cy - fallback.cy
    radius_ratio = chosen.r / fallback.r if fallback.r > 1e-6 else 1.0
    max_dx = w * 0.020
    max_dy = h * 0.050

    if abs(center_dx) > max_dx or abs(center_dy) > max_dy or not (0.95 <= radius_ratio <= 1.08):
        ctx.log(
            f"Chosen grayscale Hough circle drifted too far from the stable map anchor "
            f"(dx={center_dx:.2f}, dy={center_dy:.2f}, radius_ratio={radius_ratio:.3f}). "
            f"Using stable fallback circle instead."
        )
        chosen = fallback
        method = "fallback"
    else:
        blended_cx = fallback.cx + center_dx * 0.25
        blended_cy = fallback.cy + center_dy * 0.25
        blended_r = fallback.r + (chosen.r - fallback.r) * 0.25
        chosen = Circle(blended_cx, blended_cy, blended_r)
        ctx.log(
            f"Using stabilized grayscale Hough circle: cx={chosen.cx:.2f}, cy={chosen.cy:.2f}, r={chosen.r:.2f}"
        )
        method = "gray_hough"

    ctx.log(
        f"Chosen circle: cx={chosen.cx:.2f}, cy={chosen.cy:.2f}, r={chosen.r:.2f}, score={best_score if best_score is not None else float('nan'):.6f}"
    )
    return chosen, method


def estimate_panel_box(circle: Circle, panel_scale: float) -> CropBox:
    side = 2.0 * circle.r * panel_scale
    half = side / 2.0
    return CropBox(
        int(round(circle.cx - half)),
        int(round(circle.cy - half)),
        int(round(circle.cx + half)),
        int(round(circle.cy + half)),
    )


def clip_box(box: CropBox, w: int, h: int) -> CropBox:
    return CropBox(
        max(0, min(w - 1, box.x0)),
        max(1, min(h - 1, box.y0)),
        max(1, min(w, box.x1)),
        max(1, min(h, box.y1)),
    )


def refine_vertical_edge(gray: np.ndarray, gradx: np.ndarray, x_est: int, y0: int, y1: int, search_radius: int) -> int:
    h, w = gray.shape
    x_start = max(0, x_est - search_radius)
    x_end = min(w - 1, x_est + search_radius)
    best_x = x_est
    best_score = None
    for x in range(x_start, x_end + 1):
        col = gray[y0:y1, max(0, x - 1):min(w, x + 2)]
        gx = np.abs(gradx[y0:y1, x])
        score = float((col > 150).sum()) + 3.0 * float(gx.mean())
        if best_score is None or score > best_score:
            best_score = score
            best_x = x
    return best_x


def refine_horizontal_edge(gray: np.ndarray, grady: np.ndarray, y_est: int, x0: int, x1: int, search_radius: int) -> int:
    h, w = gray.shape
    y_start = max(0, y_est - search_radius)
    y_end = min(h - 1, y_est + search_radius)
    best_y = y_est
    best_score = None
    for y in range(y_start, y_end + 1):
        row = gray[max(0, y - 1):min(h, y + 2), x0:x1]
        gy = np.abs(grady[y, x0:x1])
        score = float((row > 150).sum()) + 3.0 * float(gy.mean())
        if best_score is None or score > best_score:
            best_score = score
            best_y = y
    return best_y


def refine_panel_box(
    img_bgr: np.ndarray,
    circle: Circle,
    panel_scale: float,
    search_radius: int,
    pad: int,
    ctx: RunContext,
) -> CropBox:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    est = clip_box(estimate_panel_box(circle, panel_scale), w, h)
    band_y0 = max(0, int(round(circle.cy - circle.r * 1.18)))
    band_y1 = min(h, int(round(circle.cy + circle.r * 1.18)))
    band_x0 = max(0, int(round(circle.cx - circle.r * 1.18)))
    band_x1 = min(w, int(round(circle.cx + circle.r * 1.18)))

    left = refine_vertical_edge(gray, gradx, est.x0, band_y0, band_y1, search_radius)
    ctx.check_cancel()
    right = refine_vertical_edge(gray, gradx, est.x1, band_y0, band_y1, search_radius)
    ctx.check_cancel()
    top = refine_horizontal_edge(gray, grady, est.y0, band_x0, band_x1, search_radius)
    ctx.check_cancel()
    bottom = refine_horizontal_edge(gray, grady, est.y1, band_x0, band_x1, search_radius)
    ctx.check_cancel()

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    side = max(32, int(round(((right - left) + (bottom - top)) / 2.0)))
    half = side / 2.0

    box = clip_box(
        CropBox(
            int(round(center_x - half)) - pad,
            int(round(center_y - half)) - pad,
            int(round(center_x + half)) + pad,
            int(round(center_y + half)) + pad,
        ),
        w,
        h,
    )
    side2 = min(box.x1 - box.x0, box.y1 - box.y0)
    return clip_box(CropBox(box.x0, box.y0, box.x0 + side2, box.y0 + side2), w, h)


def center_square_crop(img_bgr: np.ndarray, ratio: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = max(1, int(round(min(h, w) * ratio)))
    x0 = max(0, (w - side) // 2)
    y0 = max(0, (h - side) // 2)
    return img_bgr[y0:y0 + side, x0:x0 + side]


def preprocess_for_similarity(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (CMP_SIZE, CMP_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), 6.0)
    return gray


def preprocess_for_gradients(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (CMP_SIZE, CMP_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), 2.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 2.0)
    return mag


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd <= 1e-6:
        sd = 1.0
    return (arr - mu) / sd


def corr(a: np.ndarray, b: np.ndarray) -> float:
    va = a.astype(np.float32).ravel()
    vb = b.astype(np.float32).ravel()
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / denom)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def phash(gray: np.ndarray, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    img = gray.astype(np.float32)
    size = hash_size * highfreq_factor
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(img)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])
    return (low > med).astype(np.uint8).flatten()


def hash_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1.size != h2.size:
        return 0.0
    return 1.0 - float(np.count_nonzero(h1 != h2)) / float(h1.size)


def compare_precomputed(
    shot_low: np.ndarray,
    shot_grad: np.ndarray,
    shot_hash: np.ndarray,
    ref_low: np.ndarray,
    ref_grad: np.ndarray,
    ref_hash: np.ndarray,
) -> Tuple[float, dict]:
    shot_low_n = normalize(shot_low)
    shot_grad_n = normalize(shot_grad)
    ref_low_n = normalize(ref_low)
    ref_grad_n = normalize(ref_grad)

    low_corr = corr(shot_low_n, ref_low_n)
    low_mae = mae(shot_low_n, ref_low_n)
    grad_corr = corr(shot_grad_n, ref_grad_n)
    grad_mae = mae(shot_grad_n, ref_grad_n)
    hsim = hash_similarity(shot_hash, ref_hash)

    score = (
        0.42 * low_corr
        + 0.26 * grad_corr
        + 0.22 * hsim
        + 0.06 * (1.0 / (1.0 + low_mae))
        + 0.04 * (1.0 / (1.0 + grad_mae))
    )
    return float(score), {
        "low_corr": low_corr,
        "grad_corr": grad_corr,
        "hash_similarity": hsim,
        "low_mae": low_mae,
        "grad_mae": grad_mae,
    }


def build_reference_cache(refs: Sequence[Tuple[int, Path]], cache_path: Path, square_ratio: float, ctx: RunContext) -> dict:
    ensure_parent_dir(cache_path)
    n = len(refs)
    numbers = np.empty((n,), dtype=np.int32)
    filenames: List[str] = []
    low = np.empty((n, CMP_SIZE, CMP_SIZE), dtype=np.uint8)
    grad = np.empty((n, CMP_SIZE, CMP_SIZE), dtype=np.float32)
    hashes = np.empty((n, 16 * 16), dtype=np.uint8)

    for i, (number, path) in enumerate(refs, start=1):
        ctx.check_cancel()
        img = load_bgr(path)
        focus = center_square_crop(img, square_ratio)
        low_img = preprocess_for_similarity(focus)
        grad_img = preprocess_for_gradients(focus)
        hash_img = phash(low_img)
        numbers[i - 1] = int(number)
        filenames.append(path.name)
        low[i - 1] = low_img
        grad[i - 1] = grad_img
        hashes[i - 1] = hash_img
        ctx.log(f"Building cache: {i}/{n} ({path.name})")

    ctx.check_cancel()
    signature = reference_signature(refs)
    meta = {
        "cache_version": CACHE_VERSION,
        "cmp_size": CMP_SIZE,
        "square_ratio": square_ratio,
        "signature": signature,
    }
    max_name_len = max((len(name) for name in filenames), default=1)
    np.savez_compressed(
        cache_path,
        numbers=numbers,
        filenames=np.array(filenames, dtype=f"<U{max_name_len}"),
        low=low,
        grad=grad,
        hashes=hashes,
        metadata_json=np.array(json.dumps(meta)),
    )
    return {
        "numbers": numbers,
        "filenames": np.array(filenames),
        "low": low,
        "grad": grad,
        "hashes": hashes,
        "metadata": meta,
    }


def load_reference_cache(cache_path: Path) -> dict:
    with np.load(cache_path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        return {
            "numbers": data["numbers"],
            "filenames": data["filenames"],
            "low": data["low"],
            "grad": data["grad"],
            "hashes": data["hashes"],
            "metadata": meta,
        }


def cache_matches_refs(cache: dict, refs: Sequence[Tuple[int, Path]], square_ratio: float) -> bool:
    meta = cache.get("metadata", {})
    return (
        meta.get("cache_version") == CACHE_VERSION
        and meta.get("cmp_size") == CMP_SIZE
        and abs(float(meta.get("square_ratio", -1.0)) - float(square_ratio)) < 1e-12
        and meta.get("signature") == reference_signature(refs)
    )


def get_or_build_reference_cache(refs: Sequence[Tuple[int, Path]], cache_path: Path, square_ratio: float, ctx: RunContext) -> dict:
    if cache_path.exists():
        try:
            ctx.log(f"Found reference cache: {cache_path}")
            cache = load_reference_cache(cache_path)
            ctx.check_cancel()
            if cache_matches_refs(cache, refs, square_ratio):
                ctx.log("Reference cache is valid. Reusing cached features.")
                return cache
            ctx.log("Reference cache is stale. Rebuilding cache...")
        except Exception as exc:
            ctx.log(f"Could not use existing cache ({exc}). Rebuilding cache...")
    else:
        ctx.log("No reference cache found. Building cache...")

    cache = build_reference_cache(refs, cache_path, square_ratio, ctx)
    ctx.log(f"Reference cache saved to {cache_path}")
    return cache


_DYNAMIC_RENDER_CACHE_ROOT = "tmp_dynamic_render_cache_v38"


def dynamic_render_cache_path(pattern_id: int, variant: str) -> Path:
    cache_dir = runtime_dir() / _DYNAMIC_RENDER_CACHE_ROOT / str(variant)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"layout_{int(pattern_id):04d}.png"


def save_dynamic_render_to_cache(
    img: Image.Image,
    pattern_id: int,
    variant: str,
    ctx: RunContext,
) -> Optional[Path]:
    out_path = dynamic_render_cache_path(pattern_id, variant)
    try:
        if not out_path.exists():
            img.save(out_path, format="PNG")
        return out_path
    except Exception as exc:
        ctx.log(f"Warning: could not cache dynamic render for #{pattern_id} ({variant}): {exc}")
        return None


def resolve_display_paths(
    results: Sequence[RankedResult],
    ctx: RunContext,
    disp_dir: Optional[Path],
    dynamic_render: bool = True,
    allow_render_fallback: bool = True,
) -> List[Path | Image.Image]:
    paths: List[Path | Image.Image] = []
    for result in results:
        if len(paths) >= THUMB_COUNT:
            break
        if dynamic_render:
            cached_path = dynamic_render_cache_path(result.number, "main")
            if cached_path.exists():
                paths.append(cached_path)
                continue
            try:
                entry = renderer.get_entry_by_pattern_id(result.number)
                img = renderer.build_pattern_image(entry)
                if img is not None:
                    cached_saved = save_dynamic_render_to_cache(img, result.number, "main", ctx)
                    paths.append(cached_saved if cached_saved is not None else img)
                    continue
                ctx.log(f"Warning: Dynamic render returned no image for #{result.number}.")
            except Exception as exc:
                ctx.log(f"Warning: dynamic render failed for image #{result.number}: {exc}")
        if allow_render_fallback and disp_dir is not None:
            try:
                paths.append(find_image_by_number(disp_dir, result.number))
            except FileNotFoundError:
                ctx.log(f"Warning: Could not find image #{result.number} in {disp_dir}.")
    return paths


def resolve_overlay_text_only_paths(
    results: Sequence[RankedResult],
    ctx: RunContext,
    dynamic_render: bool = True,
) -> List[Optional[Path | Image.Image]]:
    paths: List[Optional[Path | Image.Image]] = []
    for result in results:
        if len(paths) >= THUMB_COUNT:
            break
        overlay_img: Optional[Path | Image.Image] = None
        if dynamic_render:
            cached_path = dynamic_render_cache_path(result.number, "overlay_text")
            if cached_path.exists():
                paths.append(cached_path)
                continue
            try:
                entry = renderer.get_entry_by_pattern_id(result.number)
                overlay_img = renderer.build_pattern_text_overlay_image(entry)
                if overlay_img is None:
                    ctx.log(f"Warning: Overlay text-only render returned no image for #{result.number}.")
                else:
                    cached_saved = save_dynamic_render_to_cache(overlay_img, result.number, "overlay_text", ctx)
                    overlay_img = cached_saved if cached_saved is not None else overlay_img
            except Exception as exc:
                ctx.log(f"Warning: overlay text-only render failed for image #{result.number}: {exc}")
        paths.append(overlay_img)
    return paths


def run_recognition(
    ctx: RunContext,
    monitor_index: int = 1,
    preview_callback: Optional[Callable[[np.ndarray], None]] = None,
    pre_captured_full_img: Optional[np.ndarray] = None,
    second_full_img: Optional[np.ndarray] = None,
    dual_shot_enabled: bool = False,
    dynamic_render: bool = False,
) -> dict:
    total_start = time.perf_counter()

    disp_dir = choose_existing_display_dir()
    if disp_dir is None:
        checked = "\n".join(f"- {p}" for p in display_search_paths())
        raise FileNotFoundError(
            "Display directory not found. Checked these locations:\n" + checked
        )

    ctx.log(f"Display directory: {disp_dir.resolve()}")
    ctx.log(f"Log file: {log_file_path().resolve()}")

    selected_monitor = None
    for idx, monitor in get_available_monitors():
        if idx == monitor_index:
            selected_monitor = monitor
            break
    if selected_monitor is None:
        raise RuntimeError(f"Selected display {monitor_index} is no longer available.")
    ctx.log(f"Selected display: {monitor_label(monitor_index, selected_monitor)}")

    t = step_start(ctx, "Load reference cache")
    cache, used_cache_path = load_cache_only(DEFAULT_SQUARE_RATIO, ctx)
    step_end(ctx, "Load reference cache", t)
    ctx.log(f"Using cache file: {used_cache_path.resolve()}")

    t = step_start(ctx, "Capture screenshot")
    full_img = capture_monitor_bgr(monitor_index)
    step_end(ctx, "Capture screenshot", t)

    t = step_start(ctx, "Detect rough map circle")
    crop_circle, circle_method = detect_map_circle(full_img, DEFAULT_HOUGH_REGION, DEFAULT_HOUGH_SCALE, ctx)
    step_end(ctx, "Detect rough map circle", t)

    if circle_method == "blue_ring":
        t = step_start(ctx, "Estimate map panel from blue ring")
        crop_box = clip_box(estimate_panel_box(crop_circle, DEFAULT_PANEL_SCALE), full_img.shape[1], full_img.shape[0])
        ctx.log("Skipping edge refinement because the blue ring is a more stable anchor than panel edges.")
        step_end(ctx, "Estimate map panel from blue ring", t)
    else:
        t = step_start(ctx, "Refine map panel boundaries")
        crop_box = refine_panel_box(full_img, crop_circle, DEFAULT_PANEL_SCALE, DEFAULT_SEARCH_RADIUS, DEFAULT_PAD, ctx)
        step_end(ctx, "Refine map panel boundaries", t)

    t = step_start(ctx, "Extract cropped panel and focus square")
    cropped_panel = full_img[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
    focus_square = center_square_crop(cropped_panel, DEFAULT_SQUARE_RATIO)
    step_end(ctx, "Extract cropped panel and focus square", t)

    if preview_callback is not None:
        preview_callback(cropped_panel.copy())
        ctx.log("Displayed captured map preview.")

    t = step_start(ctx, "Preprocess screenshot focus square")
    shot_low = preprocess_for_similarity(focus_square)
    shot_grad = preprocess_for_gradients(focus_square)
    shot_hash = phash(shot_low)
    step_end(ctx, "Preprocess screenshot focus square", t)

    numbers = cache["numbers"]
    filenames = cache["filenames"]
    low_cache = cache["low"]
    grad_cache = cache["grad"]
    hash_cache = cache["hashes"]

    t = step_start(ctx, "Recognition")
    results: List[RankedResult] = []
    total = len(numbers)
    for idx in range(total):
        ctx.check_cancel()
        score, details = compare_precomputed(
            shot_low,
            shot_grad,
            shot_hash,
            low_cache[idx],
            grad_cache[idx],
            hash_cache[idx],
        )
        result = RankedResult(number=int(numbers[idx]), filename=str(filenames[idx]), score=score, details=details)
        results.append(result)
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == total:
            ctx.log(f"Recognition progress: {idx + 1}/{total}")
    step_end(ctx, "Recognition", t)

    t = step_start(ctx, "Sort results")
    results.sort(key=lambda r: r.score, reverse=True)
    ordered_numbers = [r.number for r in results]
    ctx.log("Ordered map numbers (best first):")
    ctx.log(json.dumps(ordered_numbers))
    ctx.log("")
    ctx.log(f"Top {min(TOP_RESULTS_TO_LOG, len(results))} matches:")
    for i, r in enumerate(results[:TOP_RESULTS_TO_LOG], start=1):
        ctx.log(
            f"{i:>2}. #{r.number:<3} score={r.score:.6f} "
            f"low_corr={r.details['low_corr']:.4f} grad_corr={r.details['grad_corr']:.4f} "
            f"hash={r.details['hash_similarity']:.4f} file={r.filename}"
        )
    step_end(ctx, "Sort results", t)

    t = step_start(ctx, "Resolve top image paths")
    top_paths = resolve_display_paths(results, ctx, disp_dir, dynamic_render=dynamic_render)
    step_end(ctx, "Resolve top image paths", t)

    ctx.log(f"Crop box: x0={crop_box.x0}, y0={crop_box.y0}, x1={crop_box.x1}, y1={crop_box.y1}")
    ctx.log(f"Total execution time: {fmt_seconds(time.perf_counter() - total_start)}")
    return {
        "results": results,
        "ordered_numbers": ordered_numbers,
        "crop_box": crop_box,
        "top_paths": top_paths,
        "focus_square": focus_square,
        "cropped_panel": cropped_panel,
    }


class NightreignRecognizerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Nightreign Map Recognizer - AI Log Tab v37")
        apply_window_icons(self.root)
        self.root.geometry("1720x940")
        self.root.minsize(1180, 700)
        self.root.configure(bg="#0a0d12")

        self.current_run_id = 0
        self.settings = load_settings()
        saved_hotkey = normalize_hotkey_string(str(self.settings.get("global_hotkey", DEFAULT_GLOBAL_HOTKEY)))
        self.global_hotkey_string = saved_hotkey or DEFAULT_GLOBAL_HOTKEY
        self.hotkey_manager = GlobalHotkeyManager(lambda: self.message_queue.put(("hotkey",)))
        self.hotkey_popover: Optional[tk.Toplevel] = None
        self.hotkey_popover_content: Optional[tk.Frame] = None
        self.hotkey_popover_bind_id: Optional[str] = None
        self.hotkey_listening = False
        self.hotkey_pressed_modifiers: set[str] = set()
        self.hotkey_capture_var = tk.StringVar(value=self.global_hotkey_string)
        self.hotkey_status_var = tk.StringVar(value="")
        self.current_cancel_event: Optional[threading.Event] = None
        self.active_worker: Optional[threading.Thread] = None
        self.message_queue: queue.Queue = queue.Queue()
        self.log_file_path = log_file_path()
        self.log_panel_open = bool(self.settings.get("log_panel_open", False))
        self.log_panel_target_width = LOG_PANEL_EXPANDED_WIDTH if self.log_panel_open else 0
        self.log_panel_current_width = self.log_panel_target_width
        self.log_panel_anim_after_id: Optional[str] = None
        self.log_panel_anim_start_width = self.log_panel_current_width
        self.log_panel_anim_start_time = 0.0
        self.log_panel_animating = False
        self.layout_refresh_after_id: Optional[str] = None
        self.log_toggle_text_var = tk.StringVar()

        self.monitor_entries = get_available_monitors()
        if not self.monitor_entries:
            raise RuntimeError("No connected displays were detected.")
        self.monitor_choice_var = tk.StringVar(value="Primary / default")

        self.current_results: List[RankedResult] = []
        self.current_image_paths: List[Path | Image.Image] = []
        self.current_selected_index = 0
        self.image_cache: dict[Path, Image.Image] = {}
        self.main_photo: Optional[ImageTk.PhotoImage] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.main_canvas_photo: Optional[ImageTk.PhotoImage] = None
        self.current_main_pil_image: Optional[Image.Image] = None
        self.main_zoom_factor = 1.0
        self.main_pan_x_norm = 0.0
        self.main_pan_y_norm = 0.0
        self.main_drag_last_xy: Optional[Tuple[int, int]] = None
        self.main_placeholder_text = "Run recognition to display results"
        self.thumb_photos: List[Optional[ImageTk.PhotoImage]] = [None] * THUMB_COUNT

        self.status_var = tk.StringVar(value="Idle")
        self.best_var = tk.StringVar(value="Best: —")
        self.selected_var = tk.StringVar(value="Selected: —")
        self.score_var = tk.StringVar(value="Score: —")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._process_queue)
        self.root.after(150, self._initialize_hotkey_registration)
        self.root.bind("<Configure>", self._on_root_configure)

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Run.TButton",
            font=("Segoe UI", 13, "bold"),
            padding=(12, 8),
            foreground="#f5f7fb",
            background="#1e66ff",
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "Run.TButton",
            background=[("active", "#2b74ff"), ("pressed", "#1757de")],
            foreground=[("disabled", "#9098a3")],
        )

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0, minsize=self.log_panel_current_width)
        self.root.grid_columnconfigure(1, weight=1)

        self.log_tab_container = tk.Frame(
            self.root,
            bg="#0a0d12",
            width=self.log_panel_current_width,
            highlightthickness=0,
            bd=0,
        )
        self.log_tab_container.grid(row=0, column=0, sticky="nsew")
        self.log_tab_container.grid_propagate(False)
        self.log_tab_container.grid_rowconfigure(0, weight=1)
        self.log_tab_container.grid_columnconfigure(0, weight=1)

        self.logs_frame = tk.Frame(
            self.log_tab_container,
            bg="#0d1117",
            width=LOG_PANEL_EXPANDED_WIDTH,
            highlightthickness=0,
            bd=0,
        )
        self.logs_frame.place(x=self.log_panel_current_width - LOG_PANEL_EXPANDED_WIDTH, y=0, width=LOG_PANEL_EXPANDED_WIDTH, relheight=1.0)
        self.logs_frame.grid_propagate(False)
        self.logs_frame.grid_rowconfigure(1, weight=1)
        self.logs_frame.grid_columnconfigure(0, weight=1)

        logs_title = tk.Label(
            self.logs_frame,
            text="Recognition Log",
            bg="#111823",
            fg="#e9eef7",
            anchor="w",
            padx=12,
            pady=8,
            font=("Segoe UI", 12, "bold"),
        )
        logs_title.grid(row=0, column=0, sticky="ew")

        self.log_text = tk.Text(
            self.logs_frame,
            bg="#0d1117",
            fg="#d9e2f2",
            insertbackground="#d9e2f2",
            wrap="word",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
            font=("Consolas", 9),
            spacing1=0,
            spacing2=0,
            spacing3=0,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        self.log_toggle_button = tk.Button(
            self.root,
            textvariable=self.log_toggle_text_var,
            command=self.toggle_log_panel,
            bg="#17202c",
            fg="#eef4ff",
            activebackground="#243246",
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            highlightthickness=0,
            padx=0,
            pady=0,
            font=("Segoe UI Symbol", 11, "bold"),
            cursor="hand2",
        )

        self.right_frame = tk.Frame(self.root, bg="#07090d", highlightthickness=0, bd=0)
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.top_bar = tk.Frame(self.right_frame, bg="#0f141b", highlightthickness=0, bd=0, height=104)
        self.top_bar.grid(row=0, column=0, sticky="ew")
        self.top_bar.grid_columnconfigure(0, weight=0)
        self.top_bar.grid_columnconfigure(1, weight=0)
        self.top_bar.grid_columnconfigure(2, weight=1)
        self.top_bar.grid_columnconfigure(3, weight=0)
        self.top_bar.grid_columnconfigure(4, weight=0)
        self.top_bar.grid_propagate(False)

        self.run_button = ttk.Button(
            self.top_bar,
            text="Start recognition",
            style="Run.TButton",
            command=self.start_or_restart_recognition,
        )
        self.run_button.grid(row=0, column=0, sticky="w", padx=10, pady=10)

        self.monitor_selector_frame = tk.Frame(self.top_bar, bg="#0f141b", highlightthickness=0, bd=0)
        self.monitor_selector_frame.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=10)

        self.monitor_label_widget = tk.Label(
            self.monitor_selector_frame,
            text="Screen",
            bg="#0f141b",
            fg="#b9c4d7",
            anchor="w",
            font=("Segoe UI", 9, "bold"),
        )
        self.monitor_label_widget.pack(side="top", anchor="w")

        monitor_values = ["Primary / default"] + [monitor_label(idx, monitor) for idx, monitor in self.monitor_entries]
        self.monitor_combo = ttk.Combobox(
            self.monitor_selector_frame,
            state="readonly",
            width=34,
            textvariable=self.monitor_choice_var,
            values=monitor_values,
        )
        self.monitor_combo.pack(side="top", anchor="w")
        self.monitor_combo.current(0)

        self.info_block = tk.Frame(self.top_bar, bg="#0f141b", highlightthickness=0, bd=0)
        self.info_block.grid(row=0, column=2, sticky="w", padx=(4, 8), pady=0)

        self.best_label = tk.Label(
            self.info_block,
            textvariable=self.best_var,
            bg="#0f141b",
            fg="#f6f8fc",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
        )
        self.best_label.pack(side="top", anchor="w")

        self.selection_label = tk.Label(
            self.info_block,
            textvariable=self.selected_var,
            bg="#0f141b",
            fg="#b9c4d7",
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.selection_label.pack(side="top", anchor="w")

        self.score_label = tk.Label(
            self.info_block,
            textvariable=self.score_var,
            bg="#0f141b",
            fg="#8fb2ff",
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.score_label.pack(side="top", anchor="w")

        self.results_hint_label = tk.Label(
            self.top_bar,
            text="Top 5 matches (click to swap)",
            bg="#0f141b",
            fg="#9fb0c9",
            anchor="e",
            font=("Segoe UI", 10, "bold"),
        )
        self.results_hint_label.grid(row=0, column=3, sticky="e", padx=(8, 6), pady=10)

        self.status_column = tk.Frame(self.top_bar, bg="#0f141b", highlightthickness=0, bd=0)
        self.status_column.grid(row=0, column=4, sticky="e", padx=10, pady=8)

        self.settings_button = tk.Button(
            self.status_column,
            text="⚙",
            command=self.toggle_hotkey_popover,
            bg="#141b24",
            fg="#dce6f7",
            activebackground="#1d2632",
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            padx=8,
            pady=2,
            font=("Segoe UI Symbol", 11, "bold"),
            cursor="hand2",
        )
        self.settings_button.pack(side="top", anchor="e", pady=(0, 4))

        self.status_badge = tk.Label(
            self.status_column,
            textvariable=self.status_var,
            bg="#18212d",
            fg="#dce6f7",
            anchor="center",
            padx=12,
            pady=6,
            font=("Segoe UI", 10, "bold"),
        )
        self.status_badge.pack(side="top", anchor="e")

        self.results_frame = tk.Frame(self.right_frame, bg="#000000", highlightthickness=0, bd=0)
        self.results_frame.grid(row=1, column=0, sticky="nsew")
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(1, weight=0, minsize=220)

        self.main_image_canvas = tk.Canvas(
            self.results_frame,
            bg="#000000",
            bd=0,
            highlightthickness=0,
            relief="flat",
            cursor="fleur",
        )
        self.main_image_canvas.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        self.thumbs_frame = tk.Frame(self.results_frame, bg="#000000", highlightthickness=0, bd=0, width=220)
        self.thumbs_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.thumbs_frame.grid_propagate(False)
        self.thumbs_frame.grid_columnconfigure(0, weight=1)

        self.thumb_labels: List[tk.Label] = []
        for idx in range(THUMB_COUNT):
            self.thumbs_frame.grid_rowconfigure(idx, weight=1)
            lbl = tk.Label(
                self.thumbs_frame,
                bg="#000000",
                activebackground="#000000",
                bd=0,
                borderwidth=0,
                highlightthickness=0,
                highlightbackground="#000000",
                highlightcolor="#000000",
                relief="flat",
                anchor="center",
                cursor="hand2",
                text="",
                image="",
                compound="center",
                padx=0,
                pady=0,
                takefocus=0,
            )
            lbl.grid(row=idx, column=0, sticky="nsew", padx=0, pady=0)
            lbl.bind("<Button-1>", lambda event, i=idx: self._on_thumbnail_click(i))
            self.thumb_labels.append(lbl)

        self.results_frame.bind("<Configure>", self._on_results_configure)
        self.thumbs_frame.bind("<Configure>", self._on_results_configure)
        self.main_image_canvas.bind("<Configure>", self._on_results_configure)
        self._apply_log_panel_width(self.log_panel_current_width)
        self._update_log_toggle_button()
        self._position_log_toggle_button()
        self.main_image_canvas.bind("<MouseWheel>", self._on_main_canvas_mousewheel)
        self.main_image_canvas.bind("<Button-4>", self._on_main_canvas_mousewheel)
        self.main_image_canvas.bind("<Button-5>", self._on_main_canvas_mousewheel)
        self.main_image_canvas.bind("<Double-Button-1>", self._reset_main_zoom)
        self.main_image_canvas.bind("<ButtonPress-1>", self._on_main_canvas_button_press)
        self.main_image_canvas.bind("<B1-Motion>", self._on_main_canvas_drag)
        self.main_image_canvas.bind("<ButtonRelease-1>", self._on_main_canvas_button_release)

    def _update_log_toggle_button(self) -> None:
        target_is_open = self.log_panel_target_width > 0
        self.log_toggle_text_var.set("❮" if target_is_open else "❯")
        self.log_toggle_button.configure(bg="#223147" if target_is_open else "#17202c")

    def _position_log_toggle_button(self) -> None:
        button_height = 72
        root_h = max(1, self.root.winfo_height())
        top_bar_h = max(104, self.top_bar.winfo_height()) if hasattr(self, "top_bar") else 104
        y = min(max(top_bar_h + 12, (root_h - button_height) // 2), max(0, root_h - button_height - 12))
        x = max(0, int(self.log_panel_current_width))
        self.log_toggle_button.place(x=x, y=y, width=LOG_PANEL_TOGGLE_WIDTH, height=button_height)
        self.log_toggle_button.lift()

    def _apply_log_panel_width(self, width: int) -> None:
        width = max(0, min(LOG_PANEL_EXPANDED_WIDTH, int(round(width))))
        self.log_panel_current_width = width
        self.log_tab_container.configure(width=width)
        self.root.grid_columnconfigure(0, minsize=width)

        if width <= 0:
            self.logs_frame.place_forget()
        else:
            self.logs_frame.place(x=width - LOG_PANEL_EXPANDED_WIDTH, y=0, width=LOG_PANEL_EXPANDED_WIDTH, relheight=1.0)

        self._position_log_toggle_button()

    def _ease_log_panel(self, t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return 1.0 - ((1.0 - t) ** 3)

    def _step_log_panel_animation(self) -> None:
        self.log_panel_anim_after_id = None
        target = int(self.log_panel_target_width)
        start = int(self.log_panel_anim_start_width)
        delta = target - start
        if delta == 0:
            self.log_panel_animating = False
            self._apply_log_panel_width(target)
            self._update_log_toggle_button()
            self._schedule_layout_refresh(delay_ms=1)
            return

        elapsed_ms = (time.perf_counter() - self.log_panel_anim_start_time) * 1000.0
        progress = min(1.0, elapsed_ms / max(1.0, float(LOG_PANEL_ANIMATION_DURATION_MS)))
        eased = self._ease_log_panel(progress)
        current = start + delta * eased
        self._apply_log_panel_width(current)

        if progress < 1.0:
            self.log_panel_anim_after_id = self.root.after(LOG_PANEL_ANIMATION_FRAME_MS, self._step_log_panel_animation)
        else:
            self.log_panel_animating = False
            self._apply_log_panel_width(target)
            self._update_log_toggle_button()
            self._schedule_layout_refresh(delay_ms=1)

    def set_log_panel_open(self, is_open: bool, animate: bool = True) -> None:
        self.log_panel_open = bool(is_open)
        self.log_panel_target_width = LOG_PANEL_EXPANDED_WIDTH if self.log_panel_open else 0
        self.settings["log_panel_open"] = self.log_panel_open
        save_settings(self.settings)
        self._update_log_toggle_button()
        if self.log_panel_anim_after_id is not None:
            try:
                self.root.after_cancel(self.log_panel_anim_after_id)
            except Exception:
                pass
            self.log_panel_anim_after_id = None
        if self.layout_refresh_after_id is not None:
            try:
                self.root.after_cancel(self.layout_refresh_after_id)
            except Exception:
                pass
            self.layout_refresh_after_id = None
        if animate and self.log_panel_current_width != self.log_panel_target_width:
            self.log_panel_animating = True
            self.log_panel_anim_start_width = int(self.log_panel_current_width)
            self.log_panel_anim_start_time = time.perf_counter()
            self._step_log_panel_animation()
        else:
            self.log_panel_animating = False
            self._apply_log_panel_width(self.log_panel_target_width)
            self._update_log_toggle_button()
            self._schedule_layout_refresh(delay_ms=1)

    def toggle_log_panel(self) -> None:
        self.set_log_panel_open(not self.log_panel_open, animate=True)

    def _schedule_layout_refresh(self, delay_ms: int = 40) -> None:
        if self.layout_refresh_after_id is not None:
            try:
                self.root.after_cancel(self.layout_refresh_after_id)
            except Exception:
                pass
        self.layout_refresh_after_id = self.root.after(max(1, int(delay_ms)), self._perform_deferred_layout_refresh)

    def _perform_deferred_layout_refresh(self) -> None:
        self.layout_refresh_after_id = None
        total_width = self.results_frame.winfo_width()
        if total_width > 0:
            thumb_width = max(170, int(total_width * 0.20))
            self.results_frame.grid_columnconfigure(1, minsize=thumb_width)
            self.thumbs_frame.configure(width=thumb_width)
        self._refresh_main_canvas_image()
        self._refresh_result_images()

    def _initialize_hotkey_registration(self) -> None:
        success, message = self.hotkey_manager.configure(self.global_hotkey_string)
        self.hotkey_status_var.set(message)
        if success:
            self.settings["global_hotkey"] = self.global_hotkey_string
            save_settings(self.settings)
        else:
            self._append_log_line(self.current_run_id, timestamp_now(), message) if self.current_run_id else None

    def toggle_hotkey_popover(self) -> None:
        if self.hotkey_popover is not None:
            self._close_hotkey_popover()
        else:
            self._open_hotkey_popover()

    def _open_hotkey_popover(self) -> None:
        if self.hotkey_popover is not None:
            return
        popover = tk.Toplevel(self.root)
        popover.transient(self.root)
        popover.resizable(False, False)
        popover.configure(bg="#0f141b")
        popover.title("Hotkey")
        try:
            popover.attributes("-topmost", True)
        except Exception:
            pass
        self.hotkey_popover = popover

        panel = tk.Frame(popover, bg="#111823", highlightthickness=1, highlightbackground="#243244", bd=0)
        panel.pack(fill="both", expand=True)
        self.hotkey_popover_content = panel

        body = tk.Frame(panel, bg="#111823", bd=0)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(body, text="Current hotkey", bg="#111823", fg="#9fb0c9", anchor="w", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(body, textvariable=self.hotkey_capture_var, bg="#0d1117", fg="#f6f8fc", anchor="w", justify="left", padx=8, pady=6, font=("Segoe UI", 10, "bold")).pack(fill="x", pady=(3, 8))

        controls = tk.Frame(body, bg="#111823", bd=0)
        controls.pack(fill="x")
        self.hotkey_listen_button = tk.Button(
            controls,
            text="Set hotkey",
            command=self._toggle_hotkey_capture,
            bg="#1e66ff",
            fg="#ffffff",
            activebackground="#2b74ff",
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        self.hotkey_listen_button.pack(side="left")

        tk.Label(body, text="Click Set hotkey, then press the combination. Press Esc to cancel.", bg="#111823", fg="#9fb0c9", anchor="w", justify="left", wraplength=255, font=("Segoe UI", 8)).pack(anchor="w", pady=(8, 6))
        tk.Label(body, textvariable=self.hotkey_status_var, bg="#111823", fg="#8fb2ff", anchor="w", justify="left", wraplength=255, font=("Segoe UI", 8, "bold")).pack(anchor="w")

        self.root.update_idletasks()
        popover.update_idletasks()
        width = max(280, popover.winfo_reqwidth())
        height = popover.winfo_reqheight()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = max(1, self.root.winfo_width())
        root_h = max(1, self.root.winfo_height())
        x = root_x + max(0, (root_w - width) // 2)
        y = root_y + max(0, (root_h - height) // 2)
        popover.geometry(f"{width}x{height}+{x}+{y}")
        popover.bind("<FocusOut>", self._on_hotkey_popover_focus_out)
        popover.protocol("WM_DELETE_WINDOW", self._close_hotkey_popover)
        self.hotkey_popover_bind_id = self.root.bind_all("<ButtonPress>", self._on_global_button_press_for_popover, add="+")
        popover.lift()
        popover.focus_set()

    def _on_hotkey_popover_focus_out(self, _event=None) -> None:
        if self.hotkey_popover is None:
            return
        self.root.after(10, self._close_hotkey_popover_if_needed)

    def _close_hotkey_popover_if_needed(self) -> None:
        if self.hotkey_popover is None:
            return
        focus_widget = self.root.focus_displayof()
        if focus_widget is None:
            self._close_hotkey_popover()
            return
        widget = focus_widget
        while widget is not None:
            if widget == self.hotkey_popover:
                return
            try:
                widget = widget.master
            except Exception:
                widget = None
        self._close_hotkey_popover()

    def _on_global_button_press_for_popover(self, event) -> None:
        if self.hotkey_popover is None:
            return
        widget = event.widget
        while widget is not None:
            if widget == self.hotkey_popover or widget == self.hotkey_popover_content or widget == self.settings_button:
                return
            try:
                widget = widget.master
            except Exception:
                widget = None
        self._close_hotkey_popover()

    def _close_hotkey_popover(self) -> None:
        if self.hotkey_popover is None:
            return
        self._stop_hotkey_capture()
        if self.hotkey_popover_bind_id is not None:
            try:
                self.root.unbind_all("<ButtonPress>")
            except Exception:
                pass
            self.hotkey_popover_bind_id = None
        try:
            self.hotkey_popover.destroy()
        except Exception:
            pass
        self.hotkey_popover = None
        self.hotkey_popover_content = None

    def _toggle_hotkey_capture(self) -> None:
        if self.hotkey_listening:
            self._stop_hotkey_capture(reset_display=True)
        else:
            self._start_hotkey_capture()

    def _start_hotkey_capture(self) -> None:
        if self.hotkey_popover is None:
            return
        self.hotkey_listening = True
        self.hotkey_pressed_modifiers = set()
        self.hotkey_capture_var.set("Press new shortcut...")
        self.hotkey_status_var.set("Listening for a new global hotkey...")
        self.hotkey_listen_button.configure(text="Cancel")
        self.hotkey_popover.bind("<KeyPress>", self._on_hotkey_capture_keypress)
        self.hotkey_popover.bind("<KeyRelease>", self._on_hotkey_capture_keyrelease)
        self.hotkey_popover.focus_force()

    def _stop_hotkey_capture(self, reset_display: bool = False) -> None:
        self.hotkey_listening = False
        self.hotkey_pressed_modifiers = set()
        if self.hotkey_popover is not None:
            self.hotkey_popover.unbind("<KeyPress>")
            self.hotkey_popover.unbind("<KeyRelease>")
        if hasattr(self, "hotkey_listen_button"):
            self.hotkey_listen_button.configure(text="Set hotkey")
        if reset_display:
            self.hotkey_capture_var.set(self.global_hotkey_string)
            self.hotkey_status_var.set("Capture cancelled.")

    def _on_hotkey_capture_keypress(self, event) -> str:
        if not self.hotkey_listening:
            return "break"
        name = hotkey_from_event_keysym(event.keysym)
        if name is None:
            return "break"
        if name == "Esc":
            self._stop_hotkey_capture(reset_display=True)
            return "break"
        if name in HOTKEY_MODIFIER_SET:
            self.hotkey_pressed_modifiers.add(name)
            current = [m for m in HOTKEY_MODIFIER_ORDER if m in self.hotkey_pressed_modifiers]
            self.hotkey_capture_var.set(" + ".join(current) + (" + ..." if current else "Press new shortcut..."))
            return "break"
        if not self.hotkey_pressed_modifiers:
            self.hotkey_capture_var.set(name)
            self.hotkey_status_var.set("Use at least one modifier key, such as Ctrl or Alt.")
            return "break"
        combo = format_hotkey_combo(self.hotkey_pressed_modifiers, name)
        self.hotkey_capture_var.set(combo)
        success, message = self.hotkey_manager.configure(combo)
        self.hotkey_status_var.set(message)
        if success:
            self.global_hotkey_string = combo
            self.settings["global_hotkey"] = combo
            save_settings(self.settings)
        self._stop_hotkey_capture(reset_display=False)
        self.hotkey_capture_var.set(self.global_hotkey_string)
        return "break"

    def _on_hotkey_capture_keyrelease(self, event) -> str:
        if not self.hotkey_listening:
            return "break"
        name = hotkey_from_event_keysym(event.keysym)
        if name in HOTKEY_MODIFIER_SET and name in self.hotkey_pressed_modifiers:
            self.hotkey_pressed_modifiers.discard(name)
            current = [m for m in HOTKEY_MODIFIER_ORDER if m in self.hotkey_pressed_modifiers]
            self.hotkey_capture_var.set(" + ".join(current) + (" + ..." if current else "Press new shortcut..."))
        return "break"

    def _on_close(self) -> None:
        self._close_hotkey_popover()
        if self.log_panel_anim_after_id is not None:
            try:
                self.root.after_cancel(self.log_panel_anim_after_id)
            except Exception:
                pass
            self.log_panel_anim_after_id = None
        if self.layout_refresh_after_id is not None:
            try:
                self.root.after_cancel(self.layout_refresh_after_id)
            except Exception:
                pass
            self.layout_refresh_after_id = None
        try:
            self.log_toggle_button.place_forget()
        except Exception:
            pass
        self.hotkey_manager.stop()
        self.root.destroy()

    def _queue_log(self, run_id: int, message: str) -> None:
        self.message_queue.put(("log", run_id, message, timestamp_now()))

    def _queue_result(self, run_id: int, payload: dict) -> None:
        self.message_queue.put(("result", run_id, payload))

    def _queue_preview(self, run_id: int, cropped_panel: np.ndarray) -> None:
        self.message_queue.put(("preview", run_id, cropped_panel))

    def _queue_error(self, run_id: int, message: str) -> None:
        self.message_queue.put(("error", run_id, message, timestamp_now()))

    def _queue_cancelled(self, run_id: int) -> None:
        self.message_queue.put(("cancelled", run_id))

    def _queue_finished(self, run_id: int) -> None:
        self.message_queue.put(("finished", run_id))

    def _selected_monitor_index(self) -> int:
        selected_label = self.monitor_choice_var.get()
        if selected_label == "Primary / default":
            return 1
        for idx, monitor in self.monitor_entries:
            if monitor_label(idx, monitor) == selected_label:
                return idx
        return 1

    def start_or_restart_recognition(self) -> None:
        if self.current_cancel_event is not None:
            self.current_cancel_event.set()

        selected_monitor_index = self._selected_monitor_index()
        selected_monitor = next((monitor for idx, monitor in self.monitor_entries if idx == selected_monitor_index), None)

        self.current_run_id += 1
        run_id = self.current_run_id
        cancel_event = threading.Event()
        self.current_cancel_event = cancel_event

        self._reset_logs_for_new_run()
        self._clear_images()
        self.status_var.set("Running")
        self._set_status_badge_color("#17315e", "#dbe7ff")
        self.run_button.configure(text="Restart recognition")
        if selected_monitor is not None:
            self._append_log_line(run_id, timestamp_now(), f"Starting recognition run #{run_id} on {monitor_label(selected_monitor_index, selected_monitor)}.")
        else:
            self._append_log_line(run_id, timestamp_now(), f"Starting recognition run #{run_id} on primary/default display.")

        worker = threading.Thread(target=self._worker_main, args=(run_id, cancel_event, selected_monitor_index), daemon=True)
        self.active_worker = worker
        worker.start()

    def _worker_main(self, run_id: int, cancel_event: threading.Event, monitor_index: int) -> None:
        ctx = RunContext(run_id, cancel_event, self._queue_log)
        try:
            payload = run_recognition(
                ctx,
                monitor_index=monitor_index,
                preview_callback=lambda img: self._queue_preview(run_id, img),
                dynamic_render=os.environ.get('NR_DISABLE_DYNAMIC_RENDER') is None,
                dynamic_render_only=os.environ.get('NR_DISABLE_RENDER_FALLBACK') is None,
            )
            ctx.check_cancel()
            self._queue_result(run_id, payload)
        except CancelledError:
            self._queue_cancelled(run_id)
        except Exception as exc:
            self._queue_error(run_id, f"ERROR: {exc}")
        finally:
            self._queue_finished(run_id)

    def _reset_logs_for_new_run(self) -> None:
        ensure_parent_dir(self.log_file_path)
        self.log_file_path.write_text("", encoding="utf-8")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log_line(self, run_id: int, ts: str, message: str) -> None:
        if run_id != self.current_run_id:
            return
        line = f"[{ts}] {message}\n"
        with self.log_file_path.open("a", encoding="utf-8") as f:
            f.write(line)
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _process_queue(self) -> None:
        try:
            while True:
                item = self.message_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    _, run_id, message, ts = item
                    self._append_log_line(run_id, ts, message)
                elif kind == "error":
                    _, run_id, message, ts = item
                    self._append_log_line(run_id, ts, message)
                    if run_id == self.current_run_id:
                        self.status_var.set("Failed")
                        self._set_status_badge_color("#51212b", "#ffd9de")
                        self.run_button.configure(text="Start recognition")
                elif kind == "cancelled":
                    _, run_id = item
                    if run_id == self.current_run_id:
                        self.status_var.set("Cancelled")
                        self._set_status_badge_color("#3a2f17", "#ffe9b5")
                elif kind == "preview":
                    _, run_id, cropped_panel = item
                    if run_id == self.current_run_id:
                        self._show_preview(cropped_panel)
                elif kind == "hotkey":
                    self.start_or_restart_recognition()
                elif kind == "result":
                    _, run_id, payload = item
                    if run_id == self.current_run_id:
                        self._show_results(payload)
                        self.status_var.set("Completed")
                        self._set_status_badge_color("#1c3c2b", "#d5ffe7")
                        self.run_button.configure(text="Start recognition")
                elif kind == "finished":
                    _, run_id = item
                    if run_id == self.current_run_id:
                        self.current_cancel_event = None
                        if self.status_var.get() == "Running":
                            self.status_var.set("Completed")
                            self._set_status_badge_color("#1c3c2b", "#d5ffe7")
                            self.run_button.configure(text="Start recognition")
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    def _set_status_badge_color(self, bg: str, fg: str) -> None:
        self.status_badge.configure(bg=bg, fg=fg)

    def _clear_images(self) -> None:
        self.current_results = []
        self.current_image_paths = []
        self.current_selected_index = 0
        self.main_photo = None
        self.preview_photo = None
        self.main_canvas_photo = None
        self.current_main_pil_image = None
        self.main_zoom_factor = 1.0
        self.main_pan_x_norm = 0.0
        self.main_pan_y_norm = 0.0
        self.main_drag_last_xy: Optional[Tuple[int, int]] = None
        self.main_placeholder_text = "Run recognition to display results"
        self.thumb_photos = [None] * THUMB_COUNT
        self.best_var.set("Best: —")
        self.selected_var.set("Selected: —")
        self.score_var.set("Score: —")
        self._set_main_placeholder("Run recognition to display results")
        for label in self.thumb_labels:
            label.configure(image="", text="", bg="#000000")

    def _pil_from_bgr(self, img_bgr: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _load_pil_image(self, path: Path | Image.Image) -> Image.Image:
        if isinstance(path, Image.Image):
            if not ENABLE_OVERLAY_TEXT_ONLY_FEATURE:
                return path.convert("RGB") if path.mode != "RGB" else path
            # Handle RGBA images with transparency by replacing transparent pixels with transparent color
            if path.mode == "RGBA":
                # Convert transparent pixels to the overlay's transparent color
                transparent_color_rgb = (0x01, 0x02, 0x03)  # #010203 in RGB
                data = path.getdata()
                new_data = []
                for pixel in data:
                    if len(pixel) == 4 and pixel[3] == 0:  # Transparent pixel
                        new_data.append(transparent_color_rgb)
                    elif len(pixel) == 4:  # Has alpha, make opaque
                        new_data.append(pixel[:3])
                    else:
                        new_data.append(pixel)
                img = Image.new("RGB", path.size)
                img.putdata(new_data)
                return img
            elif path.mode != "RGB":
                return path.convert("RGB")
            return path

        img = self.image_cache.get(path)
        if img is not None:
            return img

        load_error: Optional[Exception] = None
        try:
            img = Image.open(path).convert("RGB")
            self.image_cache[path] = img
            return img
        except Exception as exc:
            load_error = exc

        cv_img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if cv_img is None:
            raise RuntimeError(f"Could not load display image: {path} ({load_error})")

        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        self.image_cache[path] = img
        return img

    def _fit_image(self, pil_image: Image.Image, max_w: int, max_h: int) -> ImageTk.PhotoImage:
        max_w = max(1, max_w)
        max_h = max(1, max_h)
        img = pil_image.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _set_main_placeholder(self, text: str) -> None:
        self.current_main_pil_image = None
        self.main_canvas_photo = None
        self.main_placeholder_text = text
        self.main_zoom_factor = 1.0
        self.main_pan_x_norm = 0.0
        self.main_pan_y_norm = 0.0
        self.main_drag_last_xy = None
        self._refresh_main_canvas_image()

    def _set_main_pil_image(self, pil_image: Image.Image, reset_zoom: bool = False) -> None:
        self.current_main_pil_image = pil_image
        if reset_zoom:
            self.main_zoom_factor = 1.0
            self.main_pan_x_norm = 0.0
            self.main_pan_y_norm = 0.0
        self.main_drag_last_xy = None
        self._refresh_main_canvas_image()

    def _refresh_main_canvas_image(self) -> None:
        canvas_w = max(1, self.main_image_canvas.winfo_width())
        canvas_h = max(1, self.main_image_canvas.winfo_height())
        if canvas_w <= 1 or canvas_h <= 1:
            return

        self.main_image_canvas.delete("all")
        self.main_image_canvas.configure(bg="#000000")

        if self.current_main_pil_image is None:
            self.main_image_canvas.create_text(
                canvas_w / 2,
                canvas_h / 2,
                text=self.main_placeholder_text,
                fill="#657086",
                font=("Segoe UI", 16),
                anchor="center",
            )
            return

        img = self.current_main_pil_image
        img_w, img_h = img.size
        base_scale = min(canvas_w / max(1, img_w), canvas_h / max(1, img_h))
        render_scale = max(0.05, min(12.0, base_scale * self.main_zoom_factor))
        render_w = max(1, int(round(img_w * render_scale)))
        render_h = max(1, int(round(img_h * render_scale)))

        max_pan_x_norm = max(0.0, (render_w - canvas_w) / (2.0 * render_w)) if render_w > 0 else 0.0
        max_pan_y_norm = max(0.0, (render_h - canvas_h) / (2.0 * render_h)) if render_h > 0 else 0.0
        self.main_pan_x_norm = max(-max_pan_x_norm, min(max_pan_x_norm, self.main_pan_x_norm))
        self.main_pan_y_norm = max(-max_pan_y_norm, min(max_pan_y_norm, self.main_pan_y_norm))

        center_x = (canvas_w / 2.0) + (self.main_pan_x_norm * render_w)
        center_y = (canvas_h / 2.0) + (self.main_pan_y_norm * render_h)

        resized = img.resize((render_w, render_h), Image.LANCZOS)
        self.main_canvas_photo = ImageTk.PhotoImage(resized)
        self.main_image_canvas.create_image(center_x, center_y, image=self.main_canvas_photo, anchor="center")

    def _on_main_canvas_mousewheel(self, event) -> str:
        if self.current_main_pil_image is None:
            return "break"
        delta_dir = 0
        if getattr(event, "delta", 0):
            delta_dir = 1 if event.delta > 0 else -1
        elif getattr(event, "num", None) == 4:
            delta_dir = 1
        elif getattr(event, "num", None) == 5:
            delta_dir = -1
        if delta_dir == 0:
            return "break"
        factor = 1.12 if delta_dir > 0 else (1.0 / 1.12)
        new_zoom = max(0.25, min(8.0, self.main_zoom_factor * factor))
        if abs(new_zoom - self.main_zoom_factor) > 1e-9:
            self.main_zoom_factor = new_zoom
            self._refresh_main_canvas_image()
        return "break"

    def _on_main_canvas_button_press(self, event) -> str:
        if self.current_main_pil_image is None:
            return "break"
        self.main_drag_last_xy = (int(event.x), int(event.y))
        return "break"

    def _on_main_canvas_drag(self, event) -> str:
        if self.current_main_pil_image is None or self.main_drag_last_xy is None:
            return "break"

        canvas_w = max(1, self.main_image_canvas.winfo_width())
        canvas_h = max(1, self.main_image_canvas.winfo_height())
        img_w, img_h = self.current_main_pil_image.size
        base_scale = min(canvas_w / max(1, img_w), canvas_h / max(1, img_h))
        render_scale = max(0.05, min(12.0, base_scale * self.main_zoom_factor))
        render_w = max(1.0, img_w * render_scale)
        render_h = max(1.0, img_h * render_scale)

        last_x, last_y = self.main_drag_last_xy
        dx = float(event.x - last_x)
        dy = float(event.y - last_y)
        self.main_drag_last_xy = (int(event.x), int(event.y))

        if render_w > 0:
            self.main_pan_x_norm += dx / render_w
        if render_h > 0:
            self.main_pan_y_norm += dy / render_h

        self._refresh_main_canvas_image()
        return "break"

    def _on_main_canvas_button_release(self, _event=None) -> str:
        self.main_drag_last_xy = None
        return "break"

    def _reset_main_zoom(self, _event=None) -> str:
        self.main_zoom_factor = 1.0
        self.main_pan_x_norm = 0.0
        self.main_pan_y_norm = 0.0
        self.main_drag_last_xy = None
        self._refresh_main_canvas_image()
        return "break"

    def _clear_thumbnail_strip(self) -> None:
        self.thumb_photos = [None] * THUMB_COUNT
        for label in self.thumb_labels:
            label.configure(
                image="",
                text="",
                bg="#000000",
                activebackground="#000000",
                bd=0,
                borderwidth=0,
                highlightthickness=0,
                highlightbackground="#000000",
                highlightcolor="#000000",
                relief="flat",
                compound="center",
            )

    def _show_preview(self, cropped_panel: np.ndarray) -> None:
        pil_img = self._pil_from_bgr(cropped_panel)
        self.current_results = []
        self.current_image_paths = []
        self.preview_photo = None
        self._clear_thumbnail_strip()
        self._set_main_pil_image(pil_img, reset_zoom=True)
        self.best_var.set("Best: recognizing…")
        self.selected_var.set("Selected: captured map preview")
        self.score_var.set("Score: —")

    def _show_results(self, payload: dict) -> None:
        self.current_results = payload["results"]
        self.current_image_paths = payload["top_paths"]
        self.current_selected_index = 0
        self._refresh_best_labels()
        self._refresh_result_images()

    def _refresh_best_labels(self) -> None:
        if not self.current_results:
            self.best_var.set("Best: —")
            self.selected_var.set("Selected: —")
            self.score_var.set("Score: —")
            return

        best = self.current_results[0]
        selected = self.current_results[min(self.current_selected_index, len(self.current_results) - 1)]
        self.best_var.set(f"Best match: #{best.number} ({Path(best.filename).name})")
        self.selected_var.set(f"Selected image: #{selected.number} | rank {min(self.current_selected_index + 1, len(self.current_results))}")
        self.score_var.set(f"Selected score: {selected.score:.6f}")

    def _on_thumbnail_click(self, index: int) -> None:
        if index < len(self.current_image_paths):
            if index == self.current_selected_index:
                return
            self.current_selected_index = index
            self._refresh_best_labels()
            self._refresh_selected_main_image()
            self._update_thumbnail_selection_highlight()

    def _on_results_configure(self, _event=None) -> None:
        self._refresh_main_canvas_image()
        self._refresh_result_images()

    def _on_root_configure(self, _event=None) -> None:
        self._position_log_toggle_button()
        if self.log_panel_animating:
            return
        self._schedule_layout_refresh(delay_ms=25)

    def _refresh_selected_main_image(self) -> None:
        if not self.current_image_paths:
            return
        selected_index = min(self.current_selected_index, len(self.current_image_paths) - 1)
        main_path = self.current_image_paths[selected_index]
        main_img = self._load_pil_image(main_path)
        self._set_main_pil_image(main_img, reset_zoom=False)

    def _update_thumbnail_selection_highlight(self) -> None:
        if not self.current_image_paths:
            return
        selected_index = min(self.current_selected_index, len(self.current_image_paths) - 1)
        for idx, label in enumerate(self.thumb_labels):
            if idx < len(self.current_image_paths):
                if idx == selected_index:
                    label.configure(bg="#111824", activebackground="#111824")
                else:
                    label.configure(bg="#000000", activebackground="#000000")

    def _refresh_result_images(self) -> None:
        if not self.current_image_paths:
            self._clear_thumbnail_strip()
            return

        self._refresh_selected_main_image()

        thumb_frame_h = max(1, self.thumbs_frame.winfo_height())
        thumb_frame_w = max(1, self.thumbs_frame.winfo_width())
        thumb_h = max(1, thumb_frame_h // THUMB_COUNT)
        caption_h = 34
        thumb_img_h = max(1, thumb_h - caption_h)
        for idx, label in enumerate(self.thumb_labels):
            if idx < len(self.current_image_paths):
                thumb_path = self.current_image_paths[idx]
                thumb_img = self._load_pil_image(thumb_path)
                photo = self._fit_image(thumb_img, thumb_frame_w, thumb_img_h)
                self.thumb_photos[idx] = photo
                if idx < len(self.current_results):
                    result = self.current_results[idx]
                    label_text = f"#{result.number}\n{result.score:.4f}"
                else:
                    label_text = ""
                label.configure(
                    image=photo,
                    text=label_text,
                    compound="bottom",
                    fg="#dbe6fb",
                    font=("Segoe UI", 9, "bold"),
                    wraplength=max(1, thumb_frame_w - 8),
                    justify="center",
                    bd=0,
                    borderwidth=0,
                    highlightthickness=0,
                    highlightbackground="#000000",
                    highlightcolor="#000000",
                    relief="flat",
                )
            else:
                self.thumb_photos[idx] = None
                label.configure(
                    image="",
                    text="",
                    bg="#000000",
                    activebackground="#000000",
                    bd=0,
                    borderwidth=0,
                    highlightthickness=0,
                    highlightbackground="#000000",
                    highlightcolor="#000000",
                    relief="flat",
                    compound="center",
                )
        self._update_thumbnail_selection_highlight()



# --- Slot-based recognition backend (v16) ---
SLOT_PACKAGE_FILENAME = "nightreign_geom_proto_v4_slots.npz"
SLOT_LANDMARK_TYPES = [
    "Church",
    "Fort",
    "Fortified Village",
    "Great Church",
    "Main Encampment",
    "Ruins",
    "Sorcerer's Rise",
    "Swamp",
    "Township",
]
SLOT_TYPE_CONFIG = {
    "Church": {
        "window_frac": 0.110,
        "empty_threshold": 0.28,
        "accept": 0.40,
        "margin": 0.025,
        "rgb_weight": 0.16,
        "gray_weight": 0.22,
        "edge_weight": 0.62,
    },
    "Fort": {
        "window_frac": 0.090,
        "empty_threshold": 0.30,
        "accept": 0.43,
        "margin": 0.030,
        "rgb_weight": 0.38,
        "gray_weight": 0.22,
        "edge_weight": 0.40,
    },
    "Fortified Village": {
        "window_frac": 0.098,
        "empty_threshold": 0.30,
        "accept": 0.42,
        "margin": 0.028,
        "rgb_weight": 0.34,
        "gray_weight": 0.22,
        "edge_weight": 0.44,
    },
    "Great Church": {
        "window_frac": 0.086,
        "empty_threshold": 0.30,
        "accept": 0.43,
        "margin": 0.030,
        "rgb_weight": 0.38,
        "gray_weight": 0.20,
        "edge_weight": 0.42,
    },
    "Main Encampment": {
        "window_frac": 0.100,
        "empty_threshold": 0.29,
        "accept": 0.40,
        "margin": 0.028,
        "rgb_weight": 0.30,
        "gray_weight": 0.22,
        "edge_weight": 0.48,
    },
    "Ruins": {
        "window_frac": 0.086,
        "empty_threshold": 0.30,
        "accept": 0.42,
        "margin": 0.028,
        "rgb_weight": 0.34,
        "gray_weight": 0.18,
        "edge_weight": 0.48,
    },
    "Sorcerer's Rise": {
        "window_frac": 0.118,
        "empty_threshold": 0.26,
        "accept": 0.36,
        "margin": 0.020,
        "rgb_weight": 0.14,
        "gray_weight": 0.22,
        "edge_weight": 0.64,
    },
    "Swamp": {
        "window_frac": 0.104,
        "empty_threshold": 0.29,
        "accept": 0.40,
        "margin": 0.026,
        "rgb_weight": 0.22,
        "gray_weight": 0.18,
        "edge_weight": 0.60,
    },
    "Township": {
        "window_frac": 0.110,
        "empty_threshold": 0.27,
        "accept": 0.38,
        "margin": 0.022,
        "rgb_weight": 0.18,
        "gray_weight": 0.22,
        "edge_weight": 0.60,
    },
}
SLOT_TEMPLATE_SCALES = (0.60, 0.72, 0.86, 1.00, 1.16, 1.34, 1.54)
SLOT_CANONICAL_PANEL_SIZE = 1536
SLOT_EARTH_SECONDARY_THRESHOLD = 0.025
SLOT_CROP_REFINE_SHIFT_FRACS = (-0.04, -0.02, 0.0, 0.02, 0.04)
SLOT_CROP_REFINE_SCALE_FACTORS = (0.97, 1.00, 1.03)
SLOT_CROP_REFINE_MARGIN_WEIGHT = 0.35
SLOT_CROP_REFINE_BLACK_WEIGHT = 0.12
SLOT_CROP_REFINE_EDGE_WEIGHT = 0.03

SLOT_CONTEXT_TOP_LAYOUTS = 12
SLOT_CONTEXT_WEIGHT_TEMPERATURE = 3.0
SLOT_CONTEXT_CONSENSUS_SHARE = 0.58
SLOT_CONTEXT_STRONG_CONSENSUS_SHARE = 0.75
SLOT_CONTEXT_PROMOTE_MIN_SCORE = 0.55
SLOT_CONTEXT_MAX_SCORE_GAP = 0.045
SLOT_CONTEXT_STRONG_MAX_SCORE_GAP = 0.080



@dataclass
class SlotObservation:
    slot_id: int
    slot_name: str
    label: str
    best_score: float
    scores: Dict[str, float]
    best_x_norm: float
    best_y_norm: float
    expected_types: List[str]


@dataclass
class SlotMatchResult:
    layout_number: int
    shifting_earth: str
    total_score: float
    earth_score: float
    details: Dict[str, float]


class SlotRecognitionPackage:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata_json"]))
            self.earth_names = [str(x) for x in data["earth_names"]]
            self.terrain_low = data["terrain_low"]
            self.terrain_grad = data["terrain_grad"]
            self.landmark_types = [str(x) for x in data["landmark_types"]]
            self.icon_names = [str(x) for x in data["icon_names"]]
            self.icon_templates = data["icon_templates"]
            self.icon_templates_gray = data["icon_templates_gray"]
            self.icon_templates_edge = data["icon_templates_edge"]
            self.icon_actual_sizes = data["icon_actual_sizes"]
        self.earth_to_idx = {name: idx for idx, name in enumerate(self.earth_names)}
        self.type_to_idx = {name: idx for idx, name in enumerate(self.landmark_types)}
        self.layouts = list(self.metadata["layouts"])
        self.layouts_by_earth: Dict[str, List[dict]] = {name: [] for name in self.earth_names}
        for item in self.layouts:
            self.layouts_by_earth[str(item["shifting_earth"])].append(item)
        self.slots_by_earth: Dict[str, List[dict]] = {
            key: list(value) for key, value in self.metadata["slots_by_earth"].items()
        }
        self._template_variants_cache: Dict[str, List[dict]] = {}

    def template_variants(self, type_name: str) -> List[dict]:
        cached = self._template_variants_cache.get(type_name)
        if cached is not None:
            return cached
        type_idx = self.type_to_idx[type_name]
        rgba = self.icon_templates[type_idx]
        gray_t = self.icon_templates_gray[type_idx]
        edge_t = self.icon_templates_edge[type_idx]
        x0, y0, x1, y1 = _slot_alpha_bbox(rgba)
        rgba = rgba[y0:y1, x0:x1]
        gray_t = gray_t[y0:y1, x0:x1]
        edge_t = edge_t[y0:y1, x0:x1]
        rgb_base = rgba[:, :, :3]
        mask_base = rgba[:, :, 3]
        variants: List[dict] = []
        for scale in SLOT_TEMPLATE_SCALES:
            new_w = max(8, int(round(rgb_base.shape[1] * scale)))
            new_h = max(8, int(round(rgb_base.shape[0] * scale)))
            rgb_t = cv2.resize(rgb_base, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gray_rs = cv2.resize(gray_t, (new_w, new_h), interpolation=cv2.INTER_AREA)
            edge_rs = cv2.resize(edge_t, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_rs = cv2.resize(mask_base, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_u8 = (mask_rs > 32).astype(np.uint8) * 255
            cx = new_w // 2
            cy = new_h // 2
            variants.append({
                "rgb": np.ascontiguousarray(rgb_t),
                "gray": np.ascontiguousarray(gray_rs),
                "edge": np.ascontiguousarray(edge_rs),
                "mask": np.ascontiguousarray(mask_u8),
                "w": int(new_w),
                "h": int(new_h),
                "cx": int(cx),
                "cy": int(cy),
            })
        self._template_variants_cache[type_name] = variants
        return variants


def recognition_package_search_paths() -> List[Path]:
    rb = resource_base_dir()
    rd = runtime_dir()
    candidates = [
        rd / SLOT_PACKAGE_FILENAME,
        rb / SLOT_PACKAGE_FILENAME,
    ]
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_slot_package_only(ctx: RunContext) -> Tuple[SlotRecognitionPackage, Path]:
    checked: List[str] = []
    for path in recognition_package_search_paths():
        checked.append(str(path.resolve() if path.exists() else path))
        if not path.exists():
            continue
        ctx.check_cancel()
        ctx.log(f"Trying slot package: {path.resolve()}")
        package = SlotRecognitionPackage(path)
        return package, path
    checked_block = "\n- ".join(checked) if checked else "(no package paths configured)"
    raise FileNotFoundError(
        "Slot recognition package not found. Checked these locations:\n- "
        + checked_block
        + "\n\nGenerate the slot package first with the v4 slot-based renderer and place the .npz next to this script/app."
    )


def _slot_finite_score_value(value: float, default: float = -1.0, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        value_f = float(value)
    except Exception:
        return default
    if not math.isfinite(value_f):
        return default
    if value_f < lo:
        return lo
    if value_f > hi:
        return hi
    return value_f


def _slot_sanitize_score_array(arr: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    out = np.nan_to_num(arr.astype(np.float32), nan=lo, posinf=hi, neginf=lo)
    return np.clip(out, lo, hi)


def normalize_panel_for_slot_recognition(panel_bgr: np.ndarray, target_size: int = SLOT_CANONICAL_PANEL_SIZE) -> np.ndarray:
    h, w = panel_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Panel image is empty.")
    if h == target_size and w == target_size:
        return panel_bgr.copy()
    interpolation = cv2.INTER_CUBIC if min(h, w) < target_size else cv2.INTER_AREA
    return cv2.resize(panel_bgr, (target_size, target_size), interpolation=interpolation)


def slot_crop_sanity_metrics(panel_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    black_ratio = float(np.mean(gray < 12))
    edge = build_panel_edge_slot(panel_bgr)
    edge_mean = float(edge.mean()) / 255.0
    return {"black_ratio": black_ratio, "edge_mean": edge_mean}


def make_square_box_from_center_slot(cx: float, cy: float, side: float, w: int, h: int) -> CropBox:
    side_i = max(32, min(int(round(side)), w, h))
    x0 = int(round(cx - side_i / 2.0))
    y0 = int(round(cy - side_i / 2.0))
    x0 = max(0, min(w - side_i, x0))
    y0 = max(0, min(h - side_i, y0))
    return CropBox(x0, y0, x0 + side_i, y0 + side_i)


def evaluate_crop_candidate_slot(raw_panel_bgr: np.ndarray, package: SlotRecognitionPackage) -> Tuple[float, List[Tuple[str, float, Dict[str, float]]], Dict[str, float], np.ndarray]:
    panel_bgr = normalize_panel_for_slot_recognition(raw_panel_bgr)
    metrics = slot_crop_sanity_metrics(panel_bgr)
    earth_rank = classify_shifting_earth_ai(raw_panel_bgr, panel_bgr, package, None)
    top_score = earth_rank[0][1] if earth_rank else -1.0
    second_score = earth_rank[1][1] if len(earth_rank) > 1 else (top_score - 1.0)
    margin = top_score - second_score
    objective = (
        float(top_score)
        + SLOT_CROP_REFINE_MARGIN_WEIGHT * float(margin)
        - SLOT_CROP_REFINE_BLACK_WEIGHT * float(metrics["black_ratio"])
        + SLOT_CROP_REFINE_EDGE_WEIGHT * float(metrics["edge_mean"])
    )
    return objective, earth_rank, metrics, panel_bgr


def refine_crop_box_by_earth_score_slot(
    full_img: np.ndarray,
    initial_box: CropBox,
    package: SlotRecognitionPackage,
    ctx: Optional[RunContext] = None,
) -> Tuple[CropBox, np.ndarray, np.ndarray, Dict[str, float], List[Tuple[str, float, Dict[str, float]]], Dict[str, object]]:
    h, w = full_img.shape[:2]
    base_side = float(min(initial_box.x1 - initial_box.x0, initial_box.y1 - initial_box.y0))
    base_cx = (initial_box.x0 + initial_box.x1) / 2.0
    base_cy = (initial_box.y0 + initial_box.y1) / 2.0

    best: Optional[Tuple[float, CropBox, np.ndarray, np.ndarray, Dict[str, float], List[Tuple[str, float, Dict[str, float]]], Dict[str, float]]] = None
    seen: set[Tuple[int, int, int, int]] = set()
    tried = 0

    for scale_factor in SLOT_CROP_REFINE_SCALE_FACTORS:
        side = base_side * float(scale_factor)
        for dx_frac in SLOT_CROP_REFINE_SHIFT_FRACS:
            for dy_frac in SLOT_CROP_REFINE_SHIFT_FRACS:
                if ctx is not None:
                    ctx.check_cancel()
                cand_box = make_square_box_from_center_slot(
                    base_cx + dx_frac * base_side,
                    base_cy + dy_frac * base_side,
                    side,
                    w,
                    h,
                )
                key = (cand_box.x0, cand_box.y0, cand_box.x1, cand_box.y1)
                if key in seen:
                    continue
                seen.add(key)
                tried += 1
                raw_panel_bgr = full_img[cand_box.y0:cand_box.y1, cand_box.x0:cand_box.x1]
                objective, earth_rank, metrics, panel_bgr = evaluate_crop_candidate_slot(raw_panel_bgr, package)
                top_score = earth_rank[0][1] if earth_rank else -1.0
                second_score = earth_rank[1][1] if len(earth_rank) > 1 else (top_score - 1.0)
                debug_info = {
                    "objective": float(objective),
                    "top_score": float(top_score),
                    "margin": float(top_score - second_score),
                    "dx_frac": float(dx_frac),
                    "dy_frac": float(dy_frac),
                    "scale_factor": float(scale_factor),
                }
                if best is None or objective > best[0]:
                    best = (objective, cand_box, raw_panel_bgr, panel_bgr, metrics, earth_rank, debug_info)

    if best is None:
        raw_panel_bgr = full_img[initial_box.y0:initial_box.y1, initial_box.x0:initial_box.x1]
        objective, earth_rank, metrics, panel_bgr = evaluate_crop_candidate_slot(raw_panel_bgr, package)
        best = (
            objective,
            initial_box,
            raw_panel_bgr,
            panel_bgr,
            metrics,
            earth_rank,
            {"objective": float(objective), "top_score": float(earth_rank[0][1] if earth_rank else -1.0), "margin": 0.0, "dx_frac": 0.0, "dy_frac": 0.0, "scale_factor": 1.0},
        )

    _, best_box, best_raw_panel, best_panel, best_metrics, best_earth_rank, best_debug = best
    refine_info: Dict[str, object] = {
        "tried_candidates": int(tried),
        "best_objective": float(best_debug["objective"]),
        "best_top_score": float(best_debug["top_score"]),
        "best_margin": float(best_debug["margin"]),
        "best_dx_frac": float(best_debug["dx_frac"]),
        "best_dy_frac": float(best_debug["dy_frac"]),
        "best_scale_factor": float(best_debug["scale_factor"]),
        "initial_box": initial_box.__dict__,
        "best_box": best_box.__dict__,
    }
    return best_box, best_raw_panel, best_panel, best_metrics, best_earth_rank, refine_info


def _slot_alpha_bbox(rgba: np.ndarray) -> Tuple[int, int, int, int]:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, rgba.shape[1], rgba.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def build_panel_edge_slot(panel_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.2)
    if float(mag.max()) > 1e-6:
        mag = mag / float(mag.max()) * 255.0
    return mag.astype(np.uint8)


def build_panel_representations_slot(panel_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "rgb": cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB),
        "gray": cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY),
        "edge": build_panel_edge_slot(panel_bgr),
    }


def classify_shifting_earth_slot(panel_bgr: np.ndarray, package: SlotRecognitionPackage) -> List[Tuple[str, float, Dict[str, float]]]:
    shot_low = preprocess_for_similarity(panel_bgr)
    shot_grad = preprocess_for_gradients(panel_bgr)
    shot_low_n = normalize(shot_low)
    shot_grad_n = normalize(shot_grad)
    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for idx, earth_name in enumerate(package.earth_names):
        low_corr = corr(shot_low_n, normalize(package.terrain_low[idx]))
        grad_corr = corr(shot_grad_n, normalize(package.terrain_grad[idx]))
        score = 0.65 * low_corr + 0.35 * grad_corr
        ranked.append((earth_name, score, {"low_corr": low_corr, "grad_corr": grad_corr}))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def estimate_global_translation_slot(panel_bgr: np.ndarray, package: SlotRecognitionPackage, earth_name: str) -> Tuple[float, float, float]:
    idx = package.earth_to_idx[earth_name]
    shot_low = preprocess_for_similarity(panel_bgr).astype(np.float32)
    ref_low = package.terrain_low[idx].astype(np.float32)
    shift, response = cv2.phaseCorrelate(ref_low, shot_low)
    dx_norm = float(np.clip(shift[0] / CMP_SIZE, -0.04, 0.04))
    dy_norm = float(np.clip(shift[1] / CMP_SIZE, -0.05, 0.05))
    return dx_norm, dy_norm, float(response)


def local_template_score_slot(panel_repr: Dict[str, np.ndarray], package: SlotRecognitionPackage, type_name: str, cx: int, cy: int, search_radius: int) -> Tuple[float, Tuple[int, int]]:
    cfg = SLOT_TYPE_CONFIG[type_name]
    panel_rgb = panel_repr["rgb"]
    panel_gray = panel_repr["gray"]
    panel_edge = panel_repr["edge"]
    h, w = panel_gray.shape[:2]
    best_score = -1.0
    best_xy = (cx, cy)
    for variant in package.template_variants(type_name):
        tw = int(variant["w"])
        th = int(variant["h"])
        halfw = search_radius + tw // 2 + 2
        halfh = search_radius + th // 2 + 2
        x0 = max(0, cx - halfw)
        y0 = max(0, cy - halfh)
        x1 = min(w, cx + halfw)
        y1 = min(h, cy + halfh)
        if x1 - x0 <= tw or y1 - y0 <= th:
            continue
        patch_rgb = panel_rgb[y0:y1, x0:x1]
        patch_gray = panel_gray[y0:y1, x0:x1]
        patch_edge = panel_edge[y0:y1, x0:x1]
        rgb_score = _slot_sanitize_score_array(cv2.matchTemplate(patch_rgb, variant["rgb"], cv2.TM_CCORR_NORMED, mask=variant["mask"]))
        gray_score = _slot_sanitize_score_array(cv2.matchTemplate(patch_gray, variant["gray"], cv2.TM_CCORR_NORMED, mask=variant["mask"]))
        edge_score = _slot_sanitize_score_array(cv2.matchTemplate(patch_edge, variant["edge"], cv2.TM_CCORR_NORMED, mask=variant["mask"]))
        combined = _slot_sanitize_score_array(
            float(cfg["rgb_weight"]) * rgb_score
            + float(cfg["gray_weight"]) * gray_score
            + float(cfg["edge_weight"]) * edge_score
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(combined)
        max_val = _slot_finite_score_value(max_val)
        if max_val > best_score:
            best_score = max_val
            best_xy = (x0 + int(max_loc[0]) + int(variant["cx"]), y0 + int(max_loc[1]) + int(variant["cy"]))
    return _slot_finite_score_value(best_score), best_xy


def classify_slots_for_earth_slot(panel_bgr: np.ndarray, package: SlotRecognitionPackage, earth_name: str) -> Tuple[List[SlotObservation], Dict[str, float]]:
    dx_norm, dy_norm, align_response = estimate_global_translation_slot(panel_bgr, package, earth_name)
    panel_repr = build_panel_representations_slot(panel_bgr)
    h, w = panel_bgr.shape[:2]
    side = min(h, w)
    observations: List[SlotObservation] = []
    for slot in package.slots_by_earth[earth_name]:
        base_cx = int(round((float(slot["x_norm"]) + dx_norm) * w))
        base_cy = int(round((float(slot["y_norm"]) + dy_norm) * h))
        base_cx = max(0, min(w - 1, base_cx))
        base_cy = max(0, min(h - 1, base_cy))
        search_radius = max(
            int(round(float(slot["radius_norm"]) * side * 1.25)),
            max(int(round(SLOT_TYPE_CONFIG[t]["window_frac"] * side * 0.55)) for t in slot.get("allowed_types", SLOT_LANDMARK_TYPES))
        )
        type_scores: Dict[str, float] = {}
        type_positions: Dict[str, Tuple[int, int]] = {}
        candidate_types = list(slot.get("allowed_types", SLOT_LANDMARK_TYPES)) or SLOT_LANDMARK_TYPES
        for type_name in candidate_types:
            score, best_xy = local_template_score_slot(panel_repr, package, type_name, base_cx, base_cy, search_radius)
            type_scores[type_name] = _slot_finite_score_value(score)
            type_positions[type_name] = best_xy
        if type_scores:
            ordered = sorted(type_scores.items(), key=lambda kv: kv[1], reverse=True)
            best_label, best_score = ordered[0]
            second_score = ordered[1][1] if len(ordered) > 1 else -1.0
        else:
            best_label, best_score, second_score = ("empty", -1.0, -1.0)
        best_score = _slot_finite_score_value(best_score)
        second_score = _slot_finite_score_value(second_score)
        cfg = SLOT_TYPE_CONFIG.get(best_label, SLOT_TYPE_CONFIG["Ruins"])
        if best_score < float(cfg["empty_threshold"]):
            final_label = "empty"
            best_xy = (base_cx, base_cy)
        elif best_score < float(cfg["accept"]) or (best_score - second_score) < float(cfg["margin"]):
            final_label = "unknown"
            best_xy = type_positions.get(best_label, (base_cx, base_cy))
        else:
            final_label = best_label
            best_xy = type_positions.get(best_label, (base_cx, base_cy))
        observations.append(
            SlotObservation(
                slot_id=int(slot["slot_id"]),
                slot_name=str(slot["slot_name"]),
                label=final_label,
                best_score=float(best_score),
                scores={k: _slot_finite_score_value(v) for k, v in type_scores.items()},
                best_x_norm=float(best_xy[0] / w),
                best_y_norm=float(best_xy[1] / h),
                expected_types=list(slot.get("allowed_types", SLOT_LANDMARK_TYPES)),
            )
        )
    return observations, {"dx_norm": dx_norm, "dy_norm": dy_norm, "response": _slot_finite_score_value(align_response, default=0.0, lo=-1.0, hi=1.0)}



def refine_unknown_slots_by_layout_consensus_slot(
    observations: Sequence[SlotObservation],
    layouts_for_earth: Sequence[dict],
    scored_results: Sequence[SlotMatchResult],
) -> Tuple[List[SlotObservation], Dict[str, float]]:
    if not observations or not layouts_for_earth or not scored_results:
        return list(observations), {"promoted_unknowns": 0.0, "considered_unknowns": 0.0}
    layout_by_number = {int(layout["layout_number"]): layout for layout in layouts_for_earth}
    usable_results = [item for item in scored_results if math.isfinite(item.total_score)]
    if not usable_results:
        return list(observations), {"promoted_unknowns": 0.0, "considered_unknowns": 0.0}
    usable_results.sort(key=lambda item: item.total_score, reverse=True)
    top_results = usable_results[:SLOT_CONTEXT_TOP_LAYOUTS]
    top_score = float(top_results[0].total_score)
    weighted_results: List[Tuple[SlotMatchResult, float]] = []
    for item in top_results:
        weight = math.exp((float(item.total_score) - top_score) / SLOT_CONTEXT_WEIGHT_TEMPERATURE)
        weighted_results.append((item, weight))
    new_observations: List[SlotObservation] = []
    promoted_unknowns = 0
    considered_unknowns = 0
    for slot_idx, obs in enumerate(observations):
        if obs.label != "unknown" or not obs.scores:
            new_observations.append(obs)
            continue
        considered_unknowns += 1
        vote_total = 0.0
        vote_by_label: Dict[str, float] = {}
        for item, weight in weighted_results:
            layout = layout_by_number.get(int(item.layout_number))
            if layout is None:
                continue
            expected = str(layout["slot_labels"][slot_idx])
            if expected == "empty":
                continue
            vote_total += weight
            vote_by_label[expected] = vote_by_label.get(expected, 0.0) + weight
        if not vote_by_label or vote_total <= 0.0:
            new_observations.append(obs)
            continue
        consensus_label, consensus_weight = max(vote_by_label.items(), key=lambda kv: kv[1])
        vote_share = consensus_weight / vote_total
        label_score = _slot_finite_score_value(obs.scores.get(consensus_label, -1.0))
        top_observed_score = max((_slot_finite_score_value(v) for v in obs.scores.values()), default=-1.0)
        score_gap = top_observed_score - label_score
        score_gap_limit = SLOT_CONTEXT_STRONG_MAX_SCORE_GAP if vote_share >= SLOT_CONTEXT_STRONG_CONSENSUS_SHARE else SLOT_CONTEXT_MAX_SCORE_GAP
        if (
            vote_share >= SLOT_CONTEXT_CONSENSUS_SHARE
            and label_score >= SLOT_CONTEXT_PROMOTE_MIN_SCORE
            and score_gap <= score_gap_limit
        ):
            new_observations.append(replace(obs, label=consensus_label, best_score=float(label_score)))
            promoted_unknowns += 1
        else:
            new_observations.append(obs)
    return new_observations, {
        "promoted_unknowns": float(promoted_unknowns),
        "considered_unknowns": float(considered_unknowns),
    }



def score_layout_slotwise(slot_observations: Sequence[SlotObservation], layout_slot_labels: Sequence[str], slot_defs: Sequence[dict], earth_score: float, earth_name: str, layout_number: int) -> SlotMatchResult:
    total = 0.0
    exact_matches = 0.0
    hard_mismatches = 0.0
    unknown_count = 0.0
    for obs, exp, slot in zip(slot_observations, layout_slot_labels, slot_defs):
        weight = float(slot.get("weight", 1.0))
        type_scores = {k: _slot_finite_score_value(v) for k, v in obs.scores.items()}
        max_nonempty = max(type_scores.values()) if type_scores else 0.0
        if exp == "empty":
            contrib = 0.85 - 2.20 * max_nonempty
            if obs.label == "empty":
                contrib += 0.18
        else:
            exp_score = _slot_finite_score_value(type_scores.get(exp, 0.0), default=0.0)
            best_other = max((_slot_finite_score_value(score, default=0.0) for t, score in type_scores.items() if t != exp), default=0.0)
            contrib = 2.4 * exp_score - 1.9 * best_other
            if obs.label == exp:
                contrib += 0.55
                exact_matches += weight
            elif obs.label == "unknown":
                unknown_count += weight
            elif obs.label == "empty":
                contrib -= 0.95
                hard_mismatches += weight
            elif obs.label != exp:
                contrib -= 0.85
                hard_mismatches += weight
        total += weight * _slot_finite_score_value(contrib, default=0.0, lo=-1000.0, hi=1000.0)
    total += 0.35 * _slot_finite_score_value(earth_score, default=0.0, lo=-1.0, hi=1.0)
    return SlotMatchResult(
        layout_number=int(layout_number),
        shifting_earth=earth_name,
        total_score=float(_slot_finite_score_value(total, default=-1e9, lo=-1e9, hi=1e9)),
        earth_score=float(_slot_finite_score_value(earth_score, default=0.0, lo=-1.0, hi=1.0)),
        details={
            "exact_match_weight": float(exact_matches),
            "hard_mismatch_weight": float(hard_mismatches),
            "unknown_weight": float(unknown_count),
        },
    )



def run_recognition(
    ctx: RunContext,
    monitor_index: int = 1,
    preview_callback: Optional[Callable[[np.ndarray], None]] = None,
    pre_captured_full_img: Optional[np.ndarray] = None,
    second_full_img: Optional[np.ndarray] = None,
    dual_shot_enabled: bool = False,
    dynamic_render: bool = True,
) -> dict:
    total_start = time.perf_counter()

    disp_dir = choose_existing_display_dir()
    if disp_dir is None:
        checked = "\n".join(f"- {p}" for p in display_search_paths())
        raise FileNotFoundError(
            "Display directory not found. Checked these locations:\n" + checked
        )

    ctx.log(f"Display directory: {disp_dir.resolve()}")
    ctx.log(f"Log file: {log_file_path().resolve()}")

    selected_monitor = None
    for idx, monitor in get_available_monitors():
        if idx == monitor_index:
            selected_monitor = monitor
            break
    if selected_monitor is None:
        raise RuntimeError(f"Selected display {monitor_index} is no longer available.")
    ctx.log(f"Selected display: {monitor_label(monitor_index, selected_monitor)}")

    proto = load_proto_backend_module(ctx)
    package_path = find_slot_package_path(ctx)

    t = step_start(ctx, "Load authoritative proto recognition package")
    package = proto.RecognitionPackage(package_path)
    step_end(ctx, "Load authoritative proto recognition package", t)
    ctx.log(f"Using slot package: {Path(package_path).resolve()}")

    t = step_start(ctx, "Capture screenshot")
    full_img = proto.capture_monitor_bgr(monitor_index)
    step_end(ctx, "Capture screenshot", t)

    t = step_start(ctx, "Detect rough map circle")
    crop_circle, circle_method = proto.detect_map_circle(full_img)
    step_end(ctx, "Detect rough map circle", t)
    ctx.log(f"Circle detection method: {circle_method}")

    crop_refinement_info: Optional[Dict[str, object]] = None
    t = step_start(ctx, "Extract cropped map panel")
    if circle_method == "blue_ring":
        crop_box = proto.clip_box(
            proto.estimate_panel_box(crop_circle, proto.DEFAULT_PANEL_SCALE),
            full_img.shape[1],
            full_img.shape[0],
        )
        raw_cropped_panel = full_img[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
        cropped_panel = proto.normalize_panel_for_recognition(raw_cropped_panel)
        metrics = proto.crop_sanity_metrics(cropped_panel)
        earth_rank = proto.classify_shifting_earth(cropped_panel, package)
        crop_method = "blue_ring"
    else:
        initial_box = proto.refine_panel_box(full_img, crop_circle)
        crop_box, raw_cropped_panel, cropped_panel, metrics, earth_rank, crop_refinement_info = proto.refine_crop_box_by_earth_score(
            full_img, initial_box, package
        )
        crop_method = "gray_hough_refined"
        ctx.log(
            "Crop refinement: "
            f"tried={int(crop_refinement_info['tried_candidates'])} "
            f"dx_frac={float(crop_refinement_info['best_dx_frac']):+.3f} "
            f"dy_frac={float(crop_refinement_info['best_dy_frac']):+.3f} "
            f"scale={float(crop_refinement_info['best_scale_factor']):.3f} "
            f"top_score={float(crop_refinement_info['best_top_score']):.6f} "
            f"margin={float(crop_refinement_info['best_margin']):.6f}"
        )
    ctx.log(
        f"Panel normalization: raw={raw_cropped_panel.shape[1]}x{raw_cropped_panel.shape[0]} -> "
        f"normalized={cropped_panel.shape[1]}x{cropped_panel.shape[0]} black_ratio={metrics['black_ratio']:.3f} edge_mean={metrics['edge_mean']:.3f}"
    )
    if preview_callback is not None:
        preview_callback(cropped_panel)
    step_end(ctx, "Extract cropped map panel", t)

    t = step_start(ctx, "Classify Shifting Earth")
    step_end(ctx, "Classify Shifting Earth", t)
    ctx.log("Shifting Earth ranking:")
    for name, score, details in earth_rank[:5]:
        ctx.log(f"  {name:<14} score={score:.6f} low_corr={details['low_corr']:.4f} grad_corr={details['grad_corr']:.4f}")
    if earth_rank and earth_rank[0][1] < 0.08:
        ctx.log("WARNING: very low top Shifting Earth score; the crop may be invalid or the on-screen map scale may differ from expected.")

    candidate_earths = [earth_rank[0][0]]
    if len(earth_rank) > 1 and (earth_rank[0][1] - earth_rank[1][1]) < proto.EARTH_SECONDARY_THRESHOLD:
        candidate_earths.append(earth_rank[1][0])

    all_results = []
    all_observation_sets: Dict[str, Tuple[List[object], Dict[str, float]]] = {}
    all_context_stats: Dict[str, Dict[str, float]] = {}
    for earth_name in candidate_earths:
        ctx.check_cancel()
        earth_score = next(score for name, score, _ in earth_rank if name == earth_name)
        t = step_start(ctx, f"Classify landmark slots for {earth_name}")
        observations, align_info = proto.classify_slots_for_earth(cropped_panel, package, earth_name)
        step_end(ctx, f"Classify landmark slots for {earth_name}", t)
        ctx.log(
            f"Alignment for {earth_name}: dx_norm={align_info['dx_norm']:.4f} dy_norm={align_info['dy_norm']:.4f} response={align_info['align_response']:.4f}"
        )
        layouts_for_earth = package.layouts_by_earth[earth_name]
        initial_results = []
        total = len(layouts_for_earth)
        for idx, layout in enumerate(layouts_for_earth, start=1):
            ctx.check_cancel()
            result = proto.score_layout(observations, layout["slot_labels"], package.slots_by_earth[earth_name], earth_score, earth_name, int(layout["layout_number"]))
            initial_results.append(result)
            if idx == 1 or idx % 25 == 0 or idx == total:
                ctx.log(f"Layout scoring progress for {earth_name}: {idx}/{total}")
        refined_observations, context_stats = proto.refine_unknown_slots_by_layout_consensus(observations, layouts_for_earth, initial_results)
        all_context_stats[earth_name] = context_stats
        if context_stats.get("promoted_unknowns", 0.0) > 0.0:
            ctx.log(
                f"Context refinement for {earth_name}: promoted "
                f"{int(context_stats['promoted_unknowns'])} of {int(context_stats['considered_unknowns'])} unknown slots"
            )
        observations = refined_observations
        all_observation_sets[earth_name] = (observations, align_info)
        for layout in layouts_for_earth:
            ctx.check_cancel()
            all_results.append(
                proto.score_layout(
                    observations,
                    layout["slot_labels"],
                    package.slots_by_earth[earth_name],
                    earth_score,
                    earth_name,
                    int(layout["layout_number"]),
                )
            )

    all_results.sort(key=lambda item: item.total_score, reverse=True)
    if not all_results:
        raise RuntimeError("No slot-matching results were produced.")
    best_earth = all_results[0].shifting_earth
    _best_observations, best_align = all_observation_sets[best_earth]
    ctx.log(
        f"Best earth selected for slot view: {best_earth} (dx_norm={best_align['dx_norm']:.4f}, dy_norm={best_align['dy_norm']:.4f})"
    )

    results: List[RankedResult] = []
    for item in all_results:
        results.append(
            RankedResult(
                number=int(item.layout_number),
                filename=f"layout_{int(item.layout_number):03d}",
                score=float(item.total_score),
                details={
                    "shifting_earth": item.shifting_earth,
                    "earth_score": float(item.earth_score),
                    "exact_match_weight": float(item.details.get("exact_match_weight", 0.0)),
                    "hard_mismatch_weight": float(item.details.get("hard_mismatch_weight", 0.0)),
                    "unknown_weight": float(item.details.get("unknown_weight", 0.0)),
                },
            )
        )

    ctx.log("")
    ctx.log(f"Top {min(TOP_RESULTS_TO_LOG, len(results))} matches:")
    for i, r in enumerate(results[:TOP_RESULTS_TO_LOG], start=1):
        ctx.log(
            f"{i:>2}. #{r.number:<3} earth={r.details['shifting_earth']:<14} score={r.score:.6f} "
            f"exactW={r.details['exact_match_weight']:.2f} mismW={r.details['hard_mismatch_weight']:.2f} unkW={r.details['unknown_weight']:.2f}"
        )

    t = step_start(ctx, "Resolve top image paths")
    top_paths = resolve_display_paths(results, ctx, disp_dir, dynamic_render=dynamic_render)
    step_end(ctx, "Resolve top image paths", t)

    ctx.log(f"Crop box: x0={crop_box.x0}, y0={crop_box.y0}, x1={crop_box.x1}, y1={crop_box.y1}")
    ctx.log(f"Total execution time: {fmt_seconds(time.perf_counter() - total_start)}")
    return {
        "results": results,
        "top_paths": top_paths,
        "cropped_panel": cropped_panel,
        "earth_ranking": [{"name": n, "score": s, "details": d} for n, s, d in earth_rank],
        "candidate_earths": candidate_earths,
        "crop_method": crop_method,
        "crop_refinement": crop_refinement_info,
    }


# ---------- Tuned backend override layer ----------
# This block intentionally overrides the earlier proto-loader and run_recognition
# so the GUI uses the tuned slot-based backend and tuned config JSON.

TUNED_BACKEND_FILENAMES = [
    "recognize_nightreign_map_proto_v12_debugplus_radiusscale_fix.py",
    "recognize_nightreign_map_proto_v12_authoritative.py",
]
TUNED_CONFIG_FILENAMES = [
    "nightreign_best_tuned_config.json",
]

_TUNED_PROTO_BACKEND_MODULE = None
_TUNED_PROTO_BACKEND_BASE_STATE = None
_TUNED_CONFIG_CACHE = None


def load_proto_backend_module(ctx: Optional[RunContext] = None):
    global _TUNED_PROTO_BACKEND_MODULE
    if _TUNED_PROTO_BACKEND_MODULE is not None:
        return _TUNED_PROTO_BACKEND_MODULE

    candidates: List[Path] = []
    for backend_name in TUNED_BACKEND_FILENAMES:
        candidates.extend([runtime_dir() / backend_name, resource_base_dir() / backend_name])

    backend_path = None
    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(cand)

    for cand in unique_candidates:
        if cand.exists():
            backend_path = cand
            break

    if backend_path is None:
        searched = "\n".join(str(p) for p in unique_candidates)
        raise FileNotFoundError(
            "Tuned proto backend not found. Looked in:\n" + searched
        )

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("nightreign_proto_backend_tuned_gui", backend_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for {backend_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _TUNED_PROTO_BACKEND_MODULE = module
        if ctx is not None:
            ctx.log(f"Loaded tuned proto backend: {backend_path.resolve()}")
        return module
    except Exception as exc:
        raise RuntimeError(f"Failed to load tuned proto backend from {backend_path}: {exc}") from exc


def find_tuned_config_path(ctx: Optional[RunContext] = None) -> Path:
    candidates: List[Path] = []
    for cfg_name in TUNED_CONFIG_FILENAMES:
        candidates.extend([runtime_dir() / cfg_name, resource_base_dir() / cfg_name])
    # also consider current working directory, because the tuner often writes there
    for cfg_name in TUNED_CONFIG_FILENAMES:
        candidates.append(Path.cwd() / cfg_name)

    seen = set()
    unique = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)

    for cand in unique:
        if cand.exists():
            if ctx is not None:
                ctx.log(f"Using tuned config path: {cand.resolve()}")
            return cand

    searched = "\n".join(str(p) for p in unique)
    raise FileNotFoundError(f"Tuned config JSON not found. Looked in:\n{searched}")


def load_tuned_config(ctx: Optional[RunContext] = None) -> dict:
    global _TUNED_CONFIG_CACHE
    if _TUNED_CONFIG_CACHE is not None:
        return _TUNED_CONFIG_CACHE
    cfg_path = find_tuned_config_path(ctx)
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read tuned config JSON from {cfg_path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Tuned config at {cfg_path} must be a JSON object.")
    _TUNED_CONFIG_CACHE = cfg
    return cfg


def _clone_type_config(cfg: dict) -> dict:
    return {k: dict(v) for k, v in cfg.items()}


def _capture_backend_base_state(module) -> dict:
    return {
        "TYPE_CONFIG": _clone_type_config(module.TYPE_CONFIG),
        "DEFAULT_TEMPLATE_SCALES": tuple(module.DEFAULT_TEMPLATE_SCALES),
        "CROP_REFINE_MARGIN_WEIGHT": float(module.CROP_REFINE_MARGIN_WEIGHT),
        "CROP_REFINE_BLACK_WEIGHT": float(module.CROP_REFINE_BLACK_WEIGHT),
        "CROP_REFINE_EDGE_WEIGHT": float(module.CROP_REFINE_EDGE_WEIGHT),
        "EARTH_SECONDARY_THRESHOLD": float(module.EARTH_SECONDARY_THRESHOLD),
        "classify_shifting_earth": module.classify_shifting_earth,
    }


def _restore_backend_base_state(module, base_state: dict) -> None:
    module.TYPE_CONFIG = _clone_type_config(base_state["TYPE_CONFIG"])
    module.DEFAULT_TEMPLATE_SCALES = tuple(base_state["DEFAULT_TEMPLATE_SCALES"])
    module.CROP_REFINE_MARGIN_WEIGHT = float(base_state["CROP_REFINE_MARGIN_WEIGHT"])
    module.CROP_REFINE_BLACK_WEIGHT = float(base_state["CROP_REFINE_BLACK_WEIGHT"])
    module.CROP_REFINE_EDGE_WEIGHT = float(base_state["CROP_REFINE_EDGE_WEIGHT"])
    module.EARTH_SECONDARY_THRESHOLD = float(base_state["EARTH_SECONDARY_THRESHOLD"])
    module.classify_shifting_earth = base_state["classify_shifting_earth"]


def _install_earth_classifier(module, low_weight, grad_weight) -> None:
    if low_weight is None and grad_weight is None:
        return
    lw = 0.65 if low_weight is None else float(low_weight)
    gw = 0.35 if grad_weight is None else float(grad_weight)
    total = abs(lw) + abs(gw)
    if total <= 1e-8:
        lw, gw = 0.65, 0.35
    else:
        lw, gw = lw / total, gw / total

    def classify_shifting_earth(panel_bgr, package):
        shot_low = module.preprocess_for_similarity(panel_bgr)
        shot_grad = module.preprocess_for_gradients(panel_bgr)
        shot_low_n = module.normalize(shot_low)
        shot_grad_n = module.normalize(shot_grad)
        ranked = []
        for idx, earth_name in enumerate(package.earth_names):
            low_corr = module.corr(shot_low_n, module.normalize(package.terrain_low[idx]))
            grad_corr = module.corr(shot_grad_n, module.normalize(package.terrain_grad[idx]))
            score = float(lw) * low_corr + float(gw) * grad_corr
            ranked.append((earth_name, score, {"low_corr": low_corr, "grad_corr": grad_corr}))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    module.classify_shifting_earth = classify_shifting_earth


def apply_config_to_module(module, cfg: dict) -> None:
    global _TUNED_PROTO_BACKEND_BASE_STATE
    if _TUNED_PROTO_BACKEND_BASE_STATE is None:
        _TUNED_PROTO_BACKEND_BASE_STATE = _capture_backend_base_state(module)
    _restore_backend_base_state(module, _TUNED_PROTO_BACKEND_BASE_STATE)

    if "template_scales" in cfg and cfg["template_scales"]:
        module.DEFAULT_TEMPLATE_SCALES = tuple(float(x) for x in cfg["template_scales"])

    if "crop_refine_margin_weight" in cfg:
        module.CROP_REFINE_MARGIN_WEIGHT = float(cfg["crop_refine_margin_weight"])
    if "crop_refine_black_weight" in cfg:
        module.CROP_REFINE_BLACK_WEIGHT = float(cfg["crop_refine_black_weight"])
    if "crop_refine_edge_weight" in cfg:
        module.CROP_REFINE_EDGE_WEIGHT = float(cfg["crop_refine_edge_weight"])
    if "earth_secondary_threshold" in cfg:
        module.EARTH_SECONDARY_THRESHOLD = float(cfg["earth_secondary_threshold"])

    type_overrides = cfg.get("type_overrides", {})
    if isinstance(type_overrides, dict):
        for type_name, overrides in type_overrides.items():
            if type_name not in module.TYPE_CONFIG or not isinstance(overrides, dict):
                continue
            for key, value in overrides.items():
                module.TYPE_CONFIG[type_name][key] = float(value)

    _install_earth_classifier(
        module,
        cfg.get("earth_low_weight"),
        cfg.get("earth_grad_weight"),
    )


def run_recognition(
    ctx: RunContext,
    monitor_index: int = 1,
    preview_callback: Optional[Callable[[np.ndarray], None]] = None,
    pre_captured_full_img: Optional[np.ndarray] = None,
    second_full_img: Optional[np.ndarray] = None,
    dual_shot_enabled: bool = False,
    dynamic_render: bool = True,
) -> dict:
    import argparse
    import tempfile

    total_start = time.perf_counter()

    disp_dir = choose_existing_display_dir()
    if disp_dir is None:
        checked = "\n".join(f"- {p}" for p in display_search_paths())
        raise FileNotFoundError(
            "Display directory not found. Checked these locations:\n" + checked
        )

    ctx.log(f"Display directory: {disp_dir.resolve()}")
    ctx.log(f"Log file: {log_file_path().resolve()}")

    selected_monitor = None
    for idx, monitor in get_available_monitors():
        if idx == monitor_index:
            selected_monitor = monitor
            break
    if selected_monitor is None:
        raise RuntimeError(f"Selected display {monitor_index} is no longer available.")
    ctx.log(f"Selected display: {monitor_label(monitor_index, selected_monitor)}")

    backend = load_proto_backend_module(ctx)
    tuned_cfg = load_tuned_config(ctx)
    apply_config_to_module(backend, tuned_cfg)
    package_path = find_slot_package_path(ctx)

    search_radius_scale = float(tuned_cfg.get("search_radius_scale", 1.0))
    ctx.log(f"Using tuned search_radius_scale={search_radius_scale:.3f}")
    ctx.log("Applying tuned runtime parameters to backend.")

    # Redirect backend logging into the GUI log pane.
    backend.log = lambda msg: ctx.log(str(msg))

    with tempfile.TemporaryDirectory(prefix="nightreign_gui_tuned_") as td:
        td_path = Path(td)
        crop_out = td_path / "recognition_panel_used.png"
        detections_out = td_path / "recognition_panel_used_with_slot_labels.png"
        detections_search_out = td_path / "recognition_panel_used_with_slot_labels_search.png"

        screenshot_path = td_path / "captured_full_screenshot.png"

        # Capture once here so the GUI preview and the backend recognition use the exact same image.
        if pre_captured_full_img is None:
            t_capture = step_start(ctx, "Capture screenshot for preview")
            full_img = backend.capture_monitor_bgr(monitor_index)
            step_end(ctx, "Capture screenshot for preview", t_capture)
        else:
            full_img = pre_captured_full_img.copy()
            ctx.log("Using pre-captured first screenshot from dual-shot trigger.")
        cv2.imwrite(str(screenshot_path), full_img)

        # Build the GUI preview from the initial crop, before later normalization/refinement.
        preview_panel = None
        try:
            crop_circle, crop_method = backend.detect_map_circle(
                full_img,
                backend.DEFAULT_HOUGH_REGION,
                backend.DEFAULT_HOUGH_SCALE,
                ctx,
            )
            if crop_method == "blue_ring":
                preview_box = backend.clip_box(
                    backend.estimate_panel_box(crop_circle, backend.DEFAULT_PANEL_SCALE),
                    full_img.shape[1],
                    full_img.shape[0],
                )
                preview_panel = full_img[preview_box.y0:preview_box.y1, preview_box.x0:preview_box.x1].copy()
            else:
                preview_box = backend.refine_panel_box(
                    full_img,
                    crop_circle,
                    backend.DEFAULT_PANEL_SCALE,
                    backend.DEFAULT_SEARCH_RADIUS,
                    backend.DEFAULT_PAD,
                    ctx,
                )
                preview_panel = full_img[preview_box.y0:preview_box.y1, preview_box.x0:preview_box.x1].copy()
        except Exception as exc:
            ctx.log(f"Warning: failed to build initial crop preview: {exc}")
            preview_panel = None

        if preview_callback is not None and preview_panel is not None:
            try:
                preview_callback(preview_panel.copy())
                ctx.log("Displayed captured map preview (initial crop).")
            except Exception as exc:
                ctx.log(f"Warning: failed to show preview image: {exc}")

        ns = argparse.Namespace(
            package=package_path,
            image=screenshot_path,
            panel_image=None,
            monitor=None,
            top=max(TOP_RESULTS_TO_LOG, THUMB_COUNT, 20),
            debug=True,
            crop_out=crop_out,
            detections_out=detections_out,
            detections_search_out=detections_search_out,
            search_radius_scale=search_radius_scale,
            json_out=None,
        )

        t = step_start(ctx, "Run tuned slot recognizer")
        payload = backend.run_recognition(ns)
        step_end(ctx, "Run tuned slot recognizer", t)

        results_payload = payload.get("results", [])
        if results_payload:
            ctx.log("")
            ctx.log(f"Top {min(len(results_payload), TOP_RESULTS_TO_LOG)} layout matches:")
            for idx_result, item in enumerate(results_payload[:TOP_RESULTS_TO_LOG], start=1):
                details = item.get("details", {}) if isinstance(item, dict) else {}
                exact_w = details.get("exact_match_weight")
                mism_w = details.get("hard_mismatch_weight")
                unk_w = details.get("unknown_weight")
                parts = [
                    f"{idx_result}. layout #{int(item['layout_number'])}",
                    f"earth={item.get('shifting_earth', '?')}",
                    f"score={float(item.get('total_score', 0.0)):.6f}",
                ]
                if "earth_score" in item:
                    try:
                        parts.append(f"earth_score={float(item.get('earth_score', 0.0)):.6f}")
                    except Exception:
                        pass
                if exact_w is not None:
                    try:
                        parts.append(f"exactW={float(exact_w):.2f}")
                    except Exception:
                        pass
                if mism_w is not None:
                    try:
                        parts.append(f"mismW={float(mism_w):.2f}")
                    except Exception:
                        pass
                if unk_w is not None:
                    try:
                        parts.append(f"unkW={float(unk_w):.2f}")
                    except Exception:
                        pass
                ctx.log("  " + " ".join(parts))
            ctx.log("")

        cropped_panel = preview_panel
        if cropped_panel is None and crop_out.exists():
            try:
                cropped_panel = cv2.imread(str(crop_out), cv2.IMREAD_COLOR)
            except Exception:
                cropped_panel = None

        ranked_results: List[RankedResult] = []
        for item in payload.get("results", [])[:]:
            ranked_results.append(
                RankedResult(
                    number=int(item["layout_number"]),
                    filename=f"{int(item['layout_number'])}",
                    score=float(item["total_score"]),
                    details=item,
                )
            )

        ordered_numbers = [r.number for r in ranked_results]
        top_paths = resolve_display_paths(ranked_results, ctx, disp_dir, dynamic_render=dynamic_render)

        crop_box = payload.get("crop_box")
        if isinstance(crop_box, dict):
            try:
                crop_box_obj = CropBox(
                    x0=int(crop_box.get("x0", 0)),
                    y0=int(crop_box.get("y0", 0)),
                    x1=int(crop_box.get("x1", 0)),
                    y1=int(crop_box.get("y1", 0)),
                )
            except Exception:
                crop_box_obj = None
        else:
            crop_box_obj = None

        ctx.log(f"Resolved {len(top_paths)} display image(s) for preview/result view.")
        ctx.log(f"GUI total execution time: {fmt_seconds(time.perf_counter() - total_start)}")

        return {
            "results": ranked_results,
            "ordered_numbers": ordered_numbers,
            "crop_box": crop_box_obj,
            "top_paths": top_paths,
            "cropped_panel": cropped_panel,
            "backend_payload": payload,
            "tuned_config": tuned_cfg,
        }



# ---------- AI slot-model GUI override layer ----------
# This block overrides the tuned slot matcher with a trained slot-patch model while
# preserving the tuned backend for crop detection, crop refinement, and earth ranking.

import torch
from torch import nn

TUNED_CONFIG_FILENAMES = [
    "nightreign_best_tuned_config_multi.json",
    "nightreign_best_tuned_config.json",
]
AI_MODEL_FILENAMES = [
    "slot_ai_test_output_dlc_no_text_nobalance/slot_patch_model_v1.pt",
    "slot_ai_test_output_dlc_no_text/slot_patch_model_v1.pt",
    "slot_patch_model_v1.pt",
    "slot_ai_test_output/slot_patch_model_v1.pt",
]
USE_EARTH_SELECTOR_V2_BY_DEFAULT = os.environ.get("NR_USE_EARTH_SELECTOR_V2", "1") != "0"
USE_EARTH_CASCADE_SELECTION = (
    os.environ.get("NR_ENABLE_EARTH_CASCADE", "0") == "1"
    or os.environ.get("NR_USE_EARTH_CASCADE_SELECTION", "0") == "1"
)
EARTH_SELECTOR_V2_SCRIPT_FILENAMES = [
    "Train_Earth_Selector_v2/infer_earth_selector_v2.py",
    "infer_earth_selector_v2.py",
]
EARTH_SELECTOR_V2_MODEL_FILENAMES = [
    "Train_Earth_Selector_v2/earth_selector_v2_model_all520_plus_new.pkl",
    "earth_selector_v2_model_all520_plus_new.pkl",
    "Train_Earth_Selector_v2/earth_selector_v2_model.pkl",
    "earth_selector_v2_model.pkl",
]
EARTH_MODEL_FILENAMES = [
    "earth_ai_test_output_no_text_prepped/earth_panel_model_v1.pt",
    "earth_ai_test_output_no_text/earth_panel_model_v1.pt",
    "earth_ai_test_output/earth_panel_model_v1.pt",
]
MAP_CLASS_MODEL_FILENAMES = [
    "map_class_model.pkl",
]
LAYOUT_INFER_CLASSIFIER_CONFIG = {
    "Great Hollow": {
        "classifier_dir": Path("tmp_great_hollow_both_crops_asym_L175_R170_T180_B180_Patched_multi_image") / "further_crop_asymmetric",
        "model_file": "layout_model.pkl",
        "input_mode": "great_hollow_asymmetric",
    },
    "Noklateo": {
        "classifier_dir": Path("tmp_noklateo_both_crops_L256_R256_T256_B256") / "further_crop_1024",
        "model_file": "layout_model.pkl",
        "input_mode": "further_crop_1024",
    },
    "The Crater": {
        "classifier_dir": Path("tmp_the_crater_both_crops_L256_R256_T256_B256") / "further_crop_1024",
        "model_file": "layout_model.pkl",
        "input_mode": "further_crop_1024",
    },
    "Mountaintop": {
        "classifier_dir": Path("tmp_mountaintop_both_crops_L256_R256_T256_B256") / "further_crop_1024",
        "model_file": "layout_model.pkl",
        "input_mode": "further_crop_1024",
    },
    "Rotted Woods": {
        "classifier_dir": Path("tmp_rotted_woods_both_crops_L256_R256_T256_B256") / "further_crop_1024",
        "model_file": "layout_model.pkl",
        "input_mode": "further_crop_1024",
    },
    "Limveld": {
        "classifier_dir": Path("tmp_limveld_both_crops_L256_R256_T256_B256") / "further_crop_1024",
        "model_file": "layout_model.pkl",
        "input_mode": "further_crop_1024",
    },
}
LAYOUT_INFER_PRIOR_STRONG_BONUS = 2.00
LAYOUT_INFER_PRIOR_CONFIDENCE_SCALE = 1.00
LAYOUT_INFER_PRIOR_MARGIN_SCALE = 0.50
LAYOUT_INFER_PRIOR_TOP5_SCALE = 0.25
LAYOUT_INFER_PRIOR_UNKNOWN_TOP5_SCALE = 0.20
USE_LEGACY_EARTH_SCORER_BY_DEFAULT = os.environ.get("NR_USE_LEARNED_EARTH_CLASSIFIER", "1") == "0"

# PatternId whitelist filtering (disabled by default)
ENABLE_PATTERN_ID_WHITELIST_FILTER = True
ALLOWED_PATTERN_IDS = {
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
    1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049,
    1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095,
    1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111,
    1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127,
    1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139,
}


class SmallSlotCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class EarthPanelCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def compute_map_class_descriptor(panel_bgr: np.ndarray, thumb_size: int = 24, hist_h_bins: int = 12, hist_s_bins: int = 4, crop_frac: float = 0.02) -> np.ndarray:
    work = preprocess_earth_selector_panel(panel_bgr)
    work = cv2.resize(work, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    small = cv2.resize(gray, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    small = (small - small.mean()) / (small.std() + 1e-6)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge = cv2.resize(mag, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA).reshape(-1)
    edge = (edge - edge.mean()) / (edge.std() + 1e-6)

    hist = cv2.calcHist([hsv], [0, 1], None, [hist_h_bins, hist_s_bins], [0, 180, 0, 256])
    hist = hist.reshape(-1).astype(np.float32)
    hist /= hist.sum() + 1e-6

    descriptor = np.concatenate([
        0.55 * small,
        0.30 * edge,
        3.00 * hist,
    ]).astype(np.float32)
    descriptor /= np.linalg.norm(descriptor) + 1e-6
    return descriptor


def preprocess_earth_selector_panel(
    panel_bgr: np.ndarray,
    input_size: int = EARTH_SELECTOR_INPUT_SIZE,
    border_crop_px: int = EARTH_SELECTOR_BORDER_CROP_PX,
) -> np.ndarray:
    """Step 1 canonical preprocessing helper: 1536 -> crop 256 per side -> 1024."""
    if panel_bgr is None or panel_bgr.size == 0:
        raise ValueError("Panel image is empty.")
    resized = cv2.resize(panel_bgr, (int(input_size), int(input_size)), interpolation=cv2.INTER_AREA)
    border = int(border_crop_px)
    if border <= 0:
        return resized
    if border * 2 >= resized.shape[0] or border * 2 >= resized.shape[1]:
        raise ValueError("Invalid border crop for earth-selector preprocessing.")
    return resized[border : resized.shape[0] - border, border : resized.shape[1] - border]


def load_map_class_model(ctx: Optional[RunContext] = None) -> Optional[dict]:
    for name in MAP_CLASS_MODEL_FILENAMES:
        model_path = Path(name)
        if not model_path.exists():
            continue
        with model_path.open("rb") as f:
            model = pickle.load(f)
        model["descriptors"] = np.asarray(model["descriptors"], dtype=np.float32)
        if ctx is not None:
            ctx.log(f"Loaded map class model: {model_path.resolve()}")
        return model
    return None


def classify_shifting_earth_map_class(panel_bgr: np.ndarray, model: dict) -> List[Tuple[str, float, Dict[str, float]]]:
    descriptor = compute_map_class_descriptor(panel_bgr)
    descs = model["descriptors"]
    labels = model["labels"]
    origins = model["origins"]
    sims = descs @ descriptor.astype(np.float32)
    k = min(int(model.get("top_k", 5)), len(sims))
    top_idx = np.argsort(sims)[-k:][::-1]
    class_scores = defaultdict(float)
    neighbors = []
    for idx in top_idx:
        label = labels[idx]
        score = float(sims[idx])
        class_scores[label] += score
        neighbors.append({"label": label, "score": score, "origin": origins[idx]})
    ranked = sorted(class_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(label, float(score), {"neighbors": neighbors, "class_scores": ranked}) for label, score in ranked]


_MAP_CLASSIFIER_MODULE = None
_MAP_CLASSIFIER_MODEL = None
_LAYOUT_INFER_BUNDLE_CACHE: dict = {}
_LAYOUT_INFER_FAILED_EARTHS: set = set()


def load_map_classifier_bundle(ctx: Optional[RunContext] = None):
    global _MAP_CLASSIFIER_MODULE, _MAP_CLASSIFIER_MODEL
    if _MAP_CLASSIFIER_MODULE is None:
        script_path = Path(__file__).resolve().parent / "map_class_classifier.py"
        if script_path.exists():
            spec = importlib.util.spec_from_file_location("nightreign_map_class_classifier", script_path)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                _MAP_CLASSIFIER_MODULE = mod
    if _MAP_CLASSIFIER_MODULE is None:
        return None
    if _MAP_CLASSIFIER_MODEL is None:
        model_path = Path(__file__).resolve().parent / "map_class_model.pkl"
        if not model_path.exists():
            return None
        _MAP_CLASSIFIER_MODEL = _MAP_CLASSIFIER_MODULE.load_model(str(model_path))
        if ctx is not None:
            ctx.log(f"Loaded map class model: {model_path.resolve()}")
    return _MAP_CLASSIFIER_MODULE, _MAP_CLASSIFIER_MODEL


def classify_shifting_earth_map_classifier(raw_panel_bgr: np.ndarray, package: SlotRecognitionPackage, ctx: Optional[RunContext] = None) -> List[Tuple[str, float, Dict[str, float]]]:
    bundle = load_map_classifier_bundle(ctx)
    if bundle is None:
        return []
    module, model = bundle
    descriptor = module.compute_descriptor(raw_panel_bgr)
    result = module.classify_descriptor(model, descriptor)
    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for item in result.get("class_scores", []):
        earth_name = str(item["label"])
        if earth_name not in package.layouts_by_earth:
            continue
        ranked.append((earth_name, float(item["score"]), {"source": "map_class_classifier", "neighbors": result.get("neighbors", [])}))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if ctx is not None and ranked:
        ctx.log("Using map_class_classifier.py as main Shifting Earth selector.")
    return ranked


def _layout_infer_module_name_for_earth(earth_name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", str(earth_name).strip().lower().replace(" ", "_"))
    return f"nightreign_layout_classifier_lib_{slug}"


def load_layout_infer_bundle_for_earth(earth_name: str, ctx: Optional[RunContext] = None) -> Optional[dict]:
    key = str(earth_name)
    if key in _LAYOUT_INFER_BUNDLE_CACHE:
        return _LAYOUT_INFER_BUNDLE_CACHE[key]
    if key in _LAYOUT_INFER_FAILED_EARTHS:
        return None

    cfg = LAYOUT_INFER_CLASSIFIER_CONFIG.get(key)
    if not cfg:
        _LAYOUT_INFER_FAILED_EARTHS.add(key)
        return None

    classifier_dir = (Path(__file__).resolve().parent / cfg["classifier_dir"]).resolve()
    lib_path = classifier_dir / "layout_classifier_lib.py"
    model_path = classifier_dir / str(cfg.get("model_file", "layout_model.pkl"))
    if not lib_path.exists() or not model_path.exists():
        if ctx is not None:
            ctx.log(
                f"Layout infer classifier assets missing for {key}: "
                f"lib={lib_path.exists()} model={model_path.exists()}"
            )
        _LAYOUT_INFER_FAILED_EARTHS.add(key)
        return None

    module_name = _layout_infer_module_name_for_earth(key)
    try:
        if module_name in sys.modules:
            lib_mod = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, lib_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load spec from {lib_path}")
            lib_mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = lib_mod
            spec.loader.exec_module(lib_mod)
        model = lib_mod.load_pickle(str(model_path))
    except Exception as exc:
        if ctx is not None:
            ctx.log(f"Failed to load layout infer classifier for {key}: {exc}")
        _LAYOUT_INFER_FAILED_EARTHS.add(key)
        return None

    bundle = {
        "earth_name": key,
        "classifier_dir": str(classifier_dir),
        "model_path": str(model_path),
        "input_mode": str(cfg.get("input_mode", "further_crop_1024")),
        "module": lib_mod,
        "model": model,
    }
    _LAYOUT_INFER_BUNDLE_CACHE[key] = bundle
    if ctx is not None:
        ctx.log(f"Loaded layout infer classifier for {key}: {model_path}")
    return bundle


def preprocess_panel_for_layout_infer(raw_panel_bgr: np.ndarray, earth_name: str) -> np.ndarray:
    if str(earth_name) == "Great Hollow":
        return preprocess_panel_for_status_recognizer_great_hollow(raw_panel_bgr)
    return preprocess_earth_selector_panel(
        raw_panel_bgr,
        input_size=EARTH_SELECTOR_INPUT_SIZE,
        border_crop_px=EARTH_SELECTOR_BORDER_CROP_PX,
    )


def run_layout_infer_classifier_for_earth(
    raw_panel_bgr: np.ndarray,
    earth_name: str,
    ctx: Optional[RunContext] = None,
) -> Optional[dict]:
    if ctx is not None:
        ctx.log(f"[DEBUG] run_layout_infer_classifier_for_earth called for {earth_name}. Filter enabled: {ENABLE_PATTERN_ID_WHITELIST_FILTER}")
    
    bundle = load_layout_infer_bundle_for_earth(earth_name, ctx)
    if bundle is None:
        return None

    panel_for_classifier = preprocess_panel_for_layout_infer(raw_panel_bgr, earth_name)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
        if not cv2.imwrite(str(temp_path), panel_for_classifier):
            raise RuntimeError(f"Failed to write temp panel image to {temp_path}")
        started = time.perf_counter()
        result = bundle["module"].classify_image(str(temp_path), bundle["model"])
        wall_latency_ms = (time.perf_counter() - started) * 1000.0
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    top5 = result.get("top5", [])
    norm_top5 = []
    filter_active = ENABLE_PATTERN_ID_WHITELIST_FILTER
    if filter_active and ctx is not None:
        ctx.log(f"PatternId whitelist filter ACTIVE. Allowed IDs: {len(ALLOWED_PATTERN_IDS)} total")
    
    for entry in top5:
        try:
            pattern_id = int(entry.get("patternId", -1))
            original_pid = pattern_id
            # Apply whitelist filtering if enabled
            if ENABLE_PATTERN_ID_WHITELIST_FILTER and pattern_id not in ALLOWED_PATTERN_IDS:
                pattern_id = 0
                if ctx is not None:
                    ctx.log(f"  Redirecting patternId {original_pid} -> 0 (not in whitelist)")
            norm_top5.append({
                "patternId": pattern_id,
                "probability": float(entry.get("probability", 0.0)),
                "score": float(entry.get("score", 0.0)),
            })
        except Exception:
            continue

    decided_pid = result.get("decided_patternId", None)
    if decided_pid is not None:
        try:
            decided_pid = int(decided_pid)
            # Apply whitelist filtering to decided_patternId if enabled
            if ENABLE_PATTERN_ID_WHITELIST_FILTER and decided_pid not in ALLOWED_PATTERN_IDS:
                decided_pid = 0
                if ctx is not None:
                    ctx.log(f"  Redirecting decided_patternId to 0 (not in whitelist)")
        except Exception:
            decided_pid = None

    best_pid = result.get("best_patternId", None)
    if best_pid is not None:
        try:
            best_pid = int(best_pid)
            # Apply whitelist filtering to best_patternId if enabled
            if ENABLE_PATTERN_ID_WHITELIST_FILTER and best_pid not in ALLOWED_PATTERN_IDS:
                best_pid = 0
                if ctx is not None:
                    ctx.log(f"  Redirecting best_patternId to 0 (not in whitelist)")
        except Exception:
            best_pid = None

    out = {
        "earth_name": str(earth_name),
        "classifier_dir": str(bundle.get("classifier_dir", "")),
        "model_path": str(bundle.get("model_path", "")),
        "input_mode": str(bundle.get("input_mode", "further_crop_1024")),
        "input_shape": [int(panel_for_classifier.shape[1]), int(panel_for_classifier.shape[0])],
        "best_patternId": best_pid,
        "decided_patternId": decided_pid,
        "unknown": bool(result.get("unknown", False)),
        "confidence": float(result.get("confidence", 0.0)),
        "margin": float(result.get("margin", 0.0)),
        "latency_ms": float(result.get("latency_ms", 0.0)),
        "pipeline_latency_ms": float(wall_latency_ms),
        "top5": norm_top5,
    }
    if ctx is not None:
        ctx.log(f"[DEBUG] Returning result for {earth_name}: best={best_pid}, decided={decided_pid}, top5_count={len(norm_top5)}")
        if norm_top5:
            ctx.log(f"[DEBUG] Top5 patternIds: {[e.get('patternId') for e in norm_top5]}")
    return out


def compute_layout_infer_prior(layout_number: int, infer_result: Optional[dict]) -> float:
    if infer_result is None:
        return 0.0

    probs_by_pattern = {}
    for entry in infer_result.get("top5", []) or []:
        try:
            pid = int(entry.get("patternId", -1))
            prob = float(entry.get("probability", 0.0))
        except Exception:
            continue
        if pid >= 0:
            probs_by_pattern[pid] = max(0.0, prob)

    decided_pid = infer_result.get("decided_patternId", None)
    if decided_pid is not None:
        try:
            decided_pid = int(decided_pid)
        except Exception:
            decided_pid = None

    confidence = max(0.0, min(1.0, float(infer_result.get("confidence", 0.0))))
    margin = max(0.0, min(1.0, float(infer_result.get("margin", 0.0))))
    if decided_pid is not None:
        if int(layout_number) == int(decided_pid):
            return float(
                LAYOUT_INFER_PRIOR_STRONG_BONUS
                + (LAYOUT_INFER_PRIOR_CONFIDENCE_SCALE * confidence)
                + (LAYOUT_INFER_PRIOR_MARGIN_SCALE * margin)
            )
        return float(LAYOUT_INFER_PRIOR_TOP5_SCALE * probs_by_pattern.get(int(layout_number), 0.0))

    if bool(infer_result.get("unknown", False)):
        return float(LAYOUT_INFER_PRIOR_UNKNOWN_TOP5_SCALE * probs_by_pattern.get(int(layout_number), 0.0))
    return float(LAYOUT_INFER_PRIOR_TOP5_SCALE * probs_by_pattern.get(int(layout_number), 0.0))


class EvergaolPatchCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def prepare_panel_for_earth_ai(panel_bgr: np.ndarray, panel_size: int) -> np.ndarray:
    side = min(panel_bgr.shape[:2])
    if panel_bgr.shape[0] != panel_bgr.shape[1]:
        y0 = max(0, (panel_bgr.shape[0] - side) // 2)
        x0 = max(0, (panel_bgr.shape[1] - side) // 2)
        panel_bgr = panel_bgr[y0:y0 + side, x0:x0 + side].copy()
    else:
        panel_bgr = panel_bgr.copy()

    side = panel_bgr.shape[0]
    cy = (side - 1) / 2.0
    cx = (side - 1) / 2.0
    yy, xx = np.ogrid[:side, :side]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outer_radius = side * 0.47
    outside_mask = dist >= outer_radius

    neutral = np.full_like(panel_bgr, 168)
    panel_bgr[outside_mask] = neutral[outside_mask]

    hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([85, 70, 60], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8))
    gold_mask = cv2.inRange(hsv, np.array([8, 80, 90], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 210], dtype=np.uint8), np.array([179, 50, 255], dtype=np.uint8))
    overlay_mask = cv2.bitwise_or(blue_mask, gold_mask)
    overlay_mask = cv2.bitwise_or(overlay_mask, white_mask)
    overlay_mask[outside_mask] = 0
    overlay_mask = cv2.dilate(overlay_mask, np.ones((5, 5), np.uint8), iterations=1)
    if int(np.count_nonzero(overlay_mask)) > 0:
        panel_bgr = cv2.inpaint(panel_bgr, overlay_mask, 3, cv2.INPAINT_TELEA)

    if panel_bgr.shape[0] != panel_size or panel_bgr.shape[1] != panel_size:
        panel_bgr = cv2.resize(panel_bgr, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
    return panel_bgr


_AI_MODEL_CACHE = None
_EARTH_MODEL_CACHE = None
_AI_DEVICE_CACHE = None
_EVERGAOL_MODEL_CACHE = None
_EARTH_SELECTOR_V2_BUNDLE_CACHE = None


def choose_torch_device() -> torch.device:
    global _AI_DEVICE_CACHE
    if _AI_DEVICE_CACHE is None:
        _AI_DEVICE_CACHE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _AI_DEVICE_CACHE


def find_ai_model_path(ctx: Optional[RunContext] = None) -> Path:
    candidates: List[Path] = []
    base_resource = resource_base_dir()
    runtime_base = runtime_dir()
    runtime_parent = runtime_base.parent if runtime_base.parent.exists() else runtime_base
    resource_parent = base_resource.parent if base_resource.parent.exists() else base_resource
    for name in AI_MODEL_FILENAMES:
        candidates.extend([
            runtime_base / name,
            runtime_parent / name,
            base_resource / name,
            resource_parent / name,
            Path.cwd() / name,
        ])

    seen = set()
    unique = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)

    for cand in unique:
        if cand.exists():
            if ctx is not None:
                ctx.log(f"Using AI slot model path: {cand.resolve()}")
            return cand

    searched = "\n".join(str(p) for p in unique)
    raise FileNotFoundError(f"AI slot model not found. Looked in:\n{searched}")


def find_earth_model_path(ctx: Optional[RunContext] = None) -> Path:
    candidates: List[Path] = []
    for name in EARTH_MODEL_FILENAMES:
        candidates.extend([
            runtime_dir() / name,
            resource_base_dir() / name,
            Path.cwd() / name,
        ])

    seen = set()
    unique = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)

    for cand in unique:
        if cand.exists():
            if ctx is not None:
                ctx.log(f"Using AI earth model path: {cand.resolve()}")
            return cand

    searched = "\n".join(str(p) for p in unique)
    raise FileNotFoundError(f"AI earth model not found. Looked in:\n{searched}")


def _resolve_existing_path_candidates(names: Sequence[str]) -> List[Path]:
    candidates: List[Path] = []
    base_dir = Path(__file__).resolve().parent
    for name in names:
        name_path = Path(name)
        candidates.extend([
            runtime_dir() / name_path,
            resource_base_dir() / name_path,
            Path.cwd() / name_path,
            base_dir / name_path,
        ])
    seen = set()
    unique: List[Path] = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def find_earth_selector_v2_script_path(ctx: Optional[RunContext] = None) -> Path:
    candidates = _resolve_existing_path_candidates(EARTH_SELECTOR_V2_SCRIPT_FILENAMES)
    for cand in candidates:
        if cand.exists():
            if ctx is not None:
                ctx.log(f"Using earth_selector_v2 script path: {cand.resolve()}")
            return cand
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"earth_selector_v2 script not found. Looked in:\n{searched}")


def find_earth_selector_v2_model_path(ctx: Optional[RunContext] = None) -> Path:
    candidates = _resolve_existing_path_candidates(EARTH_SELECTOR_V2_MODEL_FILENAMES)
    for cand in candidates:
        if cand.exists():
            if ctx is not None:
                ctx.log(f"Using earth_selector_v2 model path: {cand.resolve()}")
            return cand
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"earth_selector_v2 model not found. Looked in:\n{searched}")


def load_earth_selector_v2_bundle(ctx: Optional[RunContext] = None) -> dict:
    global _EARTH_SELECTOR_V2_BUNDLE_CACHE
    if _EARTH_SELECTOR_V2_BUNDLE_CACHE is not None:
        return _EARTH_SELECTOR_V2_BUNDLE_CACHE

    script_path = find_earth_selector_v2_script_path(ctx)
    model_path = find_earth_selector_v2_model_path(ctx)
    module_name = "nightreign_earth_selector_v2_infer"
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed loading earth_selector_v2 module from {script_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

    with model_path.open("rb") as f:
        model = pickle.load(f)
    hog = mod.make_hog_descriptor()
    _EARTH_SELECTOR_V2_BUNDLE_CACHE = {
        "module": mod,
        "model": model,
        "hog": hog,
        "model_path": model_path,
        "script_path": script_path,
    }
    if ctx is not None:
        ctx.log(f"Loaded earth_selector_v2 model: {model_path.resolve()}")
    return _EARTH_SELECTOR_V2_BUNDLE_CACHE


def find_evergaol_model_path(ctx: Optional[RunContext] = None) -> Optional[Path]:
    candidates: List[Path] = []
    for name in EVERGAOL_MODEL_FILENAMES:
        candidates.extend([
            runtime_dir() / name,
            resource_base_dir() / name,
            Path.cwd() / name,
        ])
    seen = set()
    unique = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    for cand in unique:
        if cand.exists():
            if ctx is not None:
                ctx.log(f"Using AI Evergaol model path: {cand.resolve()}")
            return cand
    return None


def load_ai_model_bundle(ctx: Optional[RunContext] = None):
    global _AI_MODEL_CACHE
    if _AI_MODEL_CACHE is not None:
        return _AI_MODEL_CACHE

    model_path = find_ai_model_path(ctx)
    device = choose_torch_device()
    bundle = torch.load(model_path, map_location=device)
    labels = list(bundle["labels"])
    model = SmallSlotCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    patch_size = int(bundle.get("patch_size", 96))
    context_scale = float(bundle.get("context_scale", 1.8))
    min_half_frac = float(bundle.get("min_half_frac", 0.075))

    if ctx is not None:
        ctx.log(f"Loaded AI slot model: {model_path.resolve()}")
        ctx.log(f"AI device: {device}")
        ctx.log(f"AI model patch_size={patch_size} context_scale={context_scale} min_half_frac={min_half_frac}")

    _AI_MODEL_CACHE = {
        "model_path": model_path,
        "device": device,
        "labels": labels,
        "model": model,
        "patch_size": patch_size,
        "context_scale": context_scale,
        "min_half_frac": min_half_frac,
    }
    return _AI_MODEL_CACHE


def load_earth_model_bundle(ctx: Optional[RunContext] = None):
    global _EARTH_MODEL_CACHE
    if _EARTH_MODEL_CACHE is not None:
        return _EARTH_MODEL_CACHE

    model_path = find_earth_model_path(ctx)
    device = choose_torch_device()
    bundle = torch.load(model_path, map_location=device)
    earth_names = list(bundle["earth_names"])
    model = EarthPanelCNN(num_classes=len(earth_names)).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    panel_size = int(bundle.get("panel_size", 256))

    if ctx is not None:
        ctx.log(f"Loaded AI earth model: {model_path.resolve()}")
        ctx.log(f"AI earth panel_size={panel_size}")

    _EARTH_MODEL_CACHE = {
        "model_path": model_path,
        "device": device,
        "earth_names": earth_names,
        "model": model,
        "panel_size": panel_size,
    }
    return _EARTH_MODEL_CACHE


def load_evergaol_model_bundle(ctx: Optional[RunContext] = None):
    global _EVERGAOL_MODEL_CACHE
    if _EVERGAOL_MODEL_CACHE is not None:
        return _EVERGAOL_MODEL_CACHE
    model_path = find_evergaol_model_path(ctx)
    if model_path is None:
        _EVERGAOL_MODEL_CACHE = None
        return None
    device = choose_torch_device()
    bundle = torch.load(model_path, map_location=device)
    labels = list(bundle.get("labels", ["absent", "present"]))
    model = EvergaolPatchCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    _EVERGAOL_MODEL_CACHE = {
        "model_path": model_path,
        "device": device,
        "labels": labels,
        "model": model,
        "patch_size": int(bundle.get("patch_size", 96)),
        "context_scale": float(bundle.get("context_scale", 2.0)),
        "min_half_frac": float(bundle.get("min_half_frac", 0.070)),
    }
    if ctx is not None:
        ctx.log(f"Loaded AI Evergaol model: {model_path.resolve()}")
    return _EVERGAOL_MODEL_CACHE


@torch.no_grad()
def classify_shifting_earth_ai(raw_panel_bgr: np.ndarray, panel_bgr: np.ndarray, package: SlotRecognitionPackage, ctx: Optional[RunContext] = None) -> List[Tuple[str, float, Dict[str, float]]]:
    if USE_EARTH_SELECTOR_V2_BY_DEFAULT:
        try:
            v2_bundle = load_earth_selector_v2_bundle(ctx)
            panel_v2 = preprocess_earth_selector_panel(raw_panel_bgr)
            mod = v2_bundle["module"]
            model = v2_bundle["model"]
            feat = mod.extract_features_from_image(panel_v2, hog=v2_bundle["hog"]).reshape(1, -1).astype(np.float32)
            centroids = np.asarray(model["centroids"], dtype=np.float32)
            _, probs = mod.predict_proba_from_centroids(feat, centroids, temperature=float(model["temperature"]))
            probs_row = probs[0]
            labels = list(model["labels"])
            scored = mod.score_probability_row(
                probs_row,
                labels,
                float(model.get("unknown_margin_threshold", 0.0)),
                float(model.get("unknown_conf_threshold", 0.0)),
            )
            ranked: List[Tuple[str, float, Dict[str, float]]] = []
            for idx in np.argsort(probs_row)[::-1]:
                earth_name = str(labels[int(idx)])
                if earth_name not in package.layouts_by_earth:
                    continue
                score = float(probs_row[int(idx)])
                ranked.append((
                    earth_name,
                    score,
                    {
                        "source": "earth_selector_v2",
                        "ai_prob": score,
                        "unknown": bool(scored.get("unknown", False)),
                        "confidence": float(scored.get("confidence", 0.0)),
                        "margin_top1_top2": float(scored.get("margin_top1_top2", 0.0)),
                    },
                ))
            if ranked:
                if ctx is not None:
                    ctx.log("Using earth_selector_v2 model for Shifting Earth ranking.")
                return ranked
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            if ctx is not None:
                ctx.log(f"earth_selector_v2 unavailable; falling back ({e})")

    map_rank = classify_shifting_earth_map_classifier(raw_panel_bgr, package, ctx)
    if map_rank:
        return map_rank
    if USE_LEGACY_EARTH_SCORER_BY_DEFAULT:
        if ctx is not None:
            ctx.log("Using legacy terrain-correlation earth scorer for Shifting Earth ranking.")
        return classify_shifting_earth_slot(panel_bgr, package)
    try:
        earth_bundle = load_earth_model_bundle(ctx)
    except FileNotFoundError:
        return classify_shifting_earth_slot(panel_bgr, package)

    panel_size = int(earth_bundle["panel_size"])
    panel_rs = prepare_panel_for_earth_ai(panel_bgr, panel_size)
    panel_rgb = cv2.cvtColor(panel_rs, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(panel_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x = x.to(earth_bundle["device"])
    logits = earth_bundle["model"](x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for earth_name, prob in zip(earth_bundle["earth_names"], probs):
        if earth_name not in package.layouts_by_earth:
            continue
        score = float(prob)
        ranked.append((earth_name, score, {"low_corr": score, "grad_corr": 0.0, "ai_prob": score}))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if ctx is not None and ranked:
        ctx.log("Using AI earth classifier for Shifting Earth ranking.")
    return ranked


def reflect_pad_extract_ai(img_bgr: np.ndarray, cx: int, cy: int, half: int, patch_size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x0 = cx - half
    y0 = cy - half
    x1 = cx + half
    y1 = cy + half

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if any(v > 0 for v in (pad_left, pad_top, pad_right, pad_bottom)):
        img_bgr = cv2.copyMakeBorder(
            img_bgr,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    patch = img_bgr[y0:y1, x0:x1]
    if patch.shape[0] <= 0 or patch.shape[1] <= 0:
        raise ValueError("Empty AI patch extracted.")
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    return patch


def patch_to_tensor_rgb_generic(patch_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0


def infer_evergaol_present_prob(bundle: dict, patch_bgr: np.ndarray) -> float:
    x = patch_to_tensor_rgb_generic(patch_bgr, bundle["device"])
    with torch.no_grad():
        logits = bundle["model"](x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    labels = list(bundle["labels"])
    present_idx = labels.index("present") if "present" in labels else min(1, len(labels) - 1)
    return float(probs[present_idx])


def patch_to_tensor_ai(patch_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return x.to(device)


@torch.no_grad()
def infer_slot_probs_ai(model: nn.Module, labels: Sequence[str], patch_bgr: np.ndarray, device: torch.device) -> Dict[str, float]:
    x = patch_to_tensor_ai(patch_bgr, device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {label: float(prob) for label, prob in zip(labels, probs)}


def _gh_slot4_classifier_mode() -> str:
    raw = str(os.getenv("GH_SLOT4_CLASSIFIER_MODE", GH_SLOT4_CLASSIFIER_MODE_DEFAULT)).strip().lower()
    if raw not in {"classic", "prototype"}:
        return "classic"
    return raw


def _gh_slot4_empty_gate_threshold() -> float:
    raw = str(os.getenv("GH_SLOT4_EMPTY_GATE_THRESHOLD", str(GH_SLOT4_EMPTY_GATE_THRESHOLD_DEFAULT))).strip()
    try:
        value = float(raw)
    except Exception:
        value = GH_SLOT4_EMPTY_GATE_THRESHOLD_DEFAULT
    return max(0.0, min(1.0, value))


def _gh_slot4_find_prototype_model_path() -> Optional[Path]:
    candidates: List[Path] = []
    for p in GH_SLOT4_PROTOTYPE_MODEL_PATHS:
        candidates.extend([
            runtime_dir() / p,
            resource_base_dir() / p,
            Path.cwd() / p,
            p,
        ])
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def _gh_slot4_load_prototype_bundle() -> Optional[dict]:
    global GH_SLOT4_PROTOTYPE_CACHE
    if GH_SLOT4_PROTOTYPE_CACHE is False:
        return None
    if isinstance(GH_SLOT4_PROTOTYPE_CACHE, dict):
        return GH_SLOT4_PROTOTYPE_CACHE
    model_path = _gh_slot4_find_prototype_model_path()
    if model_path is None:
        GH_SLOT4_PROTOTYPE_CACHE = False
        return None
    try:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        cent_empty = np.asarray(payload["centroid_empty"], dtype=np.float32).reshape(-1)
        cent_pos = np.asarray(payload["centroid_pos"], dtype=np.float32).reshape(-1)
        patch_size = int(payload.get("patch_size", 96))
        temperature = float(payload.get("temperature", 0.1))
        GH_SLOT4_PROTOTYPE_CACHE = {
            "model_path": model_path,
            "slot_id": int(payload.get("slot_id", GH_SLOT4_TARGET_SLOT_ID)),
            "patch_size": patch_size,
            "temperature": max(1e-6, temperature),
            "centroid_empty": cent_empty,
            "centroid_pos": cent_pos,
        }
        return GH_SLOT4_PROTOTYPE_CACHE
    except Exception:
        GH_SLOT4_PROTOTYPE_CACHE = False
        return None


def _gh_slot4_infer_empty_prob(bundle: dict, patch_bgr: np.ndarray) -> Tuple[float, float, float]:
    patch_size = int(bundle["patch_size"])
    if patch_bgr.shape[0] != patch_size or patch_bgr.shape[1] != patch_size:
        patch = cv2.resize(patch_bgr, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    else:
        patch = patch_bgr
    x = patch.astype(np.float32).reshape(-1)
    cent_empty = bundle["centroid_empty"]
    cent_pos = bundle["centroid_pos"]
    de = float(np.mean((x - cent_empty) ** 2))
    dp = float(np.mean((x - cent_pos) ** 2))
    t = float(bundle["temperature"])
    a = -de / t
    b = -dp / t
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    p_empty = float(ea / (ea + eb + 1e-12))
    return p_empty, de, dp


def _gh_slot4_apply_prototype_override(raw_probs: Dict[str, float], slot: dict, earth_name: str, patch_bgr: np.ndarray) -> Tuple[Dict[str, float], str, Optional[dict]]:
    if str(earth_name) != "Great Hollow":
        return raw_probs, "classic", None
    if int(slot.get("slot_id", -1)) != GH_SLOT4_TARGET_SLOT_ID:
        return raw_probs, "classic", None
    if _gh_slot4_classifier_mode() != "prototype":
        return raw_probs, "classic", None
    bundle = _gh_slot4_load_prototype_bundle()
    if bundle is None:
        return raw_probs, "classic", None
    p_empty, d_empty, d_pos = _gh_slot4_infer_empty_prob(bundle, patch_bgr)
    p_occ = 1.0 - p_empty

    out = dict(raw_probs)
    out["empty"] = float(p_empty)
    pos_labels = [lbl for lbl in ("Church", "Sorcerer's Rise") if lbl in out]
    if not pos_labels:
        allowed = {str(x) for x in slot.get("allowed_types", [])}
        pos_labels = [lbl for lbl in ("Church", "Sorcerer's Rise") if lbl in allowed]
        for lbl in pos_labels:
            out.setdefault(lbl, 0.0)
    raw_pos_sum = float(sum(max(0.0, float(raw_probs.get(lbl, 0.0))) for lbl in pos_labels))
    if raw_pos_sum <= 1e-9:
        w = 1.0 / max(1, len(pos_labels))
        for lbl in pos_labels:
            out[lbl] = float(p_occ * w)
    else:
        for lbl in pos_labels:
            out[lbl] = float(p_occ * max(0.0, float(raw_probs.get(lbl, 0.0))) / raw_pos_sum)
    meta = {
        "model_path": str(bundle.get("model_path", "")),
        "p_empty": float(p_empty),
        "p_church_like": float(p_occ),
        "d_empty": float(d_empty),
        "d_church_like": float(d_pos),
    }
    return out, "prototype", meta


def apply_slot_label_constraints(probs: Dict[str, float], slot: dict) -> Tuple[Dict[str, float], List[str]]:
    allowed = slot.get("allowed_labels")
    if not isinstance(allowed, list) or not allowed:
        allowed_types = slot.get("allowed_types")
        if isinstance(allowed_types, list) and allowed_types:
            allowed = ["empty"] + [str(name) for name in allowed_types if str(name) != "empty"]
        else:
            allowed = list(probs.keys())

    allowed_labels = []
    seen = set()
    for name in allowed:
        label = str(name)
        if label in probs and label not in seen:
            allowed_labels.append(label)
            seen.add(label)
    if not allowed_labels:
        allowed_labels = list(probs.keys())

    constrained = {label: (float(prob) if label in seen else 0.0) for label, prob in probs.items()}
    total = float(sum(constrained.values()))
    if total > 1e-12:
        constrained = {label: (value / total) for label, value in constrained.items()}
    return constrained, allowed_labels


def ai_classify_slots_for_earth(
    panel_bgr: np.ndarray,
    backend,
    package,
    earth_name: str,
    ai_bundle: dict,
    confidence_threshold: float,
    margin_threshold: float,
) -> Tuple[List[dict], Dict[str, float]]:
    estimate_fn = getattr(backend, "estimate_global_translation_slot", None)
    if estimate_fn is None:
        estimate_fn = getattr(backend, "estimate_global_translation", None)
    if estimate_fn is None:
        raise AttributeError(
            f"module '{backend.__name__}' has neither estimate_global_translation_slot nor estimate_global_translation"
        )
    dx_norm, dy_norm, align_response = estimate_fn(panel_bgr, package, earth_name)
    h, w = panel_bgr.shape[:2]
    side = min(h, w)
    slots_out: List[dict] = []

    model = ai_bundle["model"]
    labels = ai_bundle["labels"]
    device = ai_bundle["device"]
    patch_size = int(ai_bundle["patch_size"])
    context_scale = float(ai_bundle["context_scale"])
    min_half_frac = float(ai_bundle["min_half_frac"])

    for slot in package.slots_by_earth[earth_name]:
        cx = int(round((float(slot["x_norm"]) + dx_norm) * w))
        cy = int(round((float(slot["y_norm"]) + dy_norm) * h))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        half = int(round(max(float(slot["radius_norm"]) * side * context_scale, min_half_frac * side)))
        half = max(18, half)

        patch = reflect_pad_extract_ai(panel_bgr, cx, cy, half, patch_size)
        raw_probs = infer_slot_probs_ai(model, labels, patch, device)
        raw_probs, slot_classifier_mode, slot_classifier_meta = _gh_slot4_apply_prototype_override(
            raw_probs=raw_probs,
            slot=slot,
            earth_name=earth_name,
            patch_bgr=patch,
        )
        probs, allowed_labels = apply_slot_label_constraints(raw_probs, slot)
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_prob = ordered[0]
        second_prob = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = best_prob - second_prob

        if best_prob < confidence_threshold or margin < margin_threshold:
            final_label = "unknown"
        else:
            final_label = best_label

        if (
            str(earth_name) == "Great Hollow"
            and int(slot.get("slot_id", -1)) == GH_SLOT4_TARGET_SLOT_ID
            and str(slot_classifier_mode) == "prototype"
        ):
            if not isinstance(slot_classifier_meta, dict):
                slot_classifier_meta = {}
            gate_threshold = _gh_slot4_empty_gate_threshold()
            p_empty = float(slot_classifier_meta.get("p_empty", probs.get("empty", 0.0)))
            p_empty = max(0.0, min(1.0, p_empty))
            slot_classifier_meta["gate_threshold"] = float(gate_threshold)
            slot_classifier_meta["p_non_empty"] = float(1.0 - p_empty)

            if p_empty >= gate_threshold:
                other_probs = [float(v) for k, v in probs.items() if str(k) != "empty"]
                best_label = "empty"
                best_prob = float(p_empty)
                second_prob = float(max(other_probs) if other_probs else 0.0)
                margin = float(best_prob - second_prob)
                final_label = "empty"
                slot_classifier_meta["gate_decision"] = "empty"
            else:
                non_empty = {str(k): float(v) for k, v in probs.items() if str(k) != "empty"}
                non_empty_sum = float(sum(non_empty.values()))
                if non_empty_sum > 1e-12 and len(non_empty) > 0:
                    non_empty = {k: (v / non_empty_sum) for k, v in non_empty.items()}
                    non_empty_ranked = sorted(non_empty.items(), key=lambda kv: kv[1], reverse=True)
                    best_label, best_prob = non_empty_ranked[0]
                    second_prob = non_empty_ranked[1][1] if len(non_empty_ranked) > 1 else 0.0
                    margin = float(best_prob - second_prob)
                    if best_prob < confidence_threshold or margin < margin_threshold:
                        final_label = "unknown"
                    else:
                        final_label = str(best_label)
                    slot_classifier_meta["gate_decision"] = "non_empty"
                    slot_classifier_meta["non_empty_best_label"] = str(best_label)
                    slot_classifier_meta["non_empty_best_prob"] = float(best_prob)
                    slot_classifier_meta["non_empty_second_prob"] = float(second_prob)
                    slot_classifier_meta["non_empty_margin"] = float(margin)

        slots_out.append({
            "slot_id": int(slot["slot_id"]),
            "slot_name": str(slot["slot_name"]),
            "label": final_label,
            "best_prob": float(best_prob),
            "second_prob": float(second_prob),
            "margin": float(margin),
            "probs": probs,
            "raw_probs": raw_probs,
            "allowed_labels": allowed_labels,
            "x_norm": float(cx / w),
            "y_norm": float(cy / h),
            "nominal_x_norm": float(slot["x_norm"]),
            "nominal_y_norm": float(slot["y_norm"]),
            "radius_norm": float(slot["radius_norm"]),
            "weight": float(slot.get("weight", 1.0)),
            "slot_classifier_mode": str(slot_classifier_mode),
            "slot_classifier_meta": dict(slot_classifier_meta or {}),
        })

    return slots_out, {
        "dx_norm": float(dx_norm),
        "dy_norm": float(dy_norm),
        "align_response": float(align_response),
    }


SPECIAL_LAYOUT_COMPARISONS: Tuple[Tuple[int, int], ...] = (
    (101, 1089),
)
GREAT_HOLLOW_ICON_ONLY_SLOT_IDS = {2}


def score_layout_ai(
    slots_out: Sequence[dict],
    layout_slot_labels: Sequence[str],
    earth_score: float,
    earth_name: str = "",
    geometry_prior: float = 0.0,
    lost_slot_prior: float = 0.0,
    hard_constraint_prior: float = 0.0,
    evergaol_prior: float = 0.0,
    great_hollow_status_prior: float = 0.0,
    great_hollow_model_layout_prior: float = 0.0,
    layout_infer_prior: float = 0.0,
) -> Tuple[float, dict]:
    eps = 1e-6
    total = 0.0
    total_weight = 0.0
    exact_weight = 0.0
    unknown_weight = 0.0
    hard_mismatch_weight = 0.0
    slot_details = []

    for slot_obs, expected in zip(slots_out, layout_slot_labels):
        slot_id = int(slot_obs.get("slot_id", -1))
        if str(earth_name) == "Great Hollow" and slot_id in GREAT_HOLLOW_ICON_ONLY_SLOT_IDS:
            slot_details.append({
                "slot_id": int(slot_id),
                "slot_name": str(slot_obs["slot_name"]),
                "expected": str(expected),
                "observed": str(slot_obs.get("label", "unknown")),
                "expected_prob": None,
                "best_prob": float(slot_obs.get("best_prob", 0.0)),
                "second_prob": float(slot_obs.get("second_prob", 0.0)),
                "margin": float(slot_obs.get("margin", 0.0)),
                "weight": 0.0,
                "weighted_log_prob": 0.0,
                "icon_only_slot": True,
            })
            continue
        probs = slot_obs["probs"]
        p = max(eps, float(probs.get(str(expected), eps)))
        weight = float(slot_obs.get("weight", 1.0))
        total += weight * math.log(p)
        total_weight += weight

        observed = str(slot_obs["label"])
        if observed == str(expected):
            exact_weight += weight
        elif observed == "unknown":
            unknown_weight += weight
        else:
            hard_mismatch_weight += weight
        slot_details.append({
            "slot_id": int(slot_id),
            "slot_name": str(slot_obs["slot_name"]),
            "expected": str(expected),
            "observed": observed,
            "expected_prob": float(p),
            "best_prob": float(slot_obs.get("best_prob", 0.0)),
            "second_prob": float(slot_obs.get("second_prob", 0.0)),
            "margin": float(slot_obs.get("margin", 0.0)),
            "weight": float(weight),
            "weighted_log_prob": float(weight * math.log(p)),
            "icon_only_slot": False,
        })

    normalized = total / max(1.0, total_weight)
    total = (
        normalized
        + 0.75 * float(earth_score)
        + float(geometry_prior)
        + float(lost_slot_prior)
        + float(hard_constraint_prior)
        + float(evergaol_prior)
        + float(great_hollow_status_prior)
        + float(great_hollow_model_layout_prior)
        + float(layout_infer_prior)
    )
    details = {
        "normalized_log_prob": float(normalized),
        "geometry_prior": float(geometry_prior),
        "lost_slot_prior": float(lost_slot_prior),
        "hard_constraint_prior": float(hard_constraint_prior),
        "evergaol_prior": float(evergaol_prior),
        "great_hollow_status_prior": float(great_hollow_status_prior),
        "great_hollow_model_layout_prior": float(great_hollow_model_layout_prior),
        "layout_infer_prior": float(layout_infer_prior),
        "exact_match_weight": float(exact_weight),
        "hard_mismatch_weight": float(hard_mismatch_weight),
        "unknown_weight": float(unknown_weight),
        "slot_details": slot_details,
    }
    return float(total), details


def compute_earth_geometry_priors(package: SlotRecognitionPackage, observations_by_earth: dict) -> dict:
    slot_counts = {earth_name: len(package.slots_by_earth.get(earth_name, [])) for earth_name in observations_by_earth}
    standard = {
        earth_name: float(obs["align"]["align_response"])
        for earth_name, obs in observations_by_earth.items()
        if slot_counts.get(earth_name, 0) >= 20
    }
    compact = {
        earth_name: float(obs["align"]["align_response"])
        for earth_name, obs in observations_by_earth.items()
        if slot_counts.get(earth_name, 0) <= 10
    }
    best_standard = max(standard.values()) if standard else 0.0
    best_compact = max(compact.values()) if compact else 0.0
    priors = {}
    for earth_name, obs in observations_by_earth.items():
        align_response = float(obs["align"]["align_response"])
        slot_count = slot_counts.get(earth_name, 0)
        prior = 0.0
        if slot_count <= 10 and standard:
            gap = best_standard - align_response
            if gap > 0.0:
                prior -= 2.0 * gap
            if gap > 0.010:
                prior -= 0.08
            if gap > 0.020:
                prior -= 0.08
        elif slot_count >= 20 and compact:
            gap = align_response - best_compact
            if gap > 0.0:
                prior += 0.75 * gap
        priors[earth_name] = {
            "slot_count": int(slot_count),
            "align_response": float(align_response),
            "best_standard_align": float(best_standard),
            "best_compact_align": float(best_compact),
            "geometry_prior": float(prior),
        }
    return priors


def derive_lost_limveld_slot_indices(package: SlotRecognitionPackage, match_radius_norm: float = 0.035) -> dict:
    limveld_slots = list(package.slots_by_earth.get("Limveld", []))
    if not limveld_slots:
        return {}
    out = {}
    for earth_name, slots in package.slots_by_earth.items():
        if earth_name in {"Limveld", "Great Hollow"}:
            continue
        matched_limveld = set()
        for slot in slots:
            sx = float(slot["x_norm"])
            sy = float(slot["y_norm"])
            best_idx = None
            best_dist = float("inf")
            for idx, limveld_slot in enumerate(limveld_slots):
                dx = sx - float(limveld_slot["x_norm"])
                dy = sy - float(limveld_slot["y_norm"])
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist <= match_radius_norm:
                matched_limveld.add(int(best_idx))
        out[earth_name] = [idx for idx in range(len(limveld_slots)) if idx not in matched_limveld]
    return out


def compute_lost_limveld_slot_priors(package: SlotRecognitionPackage, observations_by_earth: dict) -> dict:
    limveld_obs = observations_by_earth.get("Limveld", {}).get("slots")
    if not limveld_obs:
        return {}

    lost_indices_by_earth = derive_lost_limveld_slot_indices(package)
    priors = {}
    for earth_name in observations_by_earth:
        lost_indices = list(lost_indices_by_earth.get(earth_name, []))
        prior = 0.0
        slot_evidence = []
        for idx in lost_indices:
            if idx < 0 or idx >= len(limveld_obs):
                continue
            obs = limveld_obs[idx]
            probs = dict(obs.get("probs", {}))
            empty_prob = float(probs.get("empty", 0.0))
            nonempty_prob = max(0.0, 1.0 - empty_prob)
            label = str(obs.get("label", "unknown"))
            slot_prior = 0.0
            if label == "empty":
                empty_strength = max(0.0, empty_prob - 0.55) / 0.45
                slot_prior += 0.010 + 0.015 * min(1.0, empty_strength)
            elif label == "unknown":
                if nonempty_prob > 0.70:
                    slot_prior -= 0.025 * min(1.0, (nonempty_prob - 0.70) / 0.30)
            else:
                occupied_strength = max(0.0, nonempty_prob - 0.45) / 0.55
                slot_prior -= 0.050 + 0.050 * min(1.0, occupied_strength)
            prior += slot_prior
            slot_evidence.append({
                "slot_id": int(obs.get("slot_id", idx + 1)),
                "slot_name": str(obs.get("slot_name", f"slot_{idx+1:02d}")),
                "label": label,
                "empty_prob": float(empty_prob),
                "nonempty_prob": float(nonempty_prob),
                "slot_prior": float(slot_prior),
            })
        priors[earth_name] = {
            "lost_slot_count": int(len(lost_indices)),
            "lost_slot_prior": float(prior),
            "slot_evidence": slot_evidence,
        }
    return priors


def compute_hard_constraint_priors(package: SlotRecognitionPackage, observations_by_earth: dict) -> dict:
    priors = {
        earth_name: {
            "hard_constraint_prior": 0.0,
            "evidence": [],
        }
        for earth_name in observations_by_earth
    }
    limveld_obs = observations_by_earth.get("Limveld", {}).get("slots")
    if not limveld_obs:
        return priors

    limveld_slots = list(package.slots_by_earth.get("Limveld", []))
    great_hollow_slots = list(package.slots_by_earth.get("Great Hollow", []))
    if limveld_slots and great_hollow_slots:
        matched_limveld = set()
        for gh_slot in great_hollow_slots:
            sx = float(gh_slot["x_norm"])
            sy = float(gh_slot["y_norm"])
            best_idx = None
            best_dist = float("inf")
            for idx, limveld_slot in enumerate(limveld_slots):
                dx = sx - float(limveld_slot["x_norm"])
                dy = sy - float(limveld_slot["y_norm"])
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist <= 0.050:
                matched_limveld.add(int(best_idx))
        forbidden_limveld = [idx for idx in range(len(limveld_slots)) if idx not in matched_limveld]
    else:
        forbidden_limveld = []

    evidence = []
    prior = 0.0
    ranked = []
    for idx in forbidden_limveld:
        if idx < 0 or idx >= len(limveld_obs):
            continue
        obs = limveld_obs[idx]
        probs = dict(obs.get("probs", {}))
        empty_prob = float(probs.get("empty", 0.0))
        nonempty_prob = max(0.0, 1.0 - empty_prob)
        ranked.append((nonempty_prob, idx, obs))
    ranked.sort(reverse=True, key=lambda item: item[0])
    for nonempty_prob, idx, obs in ranked:
        label = str(obs.get("label", "unknown"))
        slot_prior = 0.0
        if label == "unknown":
            if nonempty_prob > 0.75:
                slot_prior -= 0.035 * min(1.0, (nonempty_prob - 0.75) / 0.25)
        elif label != "empty":
            occupied_strength = max(0.0, nonempty_prob - 0.40) / 0.60
            slot_prior -= 0.075 + 0.055 * min(1.0, occupied_strength)
        prior += slot_prior
        evidence.append({
            "slot_id": int(obs.get("slot_id", idx + 1)),
            "slot_name": str(obs.get("slot_name", f"slot_{idx+1:02d}")),
            "label": label,
            "nonempty_prob": float(nonempty_prob),
            "slot_prior": float(slot_prior),
        })
    if "Great Hollow" in priors:
        priors["Great Hollow"] = {
            "hard_constraint_prior": float(prior),
            "evidence": evidence,
        }
    return priors


def world_to_panel_norm(x_world: float, z_world: float) -> Tuple[float, float]:
    px = (float(x_world) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X)
    py = 1.0 - ((float(z_world) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z))
    return float(px), float(py)


def load_evergaol_templates() -> List[dict]:
    global EVERGAOL_TEMPLATE_CACHE
    if EVERGAOL_TEMPLATE_CACHE is not None:
        return EVERGAOL_TEMPLATE_CACHE
    template = cv2.imread(str(EVERGAOL_ICON_PATH), cv2.IMREAD_UNCHANGED)
    if template is None:
        EVERGAOL_TEMPLATE_CACHE = []
        return EVERGAOL_TEMPLATE_CACHE
    if template.ndim == 2:
        gray = template
        alpha = np.full_like(gray, 255)
    elif template.shape[2] == 4:
        bgr = template[:, :, :3]
        alpha = template[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        bgr = template[:, :, :3]
        alpha = np.full(template.shape[:2], 255, dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 180)
    mask = (alpha > 0).astype(np.uint8) * 255
    templates: List[dict] = []
    for scale in (0.55, 0.65, 0.75, 0.85, 0.95):
        w = max(16, int(round(gray.shape[1] * scale)))
        h = max(16, int(round(gray.shape[0] * scale)))
        gray_rs = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        edge_rs = cv2.resize(edge, (w, h), interpolation=cv2.INTER_AREA)
        mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        templates.append({
            "scale": float(scale),
            "gray": gray_rs,
            "edge": edge_rs,
            "mask": mask_rs,
            "w": int(w),
            "h": int(h),
        })
    EVERGAOL_TEMPLATE_CACHE = templates
    return templates


def detect_evergaol_presence(panel_bgr: np.ndarray, dx_norm: float = 0.0, dy_norm: float = 0.0) -> List[dict]:
    templates = load_evergaol_templates()
    evergaol_bundle = load_evergaol_model_bundle(None)
    if not templates:
        return []
    panel_gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    panel_edge = cv2.Canny(panel_gray, 60, 180)
    h, w = panel_gray.shape[:2]
    side = min(h, w)
    search_radius = max(28, int(round(side * 0.045)))
    detections = []
    for idx, (x_world, z_world) in enumerate(EVERGAOL_WORLD_POINTS):
        x_norm, y_norm = world_to_panel_norm(x_world, z_world)
        cx = int(round((x_norm + dx_norm) * w))
        cy = int(round((y_norm + dy_norm) * h))
        best = {
            "score": -1.0,
            "gray_score": -1.0,
            "edge_score": -1.0,
            "scale": None,
            "x_norm": float(cx / max(1, w)),
            "y_norm": float(cy / max(1, h)),
            "present_prob": 0.0,
            "method": "template",
        }
        ai_best = None
        if evergaol_bundle is not None:
            patch_size = int(evergaol_bundle["patch_size"])
            context_scale = float(evergaol_bundle["context_scale"])
            min_half_frac = float(evergaol_bundle["min_half_frac"])
            offsets = (-search_radius, -search_radius // 2, 0, search_radius // 2, search_radius)
            for ox in offsets:
                for oy in offsets:
                    px = max(0, min(w - 1, cx + int(ox)))
                    py = max(0, min(h - 1, cy + int(oy)))
                    half = int(round(max(side * context_scale * 0.018, min_half_frac * side)))
                    half = max(20, half)
                    patch = reflect_pad_extract_ai(panel_bgr, px, py, half, patch_size)
                    present_prob = infer_evergaol_present_prob(evergaol_bundle, patch)
                    if ai_best is None or present_prob > float(ai_best["present_prob"]):
                        ai_best = {
                            "score": float(present_prob),
                            "gray_score": float("nan"),
                            "edge_score": float("nan"),
                            "scale": None,
                            "x_norm": float(px / w),
                            "y_norm": float(py / h),
                            "present_prob": float(present_prob),
                            "method": "ai",
                        }
        template_best = dict(best)
        for tpl in templates:
            tw = int(tpl["w"])
            th = int(tpl["h"])
            x0 = max(0, cx - search_radius - tw // 2)
            y0 = max(0, cy - search_radius - th // 2)
            x1 = min(w, cx + search_radius + (tw + 1) // 2)
            y1 = min(h, cy + search_radius + (th + 1) // 2)
            roi_gray = panel_gray[y0:y1, x0:x1]
            roi_edge = panel_edge[y0:y1, x0:x1]
            if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
                continue
            gray_score_map = cv2.matchTemplate(roi_gray, tpl["gray"], cv2.TM_CCORR_NORMED, mask=tpl["mask"])
            edge_score_map = cv2.matchTemplate(roi_edge, tpl["edge"], cv2.TM_CCORR_NORMED, mask=tpl["mask"])
            gray_score = float(np.nanmax(gray_score_map))
            edge_score = float(np.nanmax(edge_score_map))
            score = 0.65 * gray_score + 0.35 * edge_score
            if score > float(template_best["score"]):
                _, _, _, max_loc = cv2.minMaxLoc(gray_score_map)
                px = x0 + max_loc[0] + tw * 0.5
                py = y0 + max_loc[1] + th * 0.5
                template_best = {
                    "score": float(score),
                    "gray_score": float(gray_score),
                    "edge_score": float(edge_score),
                    "scale": float(tpl["scale"]),
                    "x_norm": float(px / w),
                    "y_norm": float(py / h),
                    "present_prob": min(1.0, max(0.0, (float(score) - 0.30) / 0.45)),
                    "method": "template",
                }
        if ai_best is not None:
            hybrid_prob = 0.35 * float(ai_best["present_prob"]) + 0.65 * float(template_best["present_prob"])
            best = dict(template_best)
            best["present_prob"] = float(hybrid_prob)
            best["method"] = "hybrid"
            best["ai_present_prob"] = float(ai_best["present_prob"])
            best["template_present_prob"] = float(template_best["present_prob"])
        else:
            best = template_best
        present_prob = float(best["present_prob"])
        detections.append({
            "evergaol_id": int(idx + 1),
            "world_x": float(x_world),
            "world_z": float(z_world),
            "expected_x_norm": float(x_norm + dx_norm),
            "expected_y_norm": float(y_norm + dy_norm),
            **best,
        })
    return detections


def compute_evergaol_priors(panel_bgr: np.ndarray, observations_by_earth: dict) -> dict:
    align = observations_by_earth.get("Limveld", {}).get("align", {})
    dx_norm = float(align.get("dx_norm", 0.0))
    dy_norm = float(align.get("dy_norm", 0.0))
    detections = detect_evergaol_presence(panel_bgr, dx_norm=dx_norm, dy_norm=dy_norm)
    priors = {}
    for earth_name in observations_by_earth:
        expected = set(EVERGAOL_EXPECTED_BY_EARTH.get(earth_name, set()))
        prior = 0.0
        evidence = []
        for idx, det in enumerate(detections):
            p = float(det["present_prob"])
            if idx in expected:
                slot_prior = 0.09 * (p - 0.35)
            else:
                slot_prior = 0.10 * (0.35 - p)
            prior += slot_prior
            evidence.append({
                "evergaol_id": int(det["evergaol_id"]),
                "expected_present": bool(idx in expected),
                "present_prob": float(p),
                "match_score": float(det["score"]),
                "slot_prior": float(slot_prior),
            })
        priors[earth_name] = {
            "evergaol_prior": float(prior),
            "detections": detections,
            "evidence": evidence,
        }
    return priors


def draw_evergaol_debug(panel_bgr: np.ndarray, evergaol_priors: dict) -> np.ndarray:
    out = panel_bgr.copy()
    h, w = out.shape[:2]
    limveld = evergaol_priors.get("Limveld", {})
    detections = list(limveld.get("detections", []))
    for idx, det in enumerate(detections):
        ex = int(round(float(det["expected_x_norm"]) * w))
        ey = int(round(float(det["expected_y_norm"]) * h))
        px = int(round(float(det["x_norm"]) * w))
        py = int(round(float(det["y_norm"]) * h))
        prob = float(det.get("present_prob", 0.0))
        color = (0, 255, 0) if prob >= 0.5 else (0, 128, 255)
        cv2.circle(out, (ex, ey), 14, (255, 255, 0), 1)
        cv2.circle(out, (px, py), 8, color, -1)
        text = f"E{idx+1}: {prob:.2f} {det.get('method','?')}"
        tx = max(6, min(w - 220, px + 8))
        ty = max(16, min(h - 8, py - 8))
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def _load_icon_gray_edge_mask(path: Path):
    icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None
    if icon.ndim == 2:
        gray = icon
        alpha = np.full_like(gray, 255)
    elif icon.shape[2] == 4:
        bgr = icon[:, :, :3]
        alpha = icon[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        bgr = icon[:, :, :3]
        alpha = np.full(icon.shape[:2], 255, dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 180)
    mask = (alpha > 0).astype(np.uint8) * 255
    return {"gray": gray, "edge": edge, "mask": mask}


def _load_icon_color_gray_edge_mask(path: Path):
    icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None
    if icon.ndim == 2:
        bgr = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)
        alpha = np.full(icon.shape[:2], 255, dtype=np.uint8)
    elif icon.shape[2] == 4:
        bgr = icon[:, :, :3]
        alpha = icon[:, :, 3]
    else:
        bgr = icon[:, :, :3]
        alpha = np.full(icon.shape[:2], 255, dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 180)
    mask = (alpha > 0).astype(np.uint8) * 255
    return {"bgr": bgr, "gray": gray, "edge": edge, "mask": mask}


def load_great_hollow_status_icon_sources() -> dict:
    global GREAT_HOLLOW_STATUS_ICON_SOURCES
    if GREAT_HOLLOW_STATUS_ICON_SOURCES is not None:
        return GREAT_HOLLOW_STATUS_ICON_SOURCES
    out = {}
    for icon_name, path in GREAT_HOLLOW_STATUS_ICON_PATHS.items():
        data = _load_icon_color_gray_edge_mask(path)
        if data is not None:
            out[icon_name] = data
    GREAT_HOLLOW_STATUS_ICON_SOURCES = out
    return out


def _find_great_hollow_pattern_path(layout_number: int) -> Optional[Path]:
    for base_dir in GREAT_HOLLOW_STATUS_PATTERN_DIRS:
        matches = sorted(base_dir.glob(f"pattern_{int(layout_number):04d}_*.json"))
        if matches:
            return matches[0]
    return None


def _great_hollow_world_to_norm(x_world: float, z_world: float) -> Tuple[float, float]:
    px = (float(x_world) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X) * IMAGE_W
    py = (float(z_world) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z) * IMAGE_H
    cropped_w = IMAGE_W - (2.0 * GREAT_HOLLOW_CROP_BORDER)
    cropped_h = IMAGE_H - (2.0 * GREAT_HOLLOW_CROP_BORDER)
    x_norm = (px - GREAT_HOLLOW_CROP_BORDER) / cropped_w
    y_norm = (py - GREAT_HOLLOW_CROP_BORDER) / cropped_h
    return float(x_norm), float(y_norm)


def _load_great_hollow_patternset_statuses(layout_number: int) -> dict:
    pattern_path = GREAT_HOLLOW_PATTERNSET_DIR / f"pattern_{int(layout_number):04d}_mappatternset_bundle.json"
    if not pattern_path.exists():
        return {}
    try:
        payload = json.loads(pattern_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    statuses = {}
    for item in payload.get("activeMapPatternSetFields", []) or []:
        field_name = str(item.get("field", "")).strip()
        mapping = GREAT_HOLLOW_PATTERNSET_STATUS_BY_FIELD.get(field_name)
        if mapping is None:
            continue
        anchor_type, status_type = mapping
        point = GREAT_HOLLOW_SYNTHETIC_WORLD_POINTS.get(anchor_type)
        if point is None:
            continue
        x_norm, y_norm = _great_hollow_world_to_norm(point[0], point[1])
        statuses[anchor_type] = {
            "status_type": status_type,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "source_id": None,
        }
    return statuses


def get_layout_great_hollow_statuses(layout: dict) -> dict:
    layout_number = int(layout.get("layout_number", -1))
    cached = GREAT_HOLLOW_LAYOUT_STATUS_CACHE.get(layout_number)
    if cached is not None:
        return cached

    statuses = {}
    for item in layout.get("great_hollow_statuses", []) or []:
        anchor_type = str(item.get("anchor_type", "")).strip()
        status_type = str(item.get("status_type", "")).strip()
        if anchor_type not in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE:
            continue
        if status_type not in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE[anchor_type]:
            continue
        statuses[anchor_type] = {
            "status_type": status_type,
            "x_norm": float(item.get("x_norm", 0.0)),
            "y_norm": float(item.get("y_norm", 0.0)),
            "source_id": item.get("source_id", None),
        }
    if statuses:
        GREAT_HOLLOW_LAYOUT_STATUS_CACHE[layout_number] = statuses
        return statuses

    pattern_path = _find_great_hollow_pattern_path(layout_number)
    if pattern_path is None:
        GREAT_HOLLOW_LAYOUT_STATUS_CACHE[layout_number] = {}
        return {}
    try:
        pattern_data = json.loads(pattern_path.read_text(encoding="utf-8"))
    except Exception:
        GREAT_HOLLOW_LAYOUT_STATUS_CACHE[layout_number] = {}
        return {}

    by_source_id = {}
    for item in pattern_data.get("landmarks", []):
        by_source_id.setdefault(item.get("source_id", None), []).append(item)

    for source_id, items in by_source_id.items():
        overlay_item = None
        base_item = None
        for item in items:
            item_type = str(item.get("type", "")).strip()
            if item_type in GREAT_HOLLOW_STATUS_ICON_PATHS:
                overlay_item = item
            if item.get("source") != "overlay" and item_type in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE:
                base_item = item
        if overlay_item is None or base_item is None:
            continue
        anchor_type = str(base_item.get("type", "")).strip()
        status_type = str(overlay_item.get("type", "")).strip()
        if status_type not in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE.get(anchor_type, ()):
            continue
        statuses[anchor_type] = {
            "status_type": status_type,
            "x_norm": float(overlay_item.get("x_norm", 0.0)),
            "y_norm": float(overlay_item.get("y_norm", 0.0)),
            "source_id": source_id,
        }
    bundle_statuses = _load_great_hollow_patternset_statuses(layout_number)
    for anchor_type, info in bundle_statuses.items():
        statuses.setdefault(anchor_type, info)
    GREAT_HOLLOW_LAYOUT_STATUS_CACHE[layout_number] = statuses
    return statuses


def get_great_hollow_status_anchor_norms(package: SlotRecognitionPackage) -> dict:
    global GREAT_HOLLOW_STATUS_ANCHOR_CACHE
    if GREAT_HOLLOW_STATUS_ANCHOR_CACHE is not None:
        return GREAT_HOLLOW_STATUS_ANCHOR_CACHE

    anchors = {}
    gh_layouts = list(package.layouts_by_earth.get("Great Hollow", []))
    for anchor_type in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE:
        xs = []
        ys = []
        for layout in gh_layouts:
            info = get_layout_great_hollow_statuses(layout).get(anchor_type)
            if info is None:
                continue
            xs.append(float(info["x_norm"]))
            ys.append(float(info["y_norm"]))
        if xs and ys:
            anchors[anchor_type] = {
                "x_norm": float(sum(xs) / len(xs)),
                "y_norm": float(sum(ys) / len(ys)),
            }
            continue
        for slot in package.slots_by_earth.get("Great Hollow", []):
            allowed = {str(name) for name in slot.get("allowed_types", [])}
            if allowed == {anchor_type}:
                offset_norm = 23.0 / 1280.0
                anchors[anchor_type] = {
                    "x_norm": float(slot["x_norm"]) + offset_norm,
                    "y_norm": float(slot["y_norm"]) + offset_norm,
                }
                break
    GREAT_HOLLOW_STATUS_ANCHOR_CACHE = anchors
    return anchors


def _gh_read_image(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _gh_resize_to_match(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    if reference.shape[:2] == query.shape[:2]:
        return query
    return cv2.resize(query, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)


def _gh_crop_with_padding(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + size)
    y1 = min(h, y + size)
    patch = img[y0:y1, x0:x1].copy()
    if patch.size == 0:
        raise RuntimeError(f"Empty crop at x={x}, y={y}, size={size} for image size {w}x{h}.")
    top = y0 - y
    left = x0 - x
    bottom = (y + size) - y1
    right = (x + size) - x1
    if top or left or bottom or right:
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
    return patch


def _gh_get_fixed_status_params(img: np.ndarray, xy_norm: Tuple[float, float]) -> Tuple[Tuple[int, int], int, int]:
    h, w = img.shape[:2]
    x = int(round(xy_norm[0] * w))
    y = int(round(xy_norm[1] * h))
    scale = min(h, w) / 1280.0
    crop_size = max(24, int(round(72 * scale)))
    search_pad = max(4, int(round(12 * scale)))
    return (x, y), crop_size, search_pad


def _gh_feature_vector(img: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hh, ww = h.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    cy, cx = hh / 2.0, ww / 2.0
    radial = np.exp(-((((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5) / (min(hh, ww) / 2.6)) ** 2)
    sat_mask = (s > 50) & (v > 40)
    weights = radial * sat_mask.astype(np.float32)
    if float(weights.sum()) < 1.0:
        weights = radial.astype(np.float32)

    angle = h.astype(np.float32) / 180.0 * 2.0 * np.pi
    hue_x = float((np.cos(angle) * weights).sum() / weights.sum())
    hue_y = float((np.sin(angle) * weights).sum() / weights.sum())
    mean_hue = float(math.atan2(hue_y, hue_x))
    mean_sat = float((s * weights).sum() / weights.sum() / 255.0)
    mean_val = float((v * weights).sum() / weights.sum() / 255.0)

    color_hist = cv2.calcHist([hsv], [0, 1], sat_mask.astype(np.uint8), [12, 8], [0, 180, 0, 256]).flatten().astype(np.float32)
    if float(color_hist.sum()) > 0:
        color_hist /= color_hist.sum()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    grad_hist = np.histogram(ang, bins=8, range=(0, 2 * np.pi), weights=mag * radial)[0].astype(np.float32)
    if float(grad_hist.sum()) > 0:
        grad_hist /= grad_hist.sum()

    return {
        "hue": mean_hue,
        "sat": mean_sat,
        "val": mean_val,
        "color_hist": color_hist,
        "grad_hist": grad_hist,
    }


def _gh_circular_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 2 * np.pi - d) / np.pi


def _gh_patch_similarity(a: np.ndarray, b: np.ndarray) -> float:
    fa = _gh_feature_vector(a)
    fb = _gh_feature_vector(b)
    hue_score = 1.0 - _gh_circular_distance(float(fa["hue"]), float(fb["hue"]))
    hist_score = float(cv2.compareHist(np.asarray(fa["color_hist"], dtype=np.float32), np.asarray(fb["color_hist"], dtype=np.float32), cv2.HISTCMP_CORREL)) if np.asarray(fa["color_hist"]).sum() > 0 and np.asarray(fb["color_hist"]).sum() > 0 else 0.0
    grad_score = float(cv2.compareHist(np.asarray(fa["grad_hist"], dtype=np.float32), np.asarray(fb["grad_hist"], dtype=np.float32), cv2.HISTCMP_CORREL)) if np.asarray(fa["grad_hist"]).sum() > 0 and np.asarray(fb["grad_hist"]).sum() > 0 else 0.0
    sv_score = 1.0 - (abs(float(fa["sat"]) - float(fb["sat"])) + abs(float(fa["val"]) - float(fb["val"]))) / 2.0
    return float(0.45 * hue_score + 0.30 * hist_score + 0.15 * grad_score + 0.10 * sv_score)


def _gh_classify_fixed_status_window(search: np.ndarray, templates: dict) -> Tuple[List[Tuple[str, float, Tuple[int, int]]], Tuple[int, int]]:
    template_h, template_w = next(iter(templates.values())).shape[:2]
    results = []
    for label, template in templates.items():
        best_score = -1.0
        best_loc = (0, 0)
        for dy in range(0, search.shape[0] - template_h + 1):
            for dx in range(0, search.shape[1] - template_w + 1):
                patch = search[dy:dy + template_h, dx:dx + template_w]
                score = _gh_patch_similarity(patch, template)
                if score > best_score:
                    best_score = score
                    best_loc = (dx, dy)
        results.append((label, float(best_score), best_loc))
    results.sort(key=lambda row: row[1], reverse=True)
    return results, (template_w, template_h)


def load_great_hollow_fixed_status_templates() -> dict:
    global GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE
    if GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE is not None:
        return GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE

    ref1 = None
    ref2 = None
    for path in GREAT_HOLLOW_FIXED_STATUS_REFS["ref1"]:
        ref1 = _gh_read_image(path)
        if ref1 is not None:
            break
    for path in GREAT_HOLLOW_FIXED_STATUS_REFS["ref2"]:
        ref2 = _gh_read_image(path)
        if ref2 is not None:
            break
    if ref1 is None or ref2 is None:
        GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE = {}
        return GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE

    ref2 = _gh_resize_to_match(ref1, ref2)
    templates = {}
    for anchor_type, cfg in GREAT_HOLLOW_FIXED_STATUS_CONFIG.items():
        (x, y), crop_size, search_pad = _gh_get_fixed_status_params(ref1, cfg["xy_norm"])
        templates[anchor_type] = {
            "templates": {
                str(cfg["ref1_label"]): _gh_crop_with_padding(ref1, x, y, crop_size),
                str(cfg["ref2_label"]): _gh_crop_with_padding(ref2, x, y, crop_size),
            },
            "meta": {
                "xy_norm": tuple(float(v) for v in cfg["xy_norm"]),
                "crop_size": int(crop_size),
                "search_pad": int(search_pad),
            },
        }
    GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE = templates
    return GREAT_HOLLOW_FIXED_STATUS_TEMPLATE_CACHE


def _load_icon_recognizer_v4_module():
    global ICON_RECOGNIZER_V4_MODULE, ICON_RECOGNIZER_V4_LOAD_ATTEMPTED
    if ICON_RECOGNIZER_V4_LOAD_ATTEMPTED:
        return ICON_RECOGNIZER_V4_MODULE
    ICON_RECOGNIZER_V4_LOAD_ATTEMPTED = True
    candidates = [
        Path("icon_recognizer_v4.py"),
        Path(__file__).resolve().parent / "icon_recognizer_v4.py",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("icon_recognizer_v4_runtime", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "recognize_markers"):
                ICON_RECOGNIZER_V4_MODULE = module
                return ICON_RECOGNIZER_V4_MODULE
        except Exception:
            continue
    ICON_RECOGNIZER_V4_MODULE = None
    return None


def _resolve_first_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
        resolved = Path(__file__).resolve().parent / path
        if resolved.exists():
            return resolved
    return None


class _GHStatusTinyCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _gh_status_classifier_mode() -> str:
    raw = str(os.getenv("GH_STATUS_CLASSIFIER_MODE", GREAT_HOLLOW_STATUS_CLASSIFIER_MODE_DEFAULT)).strip().lower()
    if raw not in {"classic", "models"}:
        return "classic"
    return raw


def _gh_load_status_tiny_model(anchor_type: str):
    key = str(anchor_type)
    if key in GREAT_HOLLOW_STATUS_MODEL_CACHE:
        return GREAT_HOLLOW_STATUS_MODEL_CACHE[key]
    candidates = GREAT_HOLLOW_STATUS_MODEL_PATHS.get(key, ())
    model_path = _resolve_first_existing_path(candidates)
    if model_path is None:
        GREAT_HOLLOW_STATUS_MODEL_CACHE[key] = None
        return None
    try:
        device = choose_torch_device()
        ckpt = torch.load(str(model_path), map_location=device)
        label_map = dict(ckpt.get("label_map", {}))
        out_size = int(ckpt.get("out_size", 40))
        model = _GHStatusTinyCNN(num_classes=max(2, len(label_map) or 2)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        id_to_label = {int(v): str(k) for k, v in label_map.items()}
        bundle = {
            "model": model,
            "device": device,
            "id_to_label": id_to_label,
            "out_size": out_size,
            "path": model_path,
        }
        GREAT_HOLLOW_STATUS_MODEL_CACHE[key] = bundle
        return bundle
    except Exception:
        GREAT_HOLLOW_STATUS_MODEL_CACHE[key] = None
        return None


def _gh_model_probs_from_patch(bundle: dict, patch_bgr: np.ndarray) -> Dict[str, float]:
    out_size = int(bundle["out_size"])
    patch_rs = cv2.resize(patch_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    x = patch_to_tensor_ai(patch_rs, bundle["device"])
    with torch.no_grad():
        logits = bundle["model"](x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    id_to_label = dict(bundle["id_to_label"])
    return {str(id_to_label.get(i, str(i))): float(probs[i]) for i in range(len(probs))}


def _gh_layout_model_match_mode() -> str:
    raw = str(os.getenv("GH_LAYOUT_MODEL_MATCH_MODE", GREAT_HOLLOW_LAYOUT_MODEL_MATCH_MODE_DEFAULT)).strip().lower()
    if raw not in {"classic", "models"}:
        return "models"
    return raw


def _gh_use_parallel_panel_classifier() -> bool:
    default_raw = "1" if GH_USE_PARALLEL_PANEL_CLASSIFIER_DEFAULT else "0"
    raw = str(os.getenv("GH_USE_PARALLEL_PANEL_CLASSIFIER", default_raw)).strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


def _gh_load_parallel_panel_classifier_module():
    global GH_PARALLEL_PANEL_CLASSIFIER_MODULE, GH_PARALLEL_PANEL_CLASSIFIER_LOAD_ATTEMPTED
    if GH_PARALLEL_PANEL_CLASSIFIER_MODULE is not None:
        return GH_PARALLEL_PANEL_CLASSIFIER_MODULE
    if GH_PARALLEL_PANEL_CLASSIFIER_LOAD_ATTEMPTED:
        return None
    GH_PARALLEL_PANEL_CLASSIFIER_LOAD_ATTEMPTED = True

    module_path = Path(__file__).resolve().parent / "Testing" / "run_parallel_panel_classifiers.py"
    if not module_path.exists():
        return None

    try:
        mod_name = "gh_parallel_panel_classifiers_runtime"
        spec = importlib.util.spec_from_file_location(mod_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        GH_PARALLEL_PANEL_CLASSIFIER_MODULE = module
        return module
    except Exception:
        GH_PARALLEL_PANEL_CLASSIFIER_MODULE = None
        return None


def _gh_parallel_pick_stage_result(entry: dict) -> Tuple[dict, str, str, dict]:
    etype = str(entry.get("type", ""))
    if etype == "two_stage_cascade":
        final_source = str(entry.get("final_source", "stage1")).strip().lower()
        if final_source == "stage2" and isinstance(entry.get("stage2"), dict):
            return (
                dict(entry.get("stage2", {})),
                "stage2",
                str(entry.get("stage2_model_path", "")),
                dict(entry.get("stage2_model_meta", {})),
            )
        return (
            dict(entry.get("stage1", {})),
            "stage1",
            str(entry.get("stage1_model_path", "")),
            dict(entry.get("stage1_model_meta", {})),
        )
    return (
        dict(entry),
        "single",
        str(entry.get("model_path", "")),
        dict(entry.get("model_meta", {})),
    )


def _gh_convert_parallel_payload_to_model_observations(par_payload: dict, ctx: Optional[RunContext] = None) -> dict:
    if not isinstance(par_payload, dict):
        return {}
    classifiers = dict(par_payload.get("classifiers", {}))
    if not classifiers:
        return {}

    mapping = {
        "slot2": "slot2_great_church_status",
        "palace": "palace_status",
        "fortress": "fortress_status",
        "slot6": "slot6_ruins_status",
        "slot4": "slot4_minor",
        "slot5": "slot5_minor",
        "slot7": "slot7_minor",
    }

    out: dict = {}
    for src_key, spec_key in mapping.items():
        src_entry = classifiers.get(str(src_key))
        spec_cfg = GREAT_HOLLOW_LAYOUT_MODEL_SPECS.get(str(spec_key))
        if not isinstance(src_entry, dict) or not isinstance(spec_cfg, dict):
            continue

        stage_entry, stage_kind, model_path, model_meta = _gh_parallel_pick_stage_result(src_entry)
        ranking = list(stage_entry.get("ranking", []))
        best_label = str(stage_entry.get("predicted_label", src_entry.get("final_label", "")))
        best_prob = float(stage_entry.get("confidence", 0.0))
        second_label = str(ranking[1].get("label", "")) if len(ranking) > 1 else ""
        second_prob = float(ranking[1].get("probability", 0.0)) if len(ranking) > 1 else 0.0
        margin = float(best_prob - second_prob)
        class_names = [str(it.get("label", "")) for it in ranking]
        crop_rect = dict(src_entry.get("crop_rect", {}))

        obs = {
            "spec_key": str(spec_key),
            "kind": str(spec_cfg.get("kind", "")),
            "source": "parallel_panel_script",
            "center_norm": tuple(spec_cfg.get("center_norm", (0.5, 0.5))),
            "center_px": {
                "x": int(crop_rect.get("cx", 0)),
                "y": int(crop_rect.get("cy", 0)),
            },
            "crop_size": int(crop_rect.get("size", spec_cfg.get("crop_size", 200))),
            "model_path": str(model_path),
            "module_path": "Testing/run_parallel_panel_classifiers.py",
            "class_names": class_names,
            "best_label": str(best_label),
            "best_prob": float(best_prob),
            "second_label": str(second_label),
            "second_prob": float(second_prob),
            "margin": float(margin),
            "ranking": ranking,
            "weight": float(spec_cfg.get("weight", 0.0)),
            "parallel_stage": str(stage_kind),
        }
        if "anchor_type" in spec_cfg:
            obs["anchor_type"] = str(spec_cfg.get("anchor_type"))
            obs["predicted_status"] = _gh_status_from_model_label(str(spec_key), str(best_label))
        if "slot_id" in spec_cfg:
            obs["slot_id"] = int(spec_cfg.get("slot_id"))
            final_label = str(src_entry.get("final_label", best_label))
            obs["predicted_slot_label"] = _gh_canonical_slot_label(final_label)

        out[str(spec_key)] = obs
        if ctx is not None:
            extra = ""
            if "predicted_status" in obs:
                extra = f" status={obs.get('predicted_status')}"
            if "predicted_slot_label" in obs:
                extra = f" slot={obs.get('predicted_slot_label')}"
            ctx.log(
                f"GH parallel {spec_key}: best={obs['best_label']} "
                f"p={float(obs['best_prob']):.4f} margin={float(obs['margin']):.4f}{extra}"
            )

    return out


def preprocess_panel_for_layout_model_great_hollow(raw_panel_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        raw_panel_bgr,
        (GREAT_HOLLOW_LAYOUT_MODEL_ASYM_INPUT_SIZE, GREAT_HOLLOW_LAYOUT_MODEL_ASYM_INPUT_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    left = int(GREAT_HOLLOW_LAYOUT_MODEL_CROP_LEFT_PX)
    right = int(GREAT_HOLLOW_LAYOUT_MODEL_CROP_RIGHT_PX)
    top = int(GREAT_HOLLOW_LAYOUT_MODEL_CROP_TOP_PX)
    bottom = int(GREAT_HOLLOW_LAYOUT_MODEL_CROP_BOTTOM_PX)
    y0 = max(0, top)
    y1 = min(GREAT_HOLLOW_LAYOUT_MODEL_ASYM_INPUT_SIZE, GREAT_HOLLOW_LAYOUT_MODEL_ASYM_INPUT_SIZE - bottom)
    x0 = max(0, left)
    x1 = min(GREAT_HOLLOW_LAYOUT_MODEL_ASYM_INPUT_SIZE, GREAT_HOLLOW_LAYOUT_MODEL_ASYM_INPUT_SIZE - right)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Invalid Great Hollow layout-model crop bounds.")
    return resized[y0:y1, x0:x1].copy()


def _gh_layout_model_load_bundle(spec_key: str) -> Optional[dict]:
    cached = GREAT_HOLLOW_LAYOUT_MODEL_CACHE.get(str(spec_key))
    if cached is False:
        return None
    if isinstance(cached, dict):
        return cached

    spec_cfg = GREAT_HOLLOW_LAYOUT_MODEL_SPECS.get(str(spec_key))
    if spec_cfg is None:
        GREAT_HOLLOW_LAYOUT_MODEL_CACHE[str(spec_key)] = False
        return None

    module_path = _resolve_first_existing_path(spec_cfg.get("module_paths", ()))
    model_path = _resolve_first_existing_path(spec_cfg.get("model_paths", ()))
    if module_path is None or model_path is None:
        GREAT_HOLLOW_LAYOUT_MODEL_CACHE[str(spec_key)] = False
        return None

    try:
        unique_mod_name = f"gh_layout_model_lib_{re.sub(r'[^a-zA-Z0-9_]+', '_', str(spec_key))}"
        spec = importlib.util.spec_from_file_location(unique_mod_name, module_path)
        if spec is None or spec.loader is None:
            GREAT_HOLLOW_LAYOUT_MODEL_CACHE[str(spec_key)] = False
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_mod_name] = module
        spec.loader.exec_module(module)

        device = choose_torch_device()
        model, class_names, image_size, meta = module.load_checkpoint(Path(model_path), device)
        bundle = {
            "spec_key": str(spec_key),
            "module_path": Path(module_path),
            "model_path": Path(model_path),
            "module": module,
            "device": device,
            "model": model,
            "class_names": [str(x) for x in class_names],
            "image_size": int(image_size),
            "meta": dict(meta or {}),
        }
        GREAT_HOLLOW_LAYOUT_MODEL_CACHE[str(spec_key)] = bundle
        return bundle
    except Exception:
        GREAT_HOLLOW_LAYOUT_MODEL_CACHE[str(spec_key)] = False
        return None


def _gh_layout_model_infer_patch(bundle: dict, patch_bgr: np.ndarray) -> dict:
    module = bundle["module"]
    device = bundle["device"]
    model = bundle["model"]
    class_names = list(bundle["class_names"])
    image_size = int(bundle["image_size"])

    x = module.preprocess_bgr_200(patch_bgr, image_size=image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    ranking = sorted(
        [{"label": str(class_names[i]), "probability": float(probs[i])} for i in range(len(class_names))],
        key=lambda item: float(item["probability"]),
        reverse=True,
    )
    best = ranking[0] if ranking else {"label": "", "probability": 0.0}
    second = ranking[1] if len(ranking) > 1 else {"label": "", "probability": 0.0}
    return {
        "best_label": str(best["label"]),
        "best_prob": float(best["probability"]),
        "second_label": str(second["label"]),
        "second_prob": float(second["probability"]),
        "margin": float(float(best["probability"]) - float(second["probability"])),
        "ranking": ranking,
    }


def _gh_canonical_slot_label(label: str) -> str:
    t = str(label or "").strip().lower()
    t = t.replace("_", " ").replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t == "empty":
        return "empty"
    if "sorcer" in t and "rise" in t:
        return "sorcerers rise"
    if "church" in t:
        return "church"
    return t


def _gh_status_from_model_label(spec_key: str, label: str) -> Optional[str]:
    low = str(label or "").strip().lower().replace("_", " ")
    if str(spec_key) == "slot2_great_church_status":
        if "sleep" in low:
            return "Status_Sleep"
        if "lightning" in low:
            return "Status_Lightning"
    if str(spec_key) == "slot6_ruins_status":
        if "bleed" in low:
            return "Status_Bleed"
        if "poison" in low:
            return "Status_Poison"
    if "status_" in low:
        m = re.search(r"status[_ ]([a-z]+)", low)
        if m:
            return "Status_" + str(m.group(1)).capitalize()
    if "rot" in low:
        return "Status_Rot"
    if "holy" in low:
        return "Status_Holy"
    if "fire" in low:
        return "Status_Fire"
    if "magic" in low:
        return "Status_Magic"
    if "sleep" in low:
        return "Status_Sleep"
    if "lightning" in low:
        return "Status_Lightning"
    if "poison" in low:
        return "Status_Poison"
    if "bleed" in low:
        return "Status_Bleed"
    return None


def infer_great_hollow_layout_model_observations(
    raw_panel_bgr: Optional[np.ndarray],
    package: SlotRecognitionPackage,
    ctx: Optional[RunContext] = None,
) -> dict:
    _ = package
    if raw_panel_bgr is None or raw_panel_bgr.size == 0:
        return {}
    if _gh_layout_model_match_mode() != "models":
        return {}

    if _gh_use_parallel_panel_classifier():
        mod = _gh_load_parallel_panel_classifier_module()
        if mod is not None and hasattr(mod, "run_all_from_panel_bgr"):
            try:
                payload = mod.run_all_from_panel_bgr(  # type: ignore[attr-defined]
                    panel_bgr=raw_panel_bgr,
                    panel_source="in_memory_panel",
                    output_json=None,
                    device_name="auto",
                )
                parallel_obs = _gh_convert_parallel_payload_to_model_observations(payload, ctx=ctx)
                if parallel_obs:
                    return parallel_obs
                if ctx is not None:
                    ctx.log("GH parallel panel classifiers produced no observations; falling back to direct model observations.")
            except Exception as exc:
                if ctx is not None:
                    ctx.log(f"Warning: GH parallel panel classifiers failed ({exc}); falling back to direct model observations.")

    out = {}
    try:
        gh_asym = preprocess_panel_for_layout_model_great_hollow(raw_panel_bgr)
    except Exception:
        gh_asym = None

    for spec_key, spec_cfg in GREAT_HOLLOW_LAYOUT_MODEL_SPECS.items():
        bundle = _gh_layout_model_load_bundle(str(spec_key))
        if bundle is None:
            continue

        source_kind = str(spec_cfg.get("source", ""))
        if source_kind == "panel_raw":
            src = raw_panel_bgr
        else:
            src = gh_asym
        if src is None or src.size == 0:
            continue

        h, w = src.shape[:2]
        x_norm, y_norm = spec_cfg.get("center_norm", (0.5, 0.5))
        cx = int(round(float(x_norm) * w))
        cy = int(round(float(y_norm) * h))
        cx += int(spec_cfg.get("dx_px", 0))
        cy += int(spec_cfg.get("dy_px", 0))
        crop_size = int(spec_cfg.get("crop_size", 200))
        patch = _gh_crop_center_with_padding(src, cx, cy, crop_size)
        infer = _gh_layout_model_infer_patch(bundle, patch)

        obs = {
            "spec_key": str(spec_key),
            "kind": str(spec_cfg.get("kind", "")),
            "source": source_kind,
            "center_norm": (float(x_norm), float(y_norm)),
            "center_px": {"x": int(cx), "y": int(cy)},
            "crop_size": int(crop_size),
            "model_path": str(bundle.get("model_path", "")),
            "module_path": str(bundle.get("module_path", "")),
            "class_names": list(bundle.get("class_names", [])),
            "best_label": str(infer.get("best_label", "")),
            "best_prob": float(infer.get("best_prob", 0.0)),
            "second_label": str(infer.get("second_label", "")),
            "second_prob": float(infer.get("second_prob", 0.0)),
            "margin": float(infer.get("margin", 0.0)),
            "ranking": list(infer.get("ranking", [])),
            "weight": float(spec_cfg.get("weight", 0.0)),
        }
        if "anchor_type" in spec_cfg:
            obs["anchor_type"] = str(spec_cfg.get("anchor_type"))
            obs["predicted_status"] = _gh_status_from_model_label(str(spec_key), str(obs["best_label"]))
        if "slot_id" in spec_cfg:
            obs["slot_id"] = int(spec_cfg.get("slot_id"))
            obs["predicted_slot_label"] = _gh_canonical_slot_label(str(obs["best_label"]))

        out[str(spec_key)] = obs
        if ctx is not None:
            extra = ""
            if "predicted_status" in obs:
                extra = f" status={obs.get('predicted_status')}"
            if "predicted_slot_label" in obs:
                extra = f" slot={obs.get('predicted_slot_label')}"
            ctx.log(
                f"GH model {spec_key}: best={obs['best_label']} "
                f"p={float(obs['best_prob']):.4f} margin={float(obs['margin']):.4f}{extra}"
            )

    return out


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def is_dual_shot_debug_enabled() -> bool:
    return _truthy_env("NIGHTREIGN_DUALSHOT_DEBUG", _truthy_env("NIGHTREIGN_DEBUG", False))


def _gh_crop_center_with_padding(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    half = int(size) // 2
    x0 = int(cx) - half
    y0 = int(cy) - half
    return _gh_crop_with_padding(img, x0, y0, int(size))


def _gh_under_softmax2(a: float, b: float, temp: float = 0.10) -> Tuple[float, float]:
    t = max(1e-6, float(temp))
    xa = float(a) / t
    xb = float(b) / t
    m = max(xa, xb)
    ea = math.exp(xa - m)
    eb = math.exp(xb - m)
    z = ea + eb + 1e-12
    return float(ea / z), float(eb / z)


def _gh_under_sharpen_prob(p: float, scale: float = 3.8) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    logit = math.log(p / (1.0 - p))
    z = float(scale) * logit
    return float(1.0 / (1.0 + math.exp(-z)))


def _gh_under_color_desc(patch_bgr: np.ndarray) -> Dict[str, object]:
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    mask = ((sch > 40) & (vch > 40)).astype(np.uint8)
    if int(mask.sum()) < 8:
        mask = np.ones_like(mask, dtype=np.uint8)

    hist = cv2.calcHist([hsv], [0, 1], mask, [12, 8], [0, 180, 0, 256]).astype(np.float32).flatten()
    hist /= float(hist.sum() + 1e-6)

    hh = hch.astype(np.float32)
    ss = sch.astype(np.float32)
    vv = vch.astype(np.float32)
    mm = mask.astype(np.float32)
    wsum = float(mm.sum() + 1e-6)
    return {
        "hist": hist,
        "mean_h": float((hh * mm).sum() / wsum),
        "mean_s": float((ss * mm).sum() / wsum),
        "mean_v": float((vv * mm).sum() / wsum),
    }


def _gh_under_color_score(desc: Dict[str, object], proto: Dict[str, object]) -> float:
    hist_corr = float(
        cv2.compareHist(
            np.asarray(desc["hist"], dtype=np.float32),
            np.asarray(proto["hist"], dtype=np.float32),
            cv2.HISTCMP_CORREL,
        )
    )
    dh = abs(float(desc["mean_h"]) - float(proto["mean_h"]))
    dh = min(dh, 180.0 - dh) / 90.0
    ds = abs(float(desc["mean_s"]) - float(proto["mean_s"])) / 255.0
    dv = abs(float(desc["mean_v"]) - float(proto["mean_v"])) / 255.0
    hsv_close = 1.0 - min(1.0, 0.55 * dh + 0.30 * ds + 0.15 * dv)
    return float(0.82 * hist_corr + 0.18 * hsv_close)


def _find_great_hollow_under_pattern_path(layout_number: int) -> Optional[Path]:
    patt = f"pattern_{int(layout_number):04d}_Great Hollow_under.json"
    for base_dir in GREAT_HOLLOW_UNDER_PATTERN_DIRS:
        path = base_dir / patt
        if path.exists():
            return path
    return None


def get_layout_great_hollow_under_status(layout_number: int) -> Optional[str]:
    layout_number = int(layout_number)
    if layout_number in GREAT_HOLLOW_UNDER_EXPECTED_STATUS_CACHE:
        return GREAT_HOLLOW_UNDER_EXPECTED_STATUS_CACHE[layout_number]

    pattern_path = _find_great_hollow_under_pattern_path(layout_number)
    if pattern_path is None:
        GREAT_HOLLOW_UNDER_EXPECTED_STATUS_CACHE[layout_number] = None
        return None

    try:
        payload = json.loads(pattern_path.read_text(encoding="utf-8"))
    except Exception:
        GREAT_HOLLOW_UNDER_EXPECTED_STATUS_CACHE[layout_number] = None
        return None

    grouped = {}
    for lm in payload.get("landmarks", []) or []:
        grouped.setdefault(lm.get("source_id"), []).append(lm)

    for _source_id, items in grouped.items():
        has_gc = any(str(it.get("type", "")).strip() == "Great Church" for it in items)
        if not has_gc:
            continue
        for it in items:
            t = str(it.get("type", "")).strip()
            if t in {"Status_Frost", "Status_Holy"}:
                GREAT_HOLLOW_UNDER_EXPECTED_STATUS_CACHE[layout_number] = t
                return t

    for lm in payload.get("landmarks", []) or []:
        t = str(lm.get("type", "")).strip()
        if t in {"Status_Frost", "Status_Holy"}:
            GREAT_HOLLOW_UNDER_EXPECTED_STATUS_CACHE[layout_number] = t
            return t

    GREAT_HOLLOW_UNDER_EXPECTED_STATUS_CACHE[layout_number] = None
    return None


def _gh_under_box_from_meta(h: int, w: int, meta: dict) -> Tuple[int, int, int]:
    mode = str(meta.get("box_mode", "legacy_anchor"))
    if mode == "proportional_bl_1024":
        x_bl_1024 = float(meta.get("x_bl_1024", 504.0))
        y_bl_1024 = float(meta.get("y_bl_1024", 182.0))
        crop_size_1024 = int(meta.get("crop_size_1024", 28))
        cx = int(round((x_bl_1024 / 1024.0) * float(w)))
        cy = int(round((1.0 - (y_bl_1024 / 1024.0)) * float(h)))
        side = max(12, int(round(float(crop_size_1024) * min(float(w), float(h)) / 1024.0)))
        return cx, cy, side
    crop_size = int(meta.get("crop_size", 28))
    cx = int(round(622.0 * float(w) / 1024.0))
    cy = int(round((1024.0 - 182.0) * float(h) / 1024.0))
    return cx, cy, max(12, crop_size)


def _gh_under_collect_train_samples(pattern_dir: Path, image_dir: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    for p in sorted(pattern_dir.glob("pattern_*_Great Hollow_under.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = int(obj.get("patternId", -1))
        if pid < 0:
            continue
        img_path = image_dir / f"layout_{pid}.png"
        if not img_path.exists():
            continue
        label = None
        for lm in obj.get("landmarks", []) or []:
            t = str(lm.get("type", "")).strip()
            if t == "Status_Frost":
                label = "Frost"
                break
            if t == "Status_Holy":
                label = "Holy"
                break
        if label:
            samples.append((img_path, label))
    return samples


def _gh_under_build_color_prototypes(samples: Sequence[Tuple[Path, str]], model_meta: dict) -> Optional[dict]:
    by_label: dict = {"Frost": [], "Holy": []}
    for img_path, label in samples:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy, side = _gh_under_box_from_meta(h, w, model_meta)
        patch = _gh_crop_center_with_padding(img, cx, cy, side)
        by_label[str(label)].append(_gh_under_color_desc(patch))
    if not by_label["Frost"] or not by_label["Holy"]:
        return None

    out = {}
    for lbl in ("Frost", "Holy"):
        xs = by_label[lbl]
        hist = np.mean(np.stack([np.asarray(x["hist"], dtype=np.float32) for x in xs], axis=0), axis=0).astype(np.float32)
        hist /= float(hist.sum() + 1e-6)
        out[lbl] = {
            "hist": hist,
            "mean_h": float(np.mean([float(x["mean_h"]) for x in xs])),
            "mean_s": float(np.mean([float(x["mean_s"]) for x in xs])),
            "mean_v": float(np.mean([float(x["mean_v"]) for x in xs])),
        }
    return out


def _gh_under_load_hybrid_bundle(ctx: Optional[RunContext] = None) -> Optional[dict]:
    global GREAT_HOLLOW_UNDER_HYBRID_BUNDLE, GREAT_HOLLOW_UNDER_HYBRID_LOAD_ATTEMPTED
    if GREAT_HOLLOW_UNDER_HYBRID_BUNDLE is not None:
        return GREAT_HOLLOW_UNDER_HYBRID_BUNDLE
    if GREAT_HOLLOW_UNDER_HYBRID_LOAD_ATTEMPTED:
        return None
    GREAT_HOLLOW_UNDER_HYBRID_LOAD_ATTEMPTED = True

    model_path = _resolve_first_existing_path(GREAT_HOLLOW_UNDER_MODEL_PATHS)
    pattern_dir = _resolve_first_existing_path(GREAT_HOLLOW_UNDER_PATTERN_DIRS)
    image_dir = _resolve_first_existing_path(GREAT_HOLLOW_UNDER_IMAGE_DIRS)
    if model_path is None or pattern_dir is None or image_dir is None:
        return None

    try:
        device = choose_torch_device()
        ckpt = torch.load(str(model_path), map_location=device)
        label_map = dict(ckpt.get("label_map", {"Frost": 0, "Holy": 1}))
        out_size = int(ckpt.get("out_size", 32))
        model = _GHStatusTinyCNN(num_classes=max(2, len(label_map) or 2)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        id_to_label = {int(v): str(k) for k, v in label_map.items()}
        if "crop_size_1024" in ckpt:
            model_meta = {
                "box_mode": "proportional_bl_1024",
                "crop_size_1024": int(ckpt["crop_size_1024"]),
                "x_bl_1024": float(ckpt.get("x_bl_1024", 504.0)),
                "y_bl_1024": float(ckpt.get("y_bl_1024", 182.0)),
            }
        else:
            model_meta = {
                "box_mode": "legacy_anchor",
                "crop_size": int(ckpt.get("crop_size", 28)),
            }
        samples = _gh_under_collect_train_samples(pattern_dir, image_dir)
        protos = _gh_under_build_color_prototypes(samples, model_meta)
        if protos is None:
            return None
        GREAT_HOLLOW_UNDER_HYBRID_BUNDLE = {
            "model": model,
            "device": device,
            "id_to_label": id_to_label,
            "out_size": out_size,
            "path": model_path,
            "prototypes": protos,
            "model_meta": model_meta,
            "alpha_model": float(GREAT_HOLLOW_UNDER_HYBRID_ALPHA_MODEL),
            "sharp_scale": float(GREAT_HOLLOW_UNDER_HYBRID_SHARP_SCALE),
        }
        if ctx is not None:
            ctx.log(f"Loaded GH under Frost/Holy hybrid bundle: {model_path.resolve()}")
        return GREAT_HOLLOW_UNDER_HYBRID_BUNDLE
    except Exception as exc:
        if ctx is not None:
            ctx.log(f"Warning: failed to load GH under Frost/Holy hybrid bundle: {exc}")
        GREAT_HOLLOW_UNDER_HYBRID_BUNDLE = None
        return None


def infer_great_hollow_under_status_hybrid(panel_bgr: np.ndarray, ctx: Optional[RunContext] = None) -> Optional[dict]:
    bundle = _gh_under_load_hybrid_bundle(ctx)
    if bundle is None:
        return None
    h, w = panel_bgr.shape[:2]
    cx, cy, side = _gh_under_box_from_meta(h, w, dict(bundle["model_meta"]))
    patch = _gh_crop_center_with_padding(panel_bgr, cx, cy, side)

    model_probs = _gh_model_probs_from_patch(bundle, patch)
    pf_model = float(model_probs.get("Frost", 0.5))
    ph_model = float(model_probs.get("Holy", 0.5))

    desc = _gh_under_color_desc(patch)
    proto_frost = dict(bundle["prototypes"]["Frost"])
    proto_holy = dict(bundle["prototypes"]["Holy"])
    score_frost = _gh_under_color_score(desc, proto_frost)
    score_holy = _gh_under_color_score(desc, proto_holy)
    pf_color, ph_color = _gh_under_softmax2(score_frost, score_holy, temp=0.08)

    alpha_model = float(bundle["alpha_model"])
    sharp_scale = float(bundle["sharp_scale"])
    pf_mix = (1.0 - alpha_model) * pf_color + alpha_model * pf_model
    ph_mix = (1.0 - alpha_model) * ph_color + alpha_model * ph_model
    z = float(max(1e-12, pf_mix + ph_mix))
    pf_mix /= z
    ph_mix /= z

    pf_hybrid = _gh_under_sharpen_prob(float(pf_mix), scale=sharp_scale)
    ph_hybrid = float(1.0 - pf_hybrid)
    if pf_hybrid >= ph_hybrid:
        best_label = "Status_Frost"
        best_score = float(pf_hybrid)
        second_label = "Status_Holy"
        second_score = float(ph_hybrid)
    else:
        best_label = "Status_Holy"
        best_score = float(ph_hybrid)
        second_label = "Status_Frost"
        second_score = float(pf_hybrid)
    margin = float(best_score - second_score)
    if margin >= 0.45:
        confidence = "strong"
        present_prob = 0.95
    elif margin >= 0.20:
        confidence = "medium"
        present_prob = 0.80
    else:
        confidence = "weak"
        present_prob = 0.60

    return {
        "anchor_type": "Great Church Under",
        "expected_x_norm": float(cx / max(1, w)),
        "expected_y_norm": float(cy / max(1, h)),
        "best_status": best_label,
        "best_score": float(best_score),
        "second_status": second_label,
        "second_score": float(second_score),
        "margin": float(margin),
        "present_prob": float(present_prob),
        "confidence": confidence,
        "x_norm": float(cx / max(1, w)),
        "y_norm": float(cy / max(1, h)),
        "status_box": {
            "x": int(round(cx - (side / 2.0))),
            "y": int(round(cy - (side / 2.0))),
            "w": int(side),
            "h": int(side),
        },
        "ranked": [
            {"status_type": "Status_Frost", "score": float(pf_hybrid), "dx": 0, "dy": 0},
            {"status_type": "Status_Holy", "score": float(ph_hybrid), "dx": 0, "dy": 0},
        ],
        "method": "dual_shot_gc_under_hybrid",
        "classifier_mode": "hybrid",
        "model_path": str(bundle.get("path", "")),
        "model_probs": {"Frost": float(pf_model), "Holy": float(ph_model)},
        "color_scores": {"Frost": float(score_frost), "Holy": float(score_holy)},
        "hybrid_probs": {"Status_Frost": float(pf_hybrid), "Status_Holy": float(ph_hybrid)},
    }


def detect_great_hollow_under_status_from_full_screenshot(
    full_img_bgr: np.ndarray,
    backend,
    ctx: RunContext,
    detect_map_circle_compat: Callable[[np.ndarray], Tuple[Circle, str]],
    refine_panel_box_compat: Callable[[np.ndarray, Circle], CropBox],
    normalize_panel_for_slot_recognition_compat: Callable[[np.ndarray], np.ndarray],
) -> Optional[dict]:
    circle, crop_method = detect_map_circle_compat(full_img_bgr)
    if crop_method == "blue_ring":
        crop_box = backend.clip_box(
            backend.estimate_panel_box(circle, float(getattr(backend, "DEFAULT_PANEL_SCALE", 1.12))),
            full_img_bgr.shape[1],
            full_img_bgr.shape[0],
        )
    else:
        crop_box = refine_panel_box_compat(full_img_bgr, circle)

    raw_panel = full_img_bgr[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
    panel = normalize_panel_for_slot_recognition_compat(raw_panel)
    inferred = infer_great_hollow_under_status_hybrid(panel, ctx)
    if inferred is None:
        return None
    inferred["crop_method"] = str(crop_method)
    inferred["panel_shape"] = [int(panel.shape[1]), int(panel.shape[0])]
    inferred["crop_box"] = {"x0": int(crop_box.x0), "y0": int(crop_box.y0), "x1": int(crop_box.x1), "y1": int(crop_box.y1)}
    return inferred


def compute_great_hollow_under_status_prior(layout_number: int, observed_status: Optional[dict]) -> Tuple[float, dict]:
    if observed_status is None:
        return 0.0, {"great_hollow_under_status_prior": 0.0, "status_evidence": []}
    expected_status = get_layout_great_hollow_under_status(int(layout_number))
    if expected_status not in {"Status_Frost", "Status_Holy"}:
        return 0.0, {"great_hollow_under_status_prior": 0.0, "status_evidence": []}

    observed_best = str(observed_status.get("best_status", ""))
    best_score = float(observed_status.get("best_score", 0.0))
    margin = float(observed_status.get("margin", 0.0))
    score_strength = max(0.0, min(1.0, (best_score - 0.50) / 0.35))
    margin_strength = max(0.0, min(1.0, margin / 0.25))
    confidence = 0.55 * score_strength + 0.45 * margin_strength

    if observed_best == expected_status:
        slot_prior = 1.35 * (0.035 + 0.085 * confidence)
    else:
        slot_prior = 1.35 * (-(0.030 + 0.110 * confidence))

    evidence = [{
        "anchor_type": "Great Church Under",
        "expected_status": str(expected_status),
        "observed_status": str(observed_best),
        "best_score": float(best_score),
        "margin": float(margin),
        "confidence": float(confidence),
        "slot_prior": float(slot_prior),
    }]
    return float(slot_prior), {
        "great_hollow_under_status_prior": float(slot_prior),
        "status_evidence": evidence,
    }


def preprocess_panel_for_status_recognizer(panel_bgr: np.ndarray) -> np.ndarray:
    # Keep GH status recognizer input aligned with earth-selector preprocessing contract (1536 -> crop 256 -> 1024).
    return preprocess_earth_selector_panel(
        panel_bgr,
        input_size=EARTH_SELECTOR_INPUT_SIZE,
        border_crop_px=EARTH_SELECTOR_BORDER_CROP_PX,
    )


def preprocess_panel_for_status_recognizer_great_hollow(raw_panel_bgr: np.ndarray) -> np.ndarray:
    # Great Hollow test path: restart from the initial raw panel crop and apply an asymmetric inner crop.
    resized = cv2.resize(
        raw_panel_bgr,
        (GREAT_HOLLOW_STATUS_INPUT_SIZE, GREAT_HOLLOW_STATUS_INPUT_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    left = int(GREAT_HOLLOW_STATUS_CROP_LEFT_PX)
    right = int(GREAT_HOLLOW_STATUS_CROP_RIGHT_PX)
    top = int(GREAT_HOLLOW_STATUS_CROP_TOP_PX)
    bottom = int(GREAT_HOLLOW_STATUS_CROP_BOTTOM_PX)

    y0 = max(0, top)
    y1 = min(GREAT_HOLLOW_STATUS_INPUT_SIZE, GREAT_HOLLOW_STATUS_INPUT_SIZE - bottom)
    x0 = max(0, left)
    x1 = min(GREAT_HOLLOW_STATUS_INPUT_SIZE, GREAT_HOLLOW_STATUS_INPUT_SIZE - right)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Invalid Great Hollow asymmetric status crop bounds.")
    return resized[y0:y1, x0:x1].copy()


def detect_great_hollow_statuses(
    panel_bgr: np.ndarray,
    package: SlotRecognitionPackage,
    observations_by_earth: dict,
    raw_panel_bgr: Optional[np.ndarray] = None,
    earth_rank: Optional[Sequence[Tuple[str, float, Dict[str, float]]]] = None,
) -> dict:
    recognizer = _load_icon_recognizer_v4_module()
    if recognizer is None:
        return {}

    ref1 = _resolve_first_existing_path(GREAT_HOLLOW_FIXED_STATUS_REFS["ref1"])
    ref2 = _resolve_first_existing_path(GREAT_HOLLOW_FIXED_STATUS_REFS["ref2"])
    if ref1 is None or ref2 is None:
        return {}

    label_map = {
        "Fire": "Status_Fire",
        "Magic": "Status_Magic",
        "Rot": "Status_Rot",
        "Holy": "Status_Holy",
        "Poison": "Status_Poison",
        "Bleed": "Status_Bleed",
        "Lightning": "Status_Lightning",
        "Sleep": "Status_Sleep",
    }
    position_map = {
        "Palace": "top_left",
        "Fortress": "bottom_right",
        "Ruins": "ruins",
        "Great Church": "great_church_top_right",
    }
    confidence_to_present_prob = {
        "strong": 0.95,
        "medium": 0.80,
        "weak": 0.60,
    }

    primary_earth = str(earth_rank[0][0]) if earth_rank else ""
    use_great_hollow_alt = primary_earth == "Great Hollow" and raw_panel_bgr is not None and raw_panel_bgr.size > 0
    status_mode = _gh_status_classifier_mode()

    tmp_query_path = None
    tmp_ref1_path = None
    tmp_ref2_path = None
    if use_great_hollow_alt:
        status_panel = preprocess_panel_for_status_recognizer_great_hollow(raw_panel_bgr)
        ref_preprocess_fn = preprocess_panel_for_status_recognizer_great_hollow
    else:
        status_panel = preprocess_panel_for_status_recognizer(panel_bgr)
        ref_preprocess_fn = preprocess_panel_for_status_recognizer
    h, w = status_panel.shape[:2]
    old_env = {}
    try:
        if use_great_hollow_alt:
            env_updates = {
                "ICON_RECOGNIZER_LEGACY_CROP_LEFT": str(int(GREAT_HOLLOW_STATUS_CROP_LEFT_PX)),
                "ICON_RECOGNIZER_LEGACY_CROP_RIGHT": str(int(GREAT_HOLLOW_STATUS_CROP_RIGHT_PX)),
                "ICON_RECOGNIZER_LEGACY_CROP_TOP": str(int(GREAT_HOLLOW_STATUS_CROP_TOP_PX)),
                "ICON_RECOGNIZER_LEGACY_CROP_BOTTOM": str(int(GREAT_HOLLOW_STATUS_CROP_BOTTOM_PX)),
            }
            for k, v in env_updates.items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = v

        ref1_img = cv2.imread(str(ref1), cv2.IMREAD_COLOR)
        ref2_img = cv2.imread(str(ref2), cv2.IMREAD_COLOR)
        if ref1_img is None or ref2_img is None:
            return {}

        ref1_status = ref_preprocess_fn(ref1_img)
        ref2_status = ref_preprocess_fn(ref2_img)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_q:
            tmp_query_path = Path(tmp_q.name)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_r1:
            tmp_ref1_path = Path(tmp_r1.name)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_r2:
            tmp_ref2_path = Path(tmp_r2.name)

        cv2.imwrite(str(tmp_query_path), status_panel)
        cv2.imwrite(str(tmp_ref1_path), ref1_status)
        cv2.imwrite(str(tmp_ref2_path), ref2_status)
        raw_result = recognizer.recognize_markers(str(tmp_ref1_path), str(tmp_ref2_path), str(tmp_query_path))

        # Optional model path for GH statuses: Palace/Fortress/Great Church.
        if primary_earth == "Great Hollow" and status_mode == "models":
            model_anchor_to_pos = {
                "Palace": "top_left",
                "Fortress": "bottom_right",
                "Ruins": "ruins",
                "Great Church": "great_church_top_right",
            }
            for anchor_type, pos_key in model_anchor_to_pos.items():
                slot = raw_result.get(pos_key, {})
                box = slot.get("box", {})
                bx = int(round(float(box.get("x", 0.0))))
                by = int(round(float(box.get("y", 0.0))))
                bw = int(round(float(box.get("w", 0.0))))
                bh = int(round(float(box.get("h", 0.0))))
                side = max(12, max(bw, bh))
                patch = _gh_crop_with_padding(status_panel, bx, by, side)

                bundle = _gh_load_status_tiny_model(anchor_type)
                if bundle is None:
                    continue
                probs = _gh_model_probs_from_patch(bundle, patch)
                allowed_status_types = list(GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE.get(anchor_type, ()))
                allowed_labels = [s.replace("Status_", "") for s in allowed_status_types]
                scored = sorted(
                    [(lbl, float(probs.get(lbl, 0.0))) for lbl in allowed_labels],
                    key=lambda t: t[1],
                    reverse=True,
                )
                if not scored:
                    continue
                winner, winner_score = scored[0]
                if len(scored) > 1:
                    runner, runner_score = scored[1]
                else:
                    runner, runner_score = "", 0.0
                margin = float(winner_score - runner_score)
                if margin >= 0.30:
                    conf = "strong"
                elif margin >= 0.12:
                    conf = "medium"
                else:
                    conf = "weak"
                slot["label"] = str(winner)
                slot["score"] = float(winner_score)
                slot["runner_up"] = str(runner)
                slot["runner_up_score"] = float(runner_score)
                slot["margin"] = float(margin)
                slot["confidence"] = str(conf)
                slot["ranked"] = [
                    {"label": str(lbl), "score": float(scr), "dx": 0, "dy": 0}
                    for lbl, scr in scored
                ]
                slot["classifier_mode"] = "models"
                slot["model_path"] = str(bundle.get("path", ""))
                slot["model_probs"] = {str(k): float(v) for k, v in probs.items()}
                raw_result[pos_key] = slot
    except Exception:
        return {}
    finally:
        for k, old_v in old_env.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v
        for tmp_path in (tmp_query_path, tmp_ref1_path, tmp_ref2_path):
            if tmp_path is None:
                continue
            try:
                tmp_path.unlink()
            except Exception:
                pass

    observed = {}
    for anchor_type, pos_key in position_map.items():
        slot = raw_result.get(pos_key, {})
        best_status = label_map.get(str(slot.get("label", "")))
        if not best_status:
            continue
        second_status = label_map.get(str(slot.get("runner_up", "")))
        best_score = float(slot.get("score", 0.0))
        second_score = float(slot.get("runner_up_score", 0.0))
        margin = float(slot.get("margin", best_score - second_score))
        conf_band = str(slot.get("confidence", "weak")).lower()
        present_prob = float(confidence_to_present_prob.get(conf_band, 0.60))
        box = slot.get("box", {})
        bx = float(box.get("x", 0.0))
        by = float(box.get("y", 0.0))
        bw = float(box.get("w", 0.0))
        bh = float(box.get("h", 0.0))
        cfg = GREAT_HOLLOW_FIXED_STATUS_CONFIG.get(anchor_type, {})
        xy_norm = cfg.get("xy_norm", (0.0, 0.0))

        ranked = []
        for item in slot.get("ranked", []):
            mapped = label_map.get(str(item.get("label", "")))
            if not mapped:
                continue
            ranked.append({
                "status_type": str(mapped),
                "score": float(item.get("score", 0.0)),
                "dx": int(item.get("dx", 0)),
                "dy": int(item.get("dy", 0)),
            })

        observed[anchor_type] = {
            "anchor_type": anchor_type,
            "expected_x_norm": float(xy_norm[0]),
            "expected_y_norm": float(xy_norm[1]),
            "best_status": str(best_status),
            "best_score": float(best_score),
            "second_status": str(second_status) if second_status else None,
            "second_score": float(second_score),
            "margin": float(margin),
            "present_prob": float(present_prob),
            "x_norm": float((bx + 0.5 * bw) / max(1, w)) if bw > 0 else float(xy_norm[0]),
            "y_norm": float((by + 0.5 * bh) / max(1, h)) if bh > 0 else float(xy_norm[1]),
            "method": (
                ("icon_recognizer_v4_gh_raw_alt" if use_great_hollow_alt else "icon_recognizer_v4")
                + (
                    f":{str(slot.get('classifier_mode', 'classic'))}"
                    if anchor_type in {"Palace", "Fortress", "Ruins", "Great Church"}
                    else ""
                )
            ),
            "classifier_mode": str(slot.get("classifier_mode", "classic")),
            "ranked": ranked,
        }
    return observed


def compute_great_hollow_status_prior(layout: dict, observed_statuses: dict) -> Tuple[float, dict]:
    if str(layout.get("shifting_earth", "")) != "Great Hollow":
        return 0.0, {"great_hollow_status_prior": 0.0, "status_evidence": []}
    expected_statuses = get_layout_great_hollow_statuses(layout)
    if not expected_statuses:
        return 0.0, {"great_hollow_status_prior": 0.0, "status_evidence": []}

    prior = 0.0
    evidence = []
    for anchor_type, expected in expected_statuses.items():
        observed = observed_statuses.get(anchor_type)
        if observed is None:
            continue
        allowed_statuses = tuple(str(s) for s in GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE.get(anchor_type, ()))
        best_score = float(observed.get("best_score", 0.0))
        margin = float(observed.get("margin", 0.0))
        score_strength = max(0.0, min(1.0, (best_score - 0.42) / 0.28))
        margin_strength = max(0.0, min(1.0, margin / 0.12))
        confidence = 0.55 * score_strength + 0.45 * margin_strength
        expected_status = str(expected.get("status_type", ""))
        observed_status = str(observed.get("best_status", ""))
        if observed_status == expected_status:
            slot_prior = 0.035 + 0.085 * confidence
        elif observed_status in allowed_statuses:
            slot_prior = -(0.030 + 0.110 * confidence)
        else:
            slot_prior = -0.180 - 0.100 * confidence
        prior += slot_prior
        evidence.append({
            "anchor_type": anchor_type,
            "allowed_statuses": list(allowed_statuses),
            "expected_status": expected_status,
            "observed_status": observed_status,
            "best_score": best_score,
            "margin": margin,
            "confidence": float(confidence),
            "slot_prior": float(slot_prior),
        })
    return float(prior), {
        "great_hollow_status_prior": float(prior),
        "status_evidence": evidence,
    }


def _gh_expected_slot_label(layout: dict, package: SlotRecognitionPackage, slot_id: int) -> Optional[str]:
    labels = list(layout.get("slot_labels", []) or [])
    gh_slots = list(package.slots_by_earth.get("Great Hollow", []) or [])
    for idx, slot in enumerate(gh_slots):
        if int(slot.get("slot_id", -1)) != int(slot_id):
            continue
        if idx < len(labels):
            return str(labels[idx])
        return None
    return None


def _gh_strength_binary(best_prob: float, margin: float) -> float:
    p = max(0.0, min(1.0, (float(best_prob) - 0.50) / 0.50))
    m = max(0.0, min(1.0, float(margin) / 0.35))
    return float(0.65 * p + 0.35 * m)


def _gh_strength_multiclass(best_prob: float, second_prob: float, class_count: int) -> float:
    base = 1.0 / max(1, int(class_count))
    p = max(0.0, min(1.0, (float(best_prob) - base) / max(1e-6, 1.0 - base)))
    m = max(0.0, min(1.0, (float(best_prob) - float(second_prob)) / 0.60))
    return float(0.55 * p + 0.45 * m)


def compute_great_hollow_model_layout_prior(
    layout: dict,
    package: SlotRecognitionPackage,
    model_observations: dict,
) -> Tuple[float, dict]:
    if str(layout.get("shifting_earth", "")) != "Great Hollow":
        return 0.0, {"great_hollow_model_layout_prior": 0.0, "model_evidence": []}
    if not model_observations:
        return 0.0, {"great_hollow_model_layout_prior": 0.0, "model_evidence": []}

    expected_statuses = get_layout_great_hollow_statuses(layout)
    prior = 0.0
    evidence = []

    for spec_key, obs in model_observations.items():
        kind = str(obs.get("kind", ""))
        weight = float(obs.get("weight", 0.0))
        if weight <= 0.0:
            continue

        if kind == "status":
            anchor_type = str(obs.get("anchor_type", ""))
            expected = expected_statuses.get(anchor_type, {})
            expected_status = str(expected.get("status_type", "")).strip()
            predicted_status = str(obs.get("predicted_status", "")).strip()
            if not expected_status or not predicted_status:
                continue
            strength = _gh_strength_binary(float(obs.get("best_prob", 0.0)), float(obs.get("margin", 0.0)))
            magnitude = float(weight) * (0.30 + 0.70 * strength)
            match = predicted_status == expected_status
            slot_prior = float(magnitude if match else -magnitude)
            prior += slot_prior
            evidence.append({
                "spec_key": str(spec_key),
                "kind": "status",
                "anchor_type": anchor_type,
                "expected": expected_status,
                "predicted": predicted_status,
                "best_prob": float(obs.get("best_prob", 0.0)),
                "margin": float(obs.get("margin", 0.0)),
                "strength": float(strength),
                "weight": float(weight),
                "slot_prior": float(slot_prior),
            })
            continue

        if kind == "slot_label":
            slot_id = int(obs.get("slot_id", -1))
            expected_label = _gh_expected_slot_label(layout, package, slot_id)
            if expected_label is None:
                continue
            expected_canon = _gh_canonical_slot_label(expected_label)
            predicted_canon = _gh_canonical_slot_label(str(obs.get("predicted_slot_label", "")))
            if not expected_canon or not predicted_canon:
                continue
            class_count = len(list(obs.get("class_names", []) or []))
            strength = _gh_strength_multiclass(
                float(obs.get("best_prob", 0.0)),
                float(obs.get("second_prob", 0.0)),
                class_count=max(2, class_count),
            )
            magnitude = float(weight) * (0.30 + 0.70 * strength)
            match = predicted_canon == expected_canon
            slot_prior = float(magnitude if match else -magnitude)
            prior += slot_prior
            evidence.append({
                "spec_key": str(spec_key),
                "kind": "slot_label",
                "slot_id": int(slot_id),
                "expected": expected_canon,
                "predicted": predicted_canon,
                "best_prob": float(obs.get("best_prob", 0.0)),
                "margin": float(obs.get("margin", 0.0)),
                "strength": float(strength),
                "weight": float(weight),
                "slot_prior": float(slot_prior),
            })

    return float(prior), {
        "great_hollow_model_layout_prior": float(prior),
        "model_evidence": evidence,
    }


def compute_great_hollow_family_support(observed_statuses: dict) -> dict:
    if not observed_statuses:
        return {"support": 0.0, "anchor_count": 0, "anchors": []}

    anchors = []
    total = 0.0
    for anchor_type, info in observed_statuses.items():
        best_score = float(info.get("best_score", 0.0))
        margin = float(info.get("margin", 0.0))
        present_prob = float(info.get("present_prob", 0.0))
        score_strength = max(0.0, min(1.0, (best_score - 0.42) / 0.30))
        margin_strength = max(0.0, min(1.0, margin / 0.08))
        anchor_support = 0.45 * present_prob + 0.35 * margin_strength + 0.20 * score_strength
        total += anchor_support
        anchors.append({
            "anchor_type": str(anchor_type),
            "best_status": str(info.get("best_status", "")),
            "best_score": float(best_score),
            "margin": float(margin),
            "present_prob": float(present_prob),
            "anchor_support": float(anchor_support),
        })

    return {
        "support": float(total / max(1, len(anchors))),
        "anchor_count": int(len(anchors)),
        "anchors": anchors,
    }


def rank_shifting_earths_cascade(
    earth_rank: Sequence[Tuple[str, float, dict]],
    geometry_priors: dict,
    lost_slot_priors: dict,
    hard_constraint_priors: dict,
    evergaol_priors: dict,
    great_hollow_status_observations: dict,
) -> List[Tuple[str, float, dict]]:
    base_by_earth = {str(name): (float(score), dict(details)) for name, score, details in earth_rank}
    gh_support = compute_great_hollow_family_support(great_hollow_status_observations)

    ranked = []
    for earth_name, (base_score, base_details) in base_by_earth.items():
        geometry_prior = float(geometry_priors.get(earth_name, {}).get("geometry_prior", 0.0))
        lost_slot_prior = float(lost_slot_priors.get(earth_name, {}).get("lost_slot_prior", 0.0))
        hard_constraint_prior = float(hard_constraint_priors.get(earth_name, {}).get("hard_constraint_prior", 0.0))
        evergaol_prior = float(evergaol_priors.get(earth_name, {}).get("evergaol_prior", 0.0))
        gh_status_support = float(gh_support.get("support", 0.0)) if earth_name == "Great Hollow" else 0.0

        if earth_name == "Great Hollow":
            cascade_score = (
                1.35 * base_score
                + 1.10 * geometry_prior
                + 1.45 * hard_constraint_prior
                + 1.15 * evergaol_prior
                + 0.10 * gh_status_support
            )
        else:
            structural_score = geometry_prior + 1.35 * lost_slot_prior + 1.35 * evergaol_prior
            cascade_score = 1.25 * base_score + structural_score

        ranked.append((
            earth_name,
            float(cascade_score),
            {
                "base_score": float(base_score),
                "geometry_prior": float(geometry_prior),
                "lost_slot_prior": float(lost_slot_prior),
                "hard_constraint_prior": float(hard_constraint_prior),
                "evergaol_prior": float(evergaol_prior),
                "great_hollow_status_support": float(gh_status_support),
                "raw_details": base_details,
            },
        ))

    by_earth = {name: {"score": score, "details": details} for name, score, details in ranked}
    gh_info = by_earth.get("Great Hollow")
    if gh_info is not None:
        gh_base = float(gh_info["details"].get("base_score", 0.0))
        best_non_gh_base = max(
            (float(item["details"].get("base_score", 0.0)) for name, item in by_earth.items() if name != "Great Hollow"),
            default=float("-inf"),
        )
        best_non_gh = max((item["score"] for name, item in by_earth.items() if name != "Great Hollow"), default=float("-inf"))
        gh_raw_dominance = gh_base >= 0.80 and gh_base >= best_non_gh_base + 0.25
        gh_gate_structural = (
            float(gh_info["score"]) >= float(best_non_gh) + 0.015
            and float(gh_info["details"].get("hard_constraint_prior", 0.0)) >= -0.18
        )
        gh_gate_pass = bool(gh_raw_dominance or gh_gate_structural)
        gh_info["details"]["raw_dominance"] = bool(gh_raw_dominance)
        gh_info["details"]["gate_structural"] = bool(gh_gate_structural)
        gh_info["details"]["gate_pass"] = bool(gh_gate_pass)
        if gh_gate_pass:
            if gh_raw_dominance:
                boosted_score = max(
                    float(gh_info["score"]) + 0.75,
                    float(best_non_gh) + 0.05,
                    gh_base + 0.05,
                )
            else:
                boosted_score = float(gh_info["score"]) + 0.050
            by_earth["Great Hollow"] = {"score": float(boosted_score), "details": gh_info["details"]}

    ranked = [
        (name, float(info["score"]), dict(info["details"]))
        for name, info in by_earth.items()
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def format_layout_debug_lines(result: dict, max_slots: int = 5) -> List[str]:
    details = dict(result.get("details", {}))
    lines = [
        (
            f"layout #{int(result['layout_number'])} earth={result['shifting_earth']} "
            f"total={float(result['total_score']):.6f} normalized={float(details.get('normalized_log_prob', 0.0)):.6f} "
            f"earth={float(result.get('earth_score', 0.0)):.6f} geometry={float(result.get('geometry_prior', 0.0)):+.4f} "
            f"lost={float(result.get('lost_slot_prior', 0.0)):+.4f} hard={float(result.get('hard_constraint_prior', 0.0)):+.4f} "
            f"evergaol={float(result.get('evergaol_prior', 0.0)):+.4f} "
            f"gh_status={float(result.get('great_hollow_status_prior', 0.0)):+.4f} "
            f"gh_models={float(result.get('great_hollow_model_layout_prior', 0.0)):+.4f}"
        ),
        (
            f"  exact_w={float(details.get('exact_match_weight', 0.0)):.2f} "
            f"mismatch_w={float(details.get('hard_mismatch_weight', 0.0)):.2f} "
            f"unknown_w={float(details.get('unknown_weight', 0.0)):.2f}"
        ),
    ]
    weakest = sorted(
        list(details.get("slot_details", [])),
        key=lambda item: (float(item.get("expected_prob", 0.0)), float(item.get("weighted_log_prob", 0.0))),
    )[:max_slots]
    if weakest:
        lines.append("  weakest expected slots:")
        for item in weakest:
            lines.append(
                "    "
                f"{item['slot_name']} exp={item['expected']} obs={item['observed']} "
                f"p={float(item['expected_prob']):.4f} best={float(item['best_prob']):.4f} "
                f"margin={float(item['margin']):.4f}"
            )
    return lines


def draw_ai_debug_image(panel_bgr: np.ndarray, slots_out: Sequence[dict]) -> np.ndarray:
    out = panel_bgr.copy()
    h, w = out.shape[:2]
    side = min(h, w)
    for slot_obs in slots_out:
        cx = int(round(float(slot_obs["nominal_x_norm"]) * w))
        cy = int(round(float(slot_obs["nominal_y_norm"]) * h))
        px = int(round(float(slot_obs["x_norm"]) * w))
        py = int(round(float(slot_obs["y_norm"]) * h))
        r = max(6, int(round(float(slot_obs["radius_norm"]) * side)))

        cv2.circle(out, (cx, cy), r, (255, 255, 0), 1)
        label = str(slot_obs["label"])
        if label == "unknown":
            color = (0, 0, 255)
        elif label == "empty":
            color = (180, 180, 180)
        else:
            color = (0, 255, 0)
        cv2.circle(out, (px, py), 5, color, -1)

        top_probs = sorted(slot_obs["probs"].items(), key=lambda kv: kv[1], reverse=True)[:2]
        if len(top_probs) >= 2:
            text = f"{slot_obs['slot_name']}: {label} ({top_probs[0][0]} {top_probs[0][1]:.2f} / {top_probs[1][0]} {top_probs[1][1]:.2f})"
        else:
            text = f"{slot_obs['slot_name']}: {label}"

        tx = max(6, min(w - 320, px + 6))
        ty = max(16, min(h - 8, py - 6))
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    return out


def run_recognition(
    ctx: RunContext,
    monitor_index: int = 1,
    preview_callback: Optional[Callable[[np.ndarray], None]] = None,
    pre_captured_full_img: Optional[np.ndarray] = None,
    second_full_img: Optional[np.ndarray] = None,
    dual_shot_enabled: bool = False,
    dynamic_render: bool = True,
    dynamic_render_only: bool = True,
) -> dict:
    import tempfile

    total_start = time.perf_counter()

    disp_dir = choose_existing_display_dir()
    if disp_dir is None and not dynamic_render_only:
        checked = "\n".join(f"- {p}" for p in display_search_paths())
        raise FileNotFoundError(
            "Display directory not found. Checked these locations:\n" + checked
        )

    if disp_dir is not None:
        ctx.log(f"Display directory: {disp_dir.resolve()}")
    else:
        ctx.log("Display directory: none (render-only mode)")
    ctx.log(f"Log file: {log_file_path().resolve()}")

    selected_monitor = None
    for idx, monitor in get_available_monitors():
        if idx == monitor_index:
            selected_monitor = monitor
            break
    if selected_monitor is None:
        raise RuntimeError(f"Selected display {monitor_index} is no longer available.")
    ctx.log(f"Selected display: {monitor_label(monitor_index, selected_monitor)}")

    backend = load_proto_backend_module(ctx)
    tuned_cfg = load_tuned_config(ctx)
    apply_config_to_module(backend, tuned_cfg)
    package_path = find_slot_package_path(ctx)
    package_cls = getattr(backend, "SlotRecognitionPackage", None)
    if package_cls is None:
        package_cls = getattr(backend, "RecognitionPackage", None)
    if package_cls is None:
        raise AttributeError(
            f"module '{backend.__name__}' has no attribute 'SlotRecognitionPackage' or 'RecognitionPackage'"
        )
    package = package_cls(package_path)
    ai_bundle = load_ai_model_bundle(ctx)
    load_earth_model_bundle(ctx)

    ai_confidence_threshold = float(tuned_cfg.get("ai_confidence_threshold", 0.55))
    ai_margin_threshold = float(tuned_cfg.get("ai_margin_threshold", 0.08))
    ctx.log(f"Using AI confidence_threshold={ai_confidence_threshold:.3f}")
    ctx.log(f"Using AI margin_threshold={ai_margin_threshold:.3f}")
    ctx.log("Using trained AI slot classifier for slot recognition.")
    backend.log = lambda msg: ctx.log(str(msg))
    timings = {}

    def _time_start() -> float:
        return time.perf_counter()

    def _time_end(name: str, start_ts: float) -> None:
        timings[str(name)] = float(max(0.0, time.perf_counter() - start_ts))

    def detect_map_circle_compat(img_bgr: np.ndarray):
        region = getattr(backend, "DEFAULT_HOUGH_REGION", "right_half")
        scale = float(getattr(backend, "DEFAULT_HOUGH_SCALE", 1.0))
        try:
            return backend.detect_map_circle(img_bgr, region, scale, ctx)
        except TypeError:
            try:
                return backend.detect_map_circle(img_bgr, region, scale)
            except TypeError:
                return backend.detect_map_circle(img_bgr)

    def refine_panel_box_compat(img_bgr: np.ndarray, circle_obj):
        panel_scale = float(getattr(backend, "DEFAULT_PANEL_SCALE", 1.12))
        search_radius = int(getattr(backend, "DEFAULT_SEARCH_RADIUS", 80))
        pad = int(getattr(backend, "DEFAULT_PAD", 0))
        try:
            return backend.refine_panel_box(img_bgr, circle_obj, panel_scale, search_radius, pad, ctx)
        except TypeError:
            try:
                return backend.refine_panel_box(img_bgr, circle_obj, panel_scale, search_radius, pad)
            except TypeError:
                return backend.refine_panel_box(img_bgr, circle_obj)

    def normalize_panel_for_slot_recognition_compat(raw_panel_bgr: np.ndarray):
        fn = getattr(backend, "normalize_panel_for_slot_recognition", None)
        if fn is None:
            fn = getattr(backend, "normalize_panel_for_recognition", None)
        if fn is None:
            raise AttributeError(
                f"module '{backend.__name__}' has neither normalize_panel_for_slot_recognition nor normalize_panel_for_recognition"
            )
        return fn(raw_panel_bgr)

    def refine_crop_box_by_earth_score_slot_compat(full_img_bgr: np.ndarray, initial_box, package_obj):
        fn = getattr(backend, "refine_crop_box_by_earth_score_slot", None)
        if fn is not None:
            try:
                return fn(full_img_bgr, initial_box, package_obj, ctx)
            except TypeError:
                return fn(full_img_bgr, initial_box, package_obj)
        fn = getattr(backend, "refine_crop_box_by_earth_score", None)
        if fn is not None:
            try:
                return fn(full_img_bgr, initial_box, package_obj, ctx)
            except TypeError:
                return fn(full_img_bgr, initial_box, package_obj)
        raise AttributeError(
            f"module '{backend.__name__}' has neither refine_crop_box_by_earth_score_slot nor refine_crop_box_by_earth_score"
        )

    with tempfile.TemporaryDirectory(prefix="nightreign_gui_ai_") as td:
        td_path = Path(td)
        crop_out = td_path / "recognition_panel_used.png"
        ai_debug_out = td_path / "recognition_panel_used_with_slot_labels_ai.png"
        evergaol_debug_out = td_path / "recognition_panel_used_with_evergaol_debug.png"
        screenshot_path = td_path / "captured_full_screenshot.png"

        t_stage = _time_start()
        if pre_captured_full_img is None:
            t_capture = step_start(ctx, "Capture screenshot for preview")
            full_img = backend.capture_monitor_bgr(monitor_index)
            step_end(ctx, "Capture screenshot for preview", t_capture)
        else:
            full_img = pre_captured_full_img.copy()
            ctx.log("Using pre-captured first screenshot from dual-shot trigger.")
        _time_end("capture_screenshot_or_use_pre_captured", t_stage)
        cv2.imwrite(str(screenshot_path), full_img)

        preview_panel = None
        t_stage = _time_start()
        try:
            crop_circle, crop_method = detect_map_circle_compat(full_img)
            if crop_method == "blue_ring":
                preview_box = backend.clip_box(
                    backend.estimate_panel_box(crop_circle, float(getattr(backend, "DEFAULT_PANEL_SCALE", 1.12))),
                    full_img.shape[1],
                    full_img.shape[0],
                )
                preview_panel = full_img[preview_box.y0:preview_box.y1, preview_box.x0:preview_box.x1].copy()
            else:
                preview_box = refine_panel_box_compat(full_img, crop_circle)
                preview_panel = full_img[preview_box.y0:preview_box.y1, preview_box.x0:preview_box.x1].copy()
        except Exception as exc:
            ctx.log(f"Warning: failed to build initial crop preview: {exc}")
            preview_panel = None
        _time_end("panel_crop_preview_initial", t_stage)

        if preview_callback is not None and preview_panel is not None:
            t_stage = _time_start()
            try:
                preview_callback(preview_panel.copy())
                ctx.log("Displayed captured map preview (initial crop).")
            except Exception as exc:
                ctx.log(f"Warning: failed to show preview image: {exc}")
            _time_end("preview_callback", t_stage)
        else:
            timings["preview_callback"] = 0.0

        t_rec = step_start(ctx, "Run AI slot recognizer")

        crop_box_obj = None
        crop_refine_info = None

        t_stage = _time_start()
        circle, crop_method = detect_map_circle_compat(full_img)
        _time_end("detect_map_circle_main", t_stage)
        if crop_method == "blue_ring":
            t_stage = _time_start()
            crop_box = backend.clip_box(
                backend.estimate_panel_box(circle, float(getattr(backend, "DEFAULT_PANEL_SCALE", 1.12))),
                full_img.shape[1],
                full_img.shape[0],
            )
            raw_panel_bgr = full_img[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
            _time_end("panel_crop_first_pass", t_stage)
            t_stage = _time_start()
            panel_bgr = normalize_panel_for_slot_recognition_compat(raw_panel_bgr)
            _time_end("panel_further_crop_normalize", t_stage)
            t_stage = _time_start()
            earth_rank = classify_shifting_earth_ai(raw_panel_bgr, panel_bgr, package, ctx)
            _time_end("earth_selector_ranking", t_stage)
            crop_box_obj = CropBox(crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1)
        else:
            t_stage = _time_start()
            initial_box = refine_panel_box_compat(full_img, circle)
            _time_end("panel_crop_first_pass", t_stage)
            t_stage = _time_start()
            crop_box, raw_panel_bgr, panel_bgr, metrics, _earth_rank_unused, crop_refine_info = refine_crop_box_by_earth_score_slot_compat(
                full_img, initial_box, package
            )
            _time_end("panel_further_crop_normalize", t_stage)
            crop_method = "gray_hough_refined"
            crop_box_obj = CropBox(crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1)
            t_stage = _time_start()
            earth_rank = classify_shifting_earth_ai(raw_panel_bgr, panel_bgr, package, ctx)
            _time_end("earth_selector_ranking", t_stage)

        save_bgr(crop_out, panel_bgr)

        ctx.log("Shifting Earth ranking:")
        for name, score, details in earth_rank[:5]:
            low_corr = float(details.get("low_corr", 0.0))
            grad_corr = float(details.get("grad_corr", 0.0))
            source = details.get("source", "legacy")
            ctx.log(f"  {name:<14} score={score:.6f} low_corr={low_corr:.4f} grad_corr={grad_corr:.4f} source={source}")

        earths_for_observation = [name for name, _score, _details in earth_rank]
        if USE_EARTH_CASCADE_SELECTION:
            if earths_for_observation:
                ctx.log(f"Evaluating structural earth cascade across {len(earths_for_observation)} families.")
        else:
            earths_for_observation = earths_for_observation[:1]
            if earths_for_observation:
                ctx.log(
                    "Earth cascade selection is disabled by default; "
                    "using raw top Shifting Earth for family selection."
                )

        all_results: List[dict] = []
        best_slots_debug = None
        best_debug_earth = None
        observations_by_earth: dict = {}
        slot_obs_total = 0.0

        for earth_name in earths_for_observation:
            earth_score = next(score for name, score, _ in earth_rank if name == earth_name)
            t_stage = _time_start()
            slots_out, align = ai_classify_slots_for_earth(
                panel_bgr=panel_bgr,
                backend=backend,
                package=package,
                earth_name=earth_name,
                ai_bundle=ai_bundle,
                confidence_threshold=ai_confidence_threshold,
                margin_threshold=ai_margin_threshold,
            )
            slot_obs_total += max(0.0, time.perf_counter() - t_stage)
            unknown_count = sum(1 for s in slots_out if s["label"] == "unknown")
            ctx.log(f"AI slots for earth {earth_name}: dx={align['dx_norm']:+.4f} dy={align['dy_norm']:+.4f} resp={align['align_response']:.4f} unknown={unknown_count}/{len(slots_out)}")

            observations_by_earth[earth_name] = {"align": align, "slots": slots_out}
        timings["slot_observation_classification_total"] = float(slot_obs_total)

        t_stage = _time_start()
        geometry_priors = compute_earth_geometry_priors(package, observations_by_earth)
        _time_end("geometry_prior_computation", t_stage)
        t_stage = _time_start()
        lost_slot_priors = compute_lost_limveld_slot_priors(package, observations_by_earth)
        _time_end("lost_slot_prior_computation", t_stage)
        t_stage = _time_start()
        hard_constraint_priors = compute_hard_constraint_priors(package, observations_by_earth)
        _time_end("hard_constraint_prior_computation", t_stage)
        t_stage = _time_start()
        evergaol_priors = compute_evergaol_priors(panel_bgr, observations_by_earth)
        _time_end("evergaol_prior_detection", t_stage)
        t_stage = _time_start()
        if _gh_use_parallel_panel_classifier() and _gh_layout_model_match_mode() == "models":
            great_hollow_status_observations = {}
            timings["status_icon_detection"] = 0.0
            if ctx is not None:
                ctx.log("Skipping Great Hollow status icon detection because parallel panel classifier + GH model-match mode are active.")
        else:
            great_hollow_status_observations = detect_great_hollow_statuses(
                panel_bgr,
                package,
                observations_by_earth,
                raw_panel_bgr=raw_panel_bgr,
                earth_rank=earth_rank,
            )
            _time_end("status_icon_detection", t_stage)
        t_stage = _time_start()
        primary_earth_for_layout_obs = str(earth_rank[0][0]) if earth_rank else ""
        if primary_earth_for_layout_obs == "Great Hollow":
            great_hollow_model_layout_observations = infer_great_hollow_layout_model_observations(
                raw_panel_bgr=raw_panel_bgr,
                package=package,
                ctx=ctx,
            )
        else:
            great_hollow_model_layout_observations = {}
            ctx.log(
                f"Skipping Great Hollow model-observation step because primary earth is {primary_earth_for_layout_obs}."
            )
        _time_end("great_hollow_layout_model_observation", t_stage)
        dual_shot_under_status_observation = None
        if bool(dual_shot_enabled) and second_full_img is not None and not (
            _gh_use_parallel_panel_classifier() and _gh_layout_model_match_mode() == "models"
        ):
            primary_earth = str(earth_rank[0][0]) if earth_rank else ""
            if primary_earth == "Great Hollow":
                ctx.log("[Dual-shot] Great Hollow is primary earth in shot #1. Running shot #2 middle-bottom Great Church status classifier.")
                try:
                    t_stage = _time_start()
                    dual_shot_under_status_observation = detect_great_hollow_under_status_from_full_screenshot(
                        second_full_img,
                        backend,
                        ctx,
                        detect_map_circle_compat=detect_map_circle_compat,
                        refine_panel_box_compat=refine_panel_box_compat,
                        normalize_panel_for_slot_recognition_compat=normalize_panel_for_slot_recognition_compat,
                    )
                    _time_end("dual_shot_second_status_detection", t_stage)
                except Exception as exc:
                    ctx.log(f"[Dual-shot] Warning: failed to classify shot #2 middle-bottom Great Church status: {exc}")
                    dual_shot_under_status_observation = None
                if dual_shot_under_status_observation is not None:
                    ctx.log(
                        "[Dual-shot] Great Church under status: "
                        f"{dual_shot_under_status_observation.get('best_status', '?')} "
                        f"(score={float(dual_shot_under_status_observation.get('best_score', 0.0)):.4f}, "
                        f"margin={float(dual_shot_under_status_observation.get('margin', 0.0)):.4f}, "
                        f"confidence={str(dual_shot_under_status_observation.get('confidence', 'weak'))})"
                    )
            else:
                ctx.log(f"[Dual-shot] Shot #1 primary earth is {primary_earth}; skipping shot #2 Great Hollow-only status assist.")
        elif bool(dual_shot_enabled) and second_full_img is not None:
            if ctx is not None:
                ctx.log("[Dual-shot] Skipping second-shot Great Hollow status detection because GH model-match mode + parallel classifier are active.")
        if "dual_shot_second_status_detection" not in timings:
            timings["dual_shot_second_status_detection"] = 0.0
        for earth_name, info in geometry_priors.items():
            ctx.log(
                f"Geometry prior for {earth_name}: slot_count={info['slot_count']} "
                f"align={info['align_response']:.4f} prior={info['geometry_prior']:+.4f}"
            )
        for earth_name, info in lost_slot_priors.items():
            ctx.log(
                f"Lost-slot prior for {earth_name}: lost_slots={info['lost_slot_count']} "
                f"prior={info['lost_slot_prior']:+.4f}"
            )
        for earth_name, info in hard_constraint_priors.items():
            prior = float(info.get("hard_constraint_prior", 0.0))
            if abs(prior) < 1e-9:
                continue
            ctx.log(f"Hard constraint prior for {earth_name}: prior={prior:+.4f}")
        for earth_name, info in evergaol_priors.items():
            ctx.log(f"Evergaol prior for {earth_name}: prior={float(info.get('evergaol_prior', 0.0)):+.4f}")
        for anchor_type, info in great_hollow_status_observations.items():
            ctx.log(
                f"Great Hollow status {anchor_type}: best={info.get('best_status', '?')} "
                f"score={float(info.get('best_score', 0.0)):.4f} margin={float(info.get('margin', 0.0)):.4f}"
            )
        if great_hollow_model_layout_observations:
            ctx.log("Great Hollow model-observation summary:")
            for spec_key, obs in great_hollow_model_layout_observations.items():
                if str(obs.get("kind", "")) == "status":
                    ctx.log(
                        f"  {spec_key}: predicted={obs.get('predicted_status', '?')} "
                        f"p={float(obs.get('best_prob', 0.0)):.4f} margin={float(obs.get('margin', 0.0)):.4f}"
                    )
                else:
                    ctx.log(
                        f"  {spec_key}: predicted={obs.get('predicted_slot_label', '?')} "
                        f"p={float(obs.get('best_prob', 0.0)):.4f} margin={float(obs.get('margin', 0.0)):.4f}"
                    )
        if dual_shot_under_status_observation is not None:
            ctx.log(
                "Great Hollow under-status assist: "
                f"best={dual_shot_under_status_observation.get('best_status', '?')} "
                f"score={float(dual_shot_under_status_observation.get('best_score', 0.0)):.4f} "
                f"margin={float(dual_shot_under_status_observation.get('margin', 0.0)):.4f}"
            )

        earth_cascade_rank: List[Tuple[str, float, Dict[str, float]]] = []
        if USE_EARTH_CASCADE_SELECTION:
            t_stage = _time_start()
            earth_cascade_rank = rank_shifting_earths_cascade(
                earth_rank,
                geometry_priors,
                lost_slot_priors,
                hard_constraint_priors,
                evergaol_priors,
                great_hollow_status_observations,
            )
            _time_end("earth_cascade_ranking", t_stage)
            ctx.log("Shifting Earth cascade ranking:")
            for name, score, details in earth_cascade_rank[:5]:
                ctx.log(
                    f"  {name:<14} score={score:.6f} base={float(details.get('base_score', 0.0)):.6f} "
                    f"geom={float(details.get('geometry_prior', 0.0)):+.4f} "
                    f"lost={float(details.get('lost_slot_prior', 0.0)):+.4f} "
                    f"hard={float(details.get('hard_constraint_prior', 0.0)):+.4f} "
                    f"ever={float(details.get('evergaol_prior', 0.0)):+.4f}"
                )
        else:
            timings["earth_cascade_ranking"] = 0.0
            ctx.log("Shifting Earth cascade selection disabled (enable with NR_ENABLE_EARTH_CASCADE=1).")

        if USE_EARTH_CASCADE_SELECTION and earth_cascade_rank:
            candidate_earths = [name for name, _score, _details in earth_cascade_rank[:1]]
            if candidate_earths:
                ctx.log(f"Restricting layout ranking to {candidate_earths[0]} family from cascade.")
        else:
            candidate_earths = [str(earth_rank[0][0])] if earth_rank else []
            if candidate_earths:
                source_txt = "earth_selector_v2" if USE_EARTH_SELECTOR_V2_BY_DEFAULT else "raw earth ranking"
                ctx.log(f"Restricting layout ranking to {candidate_earths[0]} family from {source_txt}.")
        if candidate_earths and candidate_earths[0] == "Great Hollow" and _gh_layout_model_match_mode() == "models":
            ctx.log("Great Hollow model-layout mode active: classic slot likelihood is disabled for GH layout scoring.")

        layout_infer_by_earth: dict = {}
        t_stage = _time_start()
        for earth_name in candidate_earths:
            try:
                infer_result = run_layout_infer_classifier_for_earth(raw_panel_bgr, earth_name, ctx)
            except Exception as exc:
                ctx.log(f"Layout infer classifier failed for {earth_name}: {exc}")
                infer_result = None
            if infer_result is None:
                continue
            layout_infer_by_earth[str(earth_name)] = infer_result
            decided_pid = infer_result.get("decided_patternId", None)
            best_pid = infer_result.get("best_patternId", None)
            unknown = bool(infer_result.get("unknown", False))
            conf = float(infer_result.get("confidence", 0.0))
            margin = float(infer_result.get("margin", 0.0))
            latency_ms = float(infer_result.get("pipeline_latency_ms", 0.0))
            picked_pid = decided_pid if decided_pid is not None else best_pid
            decision_str = "unknown-gated" if unknown and decided_pid is None else "decided"
            ctx.log(
                f"Layout infer for {earth_name}: {decision_str} "
                f"pattern={picked_pid} margin={margin:.4f} conf={conf:.4f} latency={latency_ms:.2f}ms"
            )
        _time_end("layout_classifier_inference", t_stage)

        t_stage_layout = _time_start()
        for earth_name in candidate_earths:
            earth_score = next(score for name, score, _ in earth_rank if name == earth_name)
            slots_out = observations_by_earth[earth_name]["slots"]
            gh_models_mode_active = bool(
                earth_name == "Great Hollow" and _gh_layout_model_match_mode() == "models"
            )
            if gh_models_mode_active:
                # In GH model-match mode, layout identity should come from inferers/models, not classic slot likelihood.
                scoring_slots_out: List[dict] = []
            else:
                scoring_slots_out = list(slots_out)
            geometry_prior = float(geometry_priors.get(earth_name, {}).get("geometry_prior", 0.0))
            lost_slot_prior = float(lost_slot_priors.get(earth_name, {}).get("lost_slot_prior", 0.0))
            hard_constraint_prior = float(hard_constraint_priors.get(earth_name, {}).get("hard_constraint_prior", 0.0))
            evergaol_prior = float(evergaol_priors.get(earth_name, {}).get("evergaol_prior", 0.0))
            layout_infer_result = layout_infer_by_earth.get(str(earth_name))
            for layout in package.layouts_by_earth[earth_name]:
                layout_num = int(layout["layout_number"])
                # Skip non-whitelisted layouts if filter is enabled
                if ENABLE_PATTERN_ID_WHITELIST_FILTER and layout_num not in ALLOWED_PATTERN_IDS and layout_num != 0:
                    continue
                
                great_hollow_status_prior_base = 0.0
                great_hollow_status_details = {"great_hollow_status_prior": 0.0, "status_evidence": []}
                great_hollow_under_status_prior = 0.0
                great_hollow_under_status_details = {"great_hollow_under_status_prior": 0.0, "status_evidence": []}
                great_hollow_model_layout_prior = 0.0
                great_hollow_model_layout_details = {"great_hollow_model_layout_prior": 0.0, "model_evidence": []}
                if earth_name == "Great Hollow":
                    great_hollow_status_prior_base, great_hollow_status_details = compute_great_hollow_status_prior(
                        layout,
                        great_hollow_status_observations,
                    )
                    if dual_shot_under_status_observation is not None:
                        great_hollow_under_status_prior, great_hollow_under_status_details = compute_great_hollow_under_status_prior(
                            int(layout["layout_number"]),
                            dual_shot_under_status_observation,
                        )
                    great_hollow_model_layout_prior, great_hollow_model_layout_details = compute_great_hollow_model_layout_prior(
                        layout=layout,
                        package=package,
                        model_observations=great_hollow_model_layout_observations,
                    )
                great_hollow_status_prior = float(great_hollow_status_prior_base + great_hollow_under_status_prior)
                if gh_models_mode_active:
                    great_hollow_status_prior = 0.0
                    great_hollow_status_details = {
                        "great_hollow_status_prior": 0.0,
                        "status_evidence": [],
                    }
                layout_infer_prior = compute_layout_infer_prior(int(layout["layout_number"]), layout_infer_result)
                total_score, details = score_layout_ai(
                    scoring_slots_out,
                    layout["slot_labels"] if not gh_models_mode_active else [],
                    earth_score,
                    earth_name=earth_name,
                    geometry_prior=geometry_prior,
                    lost_slot_prior=lost_slot_prior,
                    hard_constraint_prior=hard_constraint_prior,
                    evergaol_prior=evergaol_prior,
                    great_hollow_status_prior=great_hollow_status_prior,
                    great_hollow_model_layout_prior=great_hollow_model_layout_prior,
                    layout_infer_prior=layout_infer_prior,
                )
                details["status_evidence"] = (
                    list(great_hollow_status_details.get("status_evidence", []))
                    + list(great_hollow_under_status_details.get("status_evidence", []))
                )
                details["gh_model_evidence"] = list(great_hollow_model_layout_details.get("model_evidence", []))
                details["gh_classic_slot_likelihood_used"] = bool(not gh_models_mode_active)
                all_results.append({
                    "layout_number": int(layout["layout_number"]),
                    "shifting_earth": earth_name,
                    "total_score": float(total_score),
                    "earth_score": float(earth_score),
                    "geometry_prior": float(geometry_prior),
                    "lost_slot_prior": float(lost_slot_prior),
                    "hard_constraint_prior": float(hard_constraint_prior),
                    "evergaol_prior": float(evergaol_prior),
                    "great_hollow_status_prior": float(great_hollow_status_prior),
                    "great_hollow_status_prior_base": float(great_hollow_status_prior_base),
                    "great_hollow_under_status_prior": float(great_hollow_under_status_prior),
                    "great_hollow_model_layout_prior": float(great_hollow_model_layout_prior),
                    "layout_infer_prior": float(layout_infer_prior),
                    "gh_classic_slot_likelihood_used": bool(not gh_models_mode_active),
                    "details": details,
                })
        _time_end("layout_identification_scoring", t_stage_layout)

        t_stage = _time_start()
        all_results.sort(key=lambda item: item["total_score"], reverse=True)
        if all_results:
            best_debug_earth = str(all_results[0]["shifting_earth"])
            best_slots_debug = observations_by_earth.get(best_debug_earth, {}).get("slots")
        _time_end("layout_sort_and_select", t_stage)
        step_end(ctx, "Run AI slot recognizer", t_rec)

        t_stage = _time_start()
        if best_slots_debug is not None:
            ai_dbg = draw_ai_debug_image(panel_bgr, best_slots_debug)
            save_bgr(ai_debug_out, ai_dbg)
            ctx.log(f"Saved AI slot debug image to {ai_debug_out}")
        if evergaol_priors:
            evergaol_dbg = draw_evergaol_debug(panel_bgr, evergaol_priors)
            save_bgr(evergaol_debug_out, evergaol_dbg)
            ctx.log(f"Saved Evergaol debug image to {evergaol_debug_out}")
        _time_end("debug_annotation_rendering", t_stage)

        if all_results:
            ctx.log("")
            ctx.log(f"Top {min(len(all_results), TOP_RESULTS_TO_LOG)} layout matches:")
            for idx_result, item in enumerate(all_results[:TOP_RESULTS_TO_LOG], start=1):
                details = item.get("details", {})
                exact_w = details.get("exact_match_weight")
                mism_w = details.get("hard_mismatch_weight")
                unk_w = details.get("unknown_weight")
                parts = [
                    f"{idx_result}. layout #{int(item['layout_number'])}",
                    f"earth={item.get('shifting_earth', '?')}",
                    f"score={float(item.get('total_score', 0.0)):.6f}",
                    f"earth_score={float(item.get('earth_score', 0.0)):.6f}",
                    f"gh_status={float(item.get('great_hollow_status_prior', 0.0)):+.4f}",
                    f"gh_under={float(item.get('great_hollow_under_status_prior', 0.0)):+.4f}",
                    f"gh_models={float(item.get('great_hollow_model_layout_prior', 0.0)):+.4f}",
                    f"layout_infer={float(item.get('layout_infer_prior', 0.0)):+.4f}",
                ]
                if exact_w is not None:
                    parts.append(f"exactW={float(exact_w):.2f}")
                if mism_w is not None:
                    parts.append(f"mismW={float(mism_w):.2f}")
                if unk_w is not None:
                    parts.append(f"unkW={float(unk_w):.2f}")
                ctx.log("  " + " ".join(parts))
            by_layout_number = {int(item["layout_number"]): item for item in all_results}
            for left_id, right_id in SPECIAL_LAYOUT_COMPARISONS:
                left = by_layout_number.get(int(left_id))
                right = by_layout_number.get(int(right_id))
                if left is None or right is None:
                    continue
                ctx.log("")
                ctx.log(f"Comparison layout #{left_id} vs #{right_id}:")
                for line in format_layout_debug_lines(left):
                    ctx.log(line)
                for line in format_layout_debug_lines(right):
                    ctx.log(line)
                ctx.log(
                    f"  score delta ({left_id}-{right_id}) = "
                    f"{float(left['total_score']) - float(right['total_score']):+.6f}"
                )
            ctx.log("")

        ranked_results: List[RankedResult] = []
        for item in all_results:
            ranked_results.append(
                RankedResult(
                    number=int(item["layout_number"]),
                    filename=f"{int(item['layout_number'])}",
                    score=float(item["total_score"]),
                    details=item,
                )
            )

        ordered_numbers = [r.number for r in ranked_results]
        t_stage = _time_start()
        top_paths = resolve_display_paths(
            ranked_results,
            ctx,
            disp_dir,
            dynamic_render=dynamic_render,
            allow_render_fallback=not dynamic_render_only,
        )
        _time_end("display_path_resolution", t_stage)
        if ENABLE_OVERLAY_TEXT_ONLY_FEATURE:
            t_stage = _time_start()
            overlay_text_top_paths = resolve_overlay_text_only_paths(
                ranked_results,
                ctx,
                dynamic_render=dynamic_render,
            )
            _time_end("overlay_text_display_path_resolution", t_stage)
        else:
            overlay_text_top_paths = []
            timings["overlay_text_display_path_resolution"] = 0.0

        total_elapsed = float(max(0.0, time.perf_counter() - total_start))
        timings["total_run"] = total_elapsed

        payload = {
            "crop_method": crop_method,
            "crop_refinement": crop_refine_info,
            "earth_ranking": [{"name": n, "score": s, "details": d} for n, s, d in earth_rank],
            "earth_cascade_ranking": [{"name": n, "score": s, "details": d} for n, s, d in earth_cascade_rank],
            "candidate_earths": candidate_earths,
            "layout_infer_by_earth": layout_infer_by_earth,
            "best_earth_for_debug": best_debug_earth,
            "observations_by_earth": observations_by_earth,
            "geometry_priors": geometry_priors,
            "lost_slot_priors": lost_slot_priors,
            "hard_constraint_priors": hard_constraint_priors,
            "evergaol_priors": evergaol_priors,
            "great_hollow_status_observations": great_hollow_status_observations,
            "great_hollow_model_layout_observations": great_hollow_model_layout_observations,
            "dual_shot": {
                "enabled": bool(dual_shot_enabled),
                "used_for_great_hollow": bool(dual_shot_under_status_observation is not None),
                "great_hollow_under_status_observation": dual_shot_under_status_observation,
            },
            "evergaol_debug_out": str(evergaol_debug_out),
            "results": all_results,
            "model_path": str(ai_bundle["model_path"]),
            "ai_confidence_threshold": ai_confidence_threshold,
            "ai_margin_threshold": ai_margin_threshold,
            "timings": timings,
        }

        ctx.log("Timing breakdown:")
        for key in [
            "capture_screenshot_or_use_pre_captured",
            "panel_crop_preview_initial",
            "preview_callback",
            "detect_map_circle_main",
            "panel_crop_first_pass",
            "panel_further_crop_normalize",
            "earth_selector_ranking",
            "slot_observation_classification_total",
            "geometry_prior_computation",
            "lost_slot_prior_computation",
            "hard_constraint_prior_computation",
            "evergaol_prior_detection",
            "status_icon_detection",
            "great_hollow_layout_model_observation",
            "dual_shot_second_status_detection",
            "earth_cascade_ranking",
            "layout_classifier_inference",
            "layout_identification_scoring",
            "layout_sort_and_select",
            "debug_annotation_rendering",
            "display_path_resolution",
            "overlay_text_display_path_resolution",
            "total_run",
        ]:
            ctx.log(f"  {key}: {float(timings.get(key, 0.0)):.3f}s")

        ctx.log(f"Resolved {len(top_paths)} display image(s) for preview/result view.")
        ctx.log(
            "Resolved "
            f"{sum(1 for item in overlay_text_top_paths if item is not None)} "
            "overlay text-only image(s)."
        )
        ctx.log(f"GUI total execution time: {fmt_seconds(total_elapsed)}")

        return {
            "results": ranked_results,
            "ordered_numbers": ordered_numbers,
            "crop_box": crop_box_obj,
            "top_paths": top_paths,
            "overlay_text_top_paths": overlay_text_top_paths,
            "cropped_panel": preview_panel if preview_panel is not None else panel_bgr,
            "backend_payload": payload,
            "tuned_config": tuned_cfg,
            "ai_model_path": str(ai_bundle["model_path"]),
        }



# ---------- Overlay support patch for the AI GUI ----------
DEFAULT_OVERLAY_HOTKEY = "Ctrl + Alt + O"
DEFAULT_OVERLAY_OPACITY = 0.85
DEFAULT_COMPLETION_TOAST_ENABLED = True
DEFAULT_OVERLAY_TEXT_ONLY_ENABLED = True
# In-code feature gate for both the text-only overlay behavior and its settings checkbox.
ENABLE_OVERLAY_TEXT_ONLY_FEATURE = True
COMPLETION_TOAST_TEXT = "Recognition Complete"
COMPLETION_TOAST_HOLD_MS = 3000
COMPLETION_TOAST_FADE_MS = 700
DUAL_SHOT_TIMEOUT_MS = 10_000

if sys.platform == "win32" and ctypes is not None:
    GWL_EXSTYLE = -20
    WS_EX_NOACTIVATE = 0x08000000
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_TOPMOST = 0x00000008
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOZORDER = 0x0004
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040
    SWP_FRAMECHANGED = 0x0020
    HWND_TOPMOST = -1


def get_foreground_window_handle() -> Optional[int]:
    if sys.platform != "win32" or ctypes is None:
        return None
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        return int(hwnd) if hwnd else None
    except Exception:
        return None


def restore_foreground_window(hwnd: Optional[int]) -> None:
    if sys.platform != "win32" or ctypes is None or not hwnd:
        return
    try:
        ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


class OverlayMapWindow:
    TRANSPARENT_COLOR = "#010203"
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.window: Optional[tk.Toplevel] = None
        self.label: Optional[tk.Label] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.visible = False
        self.last_geometry: Optional[Tuple[int, int, int, int]] = None
        self.last_image_signature: Optional[str] = None
        self.on_click_callback: Optional[Callable[[], None]] = None
        self.opacity: float = float(DEFAULT_OVERLAY_OPACITY)

    def set_opacity(self, value: float) -> None:
        try:
            value = float(value)
        except Exception:
            value = float(DEFAULT_OVERLAY_OPACITY)
        value = max(0.10, min(1.00, value))
        self.opacity = value
        if self.window is not None:
            try:
                self.window.attributes("-alpha", self.opacity)
            except Exception:
                pass

    def is_visible(self) -> bool:
        return self.visible and self.window is not None

    def hide(self) -> None:
        if self.window is None:
            self.visible = False
            return
        try:
            self.window.withdraw()
        except Exception:
            pass
        self.visible = False

    def destroy(self) -> None:
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
        self.window = None
        self.label = None
        self.photo = None
        self.visible = False

    def _ensure_window(self) -> None:
        if self.window is not None and self.label is not None:
            return
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.configure(bg=self.TRANSPARENT_COLOR)
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        try:
            if sys.platform == "win32":
                win.wm_attributes("-transparentcolor", self.TRANSPARENT_COLOR)
        except Exception:
            pass
        try:
            win.attributes("-alpha", self.opacity)
        except Exception:
            pass
        lbl = tk.Label(win, bg=self.TRANSPARENT_COLOR, bd=0, highlightthickness=0, relief="flat", cursor="hand2")
        lbl.pack(fill="both", expand=True)
        for widget in (win, lbl):
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Button-3>", self._on_click)
        self.window = win
        self.label = lbl
        self.root.update_idletasks()
        self._apply_noactivate_style()

    def _apply_noactivate_style(self) -> None:
        if sys.platform != "win32" or ctypes is None or self.window is None:
            return
        try:
            hwnd = self.window.winfo_id()
            user32 = ctypes.windll.user32
            exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            exstyle |= WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, exstyle)
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW)
        except Exception:
            pass

    def _on_click(self, _event=None) -> str:
        if callable(self.on_click_callback):
            try:
                self.on_click_callback()
            except Exception:
                pass
        else:
            self.hide()
        return "break"

    def show_pil(self, pil_img: Image.Image, x: int, y: int, width: int, height: int, previous_foreground_hwnd: Optional[int] = None) -> None:
        self._ensure_window()
        assert self.window is not None and self.label is not None

        width = max(1, int(width))
        height = max(1, int(height))
        x = int(x)
        y = int(y)

        resized = pil_img.copy().resize((width, height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.label.configure(image=self.photo)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.deiconify()
        try:
            self.window.lift()
        except Exception:
            pass
        self._apply_noactivate_style()
        self.visible = True
        self.last_geometry = (x, y, width, height)

        if previous_foreground_hwnd:
            self.root.after(10, lambda hwnd=previous_foreground_hwnd: restore_foreground_window(hwnd))



class OverlayToastWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.window: Optional[tk.Toplevel] = None
        self.label: Optional[tk.Label] = None
        self.visible = False
        self._after_ids: List[str] = []

    def _cancel_pending(self) -> None:
        while self._after_ids:
            after_id = self._after_ids.pop()
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass

    def hide(self) -> None:
        self._cancel_pending()
        if self.window is not None:
            try:
                self.window.withdraw()
            except Exception:
                pass
        self.visible = False

    def destroy(self) -> None:
        self.hide()
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
        self.window = None
        self.label = None

    def _ensure_window(self) -> None:
        if self.window is not None and self.label is not None:
            return
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        transparent_bg = "#010203"
        win.configure(bg=transparent_bg)
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        try:
            if sys.platform == "win32":
                win.wm_attributes("-transparentcolor", transparent_bg)
        except Exception:
            pass
        lbl = tk.Label(
            win,
            text=COMPLETION_TOAST_TEXT,
            bg=transparent_bg,
            fg="#ff9f1c",
            bd=0,
            highlightthickness=0,
            relief="flat",
            padx=18,
            pady=10,
            font=("Segoe UI", 26, "bold"),
        )
        lbl.pack(fill="both", expand=True)
        for widget in (win, lbl):
            widget.bind("<Button-1>", lambda _event=None: "break")
            widget.bind("<Button-2>", lambda _event=None: "break")
            widget.bind("<Button-3>", lambda _event=None: "break")
        self.window = win
        self.label = lbl
        self.root.update_idletasks()
        self._apply_noactivate_style()

    def _apply_noactivate_style(self) -> None:
        if sys.platform != "win32" or ctypes is None or self.window is None:
            return
        try:
            hwnd = self.window.winfo_id()
            user32 = ctypes.windll.user32
            exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            exstyle |= WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, exstyle)
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
            )
        except Exception:
            pass

    def show_message(
        self,
        text: str,
        monitor: dict,
        hold_ms: int = COMPLETION_TOAST_HOLD_MS,
        fade_ms: int = COMPLETION_TOAST_FADE_MS,
        previous_foreground_hwnd: Optional[int] = None,
    ) -> None:
        self._ensure_window()
        assert self.window is not None and self.label is not None
        self._cancel_pending()

        self.label.configure(text=text)
        self.window.update_idletasks()

        req_w = max(1, self.window.winfo_reqwidth())
        req_h = max(1, self.window.winfo_reqheight())
        mon_left = int(monitor.get("left", 0))
        mon_top = int(monitor.get("top", 0))
        mon_width = max(1, int(monitor.get("width", req_w)))
        mon_height = max(1, int(monitor.get("height", req_h)))

        x = mon_left + max(0, (mon_width - req_w) // 2)
        y = mon_top + max(24, int(round(mon_height * 0.10)))

        try:
            self.window.attributes("-alpha", 1.0)
        except Exception:
            pass
        self.window.geometry(f"{req_w}x{req_h}+{x}+{y}")
        self.window.deiconify()
        try:
            self.window.lift()
        except Exception:
            pass
        self._apply_noactivate_style()
        self.visible = True

        if previous_foreground_hwnd:
            self._after_ids.append(
                self.root.after(10, lambda hwnd=previous_foreground_hwnd: restore_foreground_window(hwnd))
            )

        self._after_ids.append(self.root.after(max(0, int(hold_ms)), lambda: self._begin_fade(max(120, int(fade_ms)))))

    def _begin_fade(self, fade_ms: int) -> None:
        if self.window is None:
            self.visible = False
            return
        steps = max(8, int(round(fade_ms / 40.0)))
        step_delay = max(15, int(round(fade_ms / steps)))

        def _step(index: int) -> None:
            if self.window is None:
                self.visible = False
                return
            if index >= steps:
                self.hide()
                return
            alpha = max(0.0, 1.0 - ((index + 1) / steps))
            try:
                self.window.attributes("-alpha", alpha)
            except Exception:
                pass
            self._after_ids.append(self.root.after(step_delay, lambda i=index + 1: _step(i)))

        _step(0)

# Keep references to original class methods so we can extend them cleanly.
_ORIG_GUI_INIT = NightreignRecognizerGUI.__init__
_ORIG_GUI_OPEN_POPOVER = NightreignRecognizerGUI._open_hotkey_popover
_ORIG_GUI_CLOSE_POPOVER = NightreignRecognizerGUI._close_hotkey_popover
_ORIG_GUI_PROCESS_QUEUE = NightreignRecognizerGUI._process_queue
_ORIG_GUI_SHOW_RESULTS = NightreignRecognizerGUI._show_results
_ORIG_GUI_THUMBNAIL_CLICK = NightreignRecognizerGUI._on_thumbnail_click
_ORIG_GUI_CLOSE = NightreignRecognizerGUI._on_close
_ORIG_GUI_START = NightreignRecognizerGUI.start_or_restart_recognition
_ORIG_GUI_WORKER_MAIN = NightreignRecognizerGUI._worker_main


def _overlay_gui_init(self, root: tk.Tk) -> None:
    _ORIG_GUI_INIT(self, root)
    self.root.title("Nightreign Map Recognizer AI + Overlay v38 (Model-First GH)")

    saved_overlay_hotkey = normalize_hotkey_string(str(self.settings.get("overlay_hotkey", DEFAULT_OVERLAY_HOTKEY)))
    self.overlay_hotkey_string = saved_overlay_hotkey or DEFAULT_OVERLAY_HOTKEY
    self.overlay_hotkey_manager = GlobalHotkeyManager(lambda: self.message_queue.put(("overlay_hotkey",)))
    self.overlay_hotkey_manager.HOTKEY_ID = 0xB124

    self.overlay_hotkey_capture_var = tk.StringVar(value=self.overlay_hotkey_string)
    saved_overlay_opacity = self.settings.get("overlay_opacity", DEFAULT_OVERLAY_OPACITY)
    try:
        saved_overlay_opacity = float(saved_overlay_opacity)
    except Exception:
        saved_overlay_opacity = float(DEFAULT_OVERLAY_OPACITY)
    saved_overlay_opacity = max(0.10, min(1.00, saved_overlay_opacity))
    self.overlay_opacity_var = tk.DoubleVar(value=saved_overlay_opacity)
    self.overlay_opacity_text_var = tk.StringVar(value=f"{int(round(saved_overlay_opacity * 100))}%")
    self.hotkey_capture_target: Optional[str] = None
    self.overlay_window = OverlayMapWindow(self.root)
    self.overlay_window.set_opacity(saved_overlay_opacity)
    self.overlay_window.on_click_callback = self.hide_overlay

    saved_completion_toast_enabled = self.settings.get("completion_toast_enabled", DEFAULT_COMPLETION_TOAST_ENABLED)
    self.completion_toast_enabled_var = tk.BooleanVar(value=bool(saved_completion_toast_enabled))
    if ENABLE_OVERLAY_TEXT_ONLY_FEATURE:
        saved_overlay_text_only_enabled = bool(self.settings.get("overlay_text_only_enabled", DEFAULT_OVERLAY_TEXT_ONLY_ENABLED))
    else:
        saved_overlay_text_only_enabled = False
    self.overlay_text_only_var = tk.BooleanVar(value=saved_overlay_text_only_enabled)
    self.completion_toast_window = OverlayToastWindow(self.root)
    self.current_overlay_text_image_paths: List[Optional[Path | Image.Image]] = []

    self.last_result_monitor_index = 1
    self.last_crop_box: Optional[CropBox] = None
    self.last_result_payload: Optional[dict] = None
    self._next_run_capture_first_full_img: Optional[np.ndarray] = None
    self._next_run_capture_second_full_img: Optional[np.ndarray] = None
    self._next_run_dual_mode: bool = False
    self._next_run_dualshot_debug_dir: Optional[str] = None
    self.dual_shot_waiting_for_second: bool = False
    self.dual_shot_first_full_img: Optional[np.ndarray] = None
    self.dual_shot_monitor_index: int = 1
    self.dual_shot_timeout_token: int = 0
    self.dual_shot_timeout_after_id: Optional[str] = None

    saved_dual_shot_enabled = bool(self.settings.get("dual_shot_enabled", False))
    self.dual_shot_enabled_var = tk.BooleanVar(value=saved_dual_shot_enabled)
    try:
        self.top_bar.configure(height=126)
    except Exception:
        pass
    if os.environ.get('NR_SHOW_DUAL_SHOT_CHECKBOX') is not None:
        self.dual_shot_checkbox = tk.Checkbutton(
            self.top_bar,
            text="Use dual-shot GH assist",
            variable=self.dual_shot_enabled_var,
            command=self._on_dual_shot_toggle,
            bg="#0f141b",
            fg="#b9c4d7",
            activebackground="#0f141b",
            activeforeground="#dbe7ff",
            selectcolor="#162436",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            padx=2,
        pady=0,
        )
        self.dual_shot_checkbox.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

    # small status line addition if desired later
    if hasattr(self, "hotkey_status_var"):
        current = self.hotkey_status_var.get().strip()
        if not current:
            self.hotkey_status_var.set("")


def _overlay_initialize_hotkeys(self) -> None:
    success_main, message_main = self.hotkey_manager.configure(self.global_hotkey_string)
    success_overlay, message_overlay = self.overlay_hotkey_manager.configure(self.overlay_hotkey_string)

    messages = []
    if success_main:
        self.settings["global_hotkey"] = self.global_hotkey_string
        save_settings(self.settings)
    messages.append(f"Recognition: {message_main}")

    if success_overlay:
        self.settings["overlay_hotkey"] = self.overlay_hotkey_string
        save_settings(self.settings)
    messages.append(f"Overlay: {message_overlay}")

    self.hotkey_status_var.set(" | ".join(messages))
    if not success_main or not success_overlay:
        if self.current_run_id:
            self._append_log_line(self.current_run_id, timestamp_now(), self.hotkey_status_var.get())


def _overlay_toggle_hotkey_capture(self, target: str) -> None:
    if self.hotkey_listening and self.hotkey_capture_target == target:
        self._stop_hotkey_capture(reset_display=True)
    else:
        self._start_hotkey_capture(target)


def _overlay_start_hotkey_capture(self, target: str) -> None:
    if self.hotkey_popover is None:
        return
    self.hotkey_listening = True
    self.hotkey_capture_target = target
    self.hotkey_pressed_modifiers = set()
    if target == "recognition":
        self.hotkey_capture_var.set("Press new shortcut...")
        self.hotkey_status_var.set("Listening for a new recognition hotkey...")
        if hasattr(self, "hotkey_listen_button"):
            self.hotkey_listen_button.configure(text="Cancel")
        if hasattr(self, "overlay_hotkey_listen_button"):
            self.overlay_hotkey_listen_button.configure(text="Set overlay hotkey")
    else:
        self.overlay_hotkey_capture_var.set("Press new shortcut...")
        self.hotkey_status_var.set("Listening for a new overlay hotkey...")
        if hasattr(self, "overlay_hotkey_listen_button"):
            self.overlay_hotkey_listen_button.configure(text="Cancel")
        if hasattr(self, "hotkey_listen_button"):
            self.hotkey_listen_button.configure(text="Set hotkey")
    self.hotkey_popover.bind("<KeyPress>", self._on_hotkey_capture_keypress)
    self.hotkey_popover.bind("<KeyRelease>", self._on_hotkey_capture_keyrelease)
    try:
        self.hotkey_popover.focus_force()
    except Exception:
        self.hotkey_popover.focus_set()


def _overlay_stop_hotkey_capture(self, reset_display: bool = False) -> None:
    target = getattr(self, "hotkey_capture_target", None)
    self.hotkey_listening = False
    self.hotkey_pressed_modifiers = set()
    self.hotkey_capture_target = None
    if self.hotkey_popover is not None:
        self.hotkey_popover.unbind("<KeyPress>")
        self.hotkey_popover.unbind("<KeyRelease>")
    if hasattr(self, "hotkey_listen_button"):
        self.hotkey_listen_button.configure(text="Set hotkey")
    if hasattr(self, "overlay_hotkey_listen_button"):
        self.overlay_hotkey_listen_button.configure(text="Set overlay hotkey")
    if reset_display:
        if target == "overlay":
            self.overlay_hotkey_capture_var.set(self.overlay_hotkey_string)
        else:
            self.hotkey_capture_var.set(self.global_hotkey_string)
        self.hotkey_status_var.set("Capture cancelled.")


def _overlay_on_hotkey_capture_keypress(self, event) -> str:
    if not self.hotkey_listening:
        return "break"
    name = hotkey_from_event_keysym(event.keysym)
    if name is None:
        return "break"
    if name == "Esc":
        self._stop_hotkey_capture(reset_display=True)
        return "break"
    target = self.hotkey_capture_target or "recognition"
    if name in HOTKEY_MODIFIER_SET:
        self.hotkey_pressed_modifiers.add(name)
        current = [m for m in HOTKEY_MODIFIER_ORDER if m in self.hotkey_pressed_modifiers]
        display = " + ".join(current) + (" + ..." if current else "Press new shortcut...")
        if target == "overlay":
            self.overlay_hotkey_capture_var.set(display)
        else:
            self.hotkey_capture_var.set(display)
        return "break"
    if not self.hotkey_pressed_modifiers:
        if target == "overlay":
            self.overlay_hotkey_capture_var.set(name)
        else:
            self.hotkey_capture_var.set(name)
        self.hotkey_status_var.set("Use at least one modifier key, such as Ctrl or Alt.")
        return "break"

    combo = format_hotkey_combo(self.hotkey_pressed_modifiers, name)
    if target == "overlay":
        self.overlay_hotkey_capture_var.set(combo)
        success, message = self.overlay_hotkey_manager.configure(combo)
        self.hotkey_status_var.set("Overlay: " + message)
        if success:
            self.overlay_hotkey_string = combo
            self.settings["overlay_hotkey"] = combo
            save_settings(self.settings)
        self._stop_hotkey_capture(reset_display=False)
        self.overlay_hotkey_capture_var.set(self.overlay_hotkey_string)
    else:
        self.hotkey_capture_var.set(combo)
        success, message = self.hotkey_manager.configure(combo)
        self.hotkey_status_var.set("Recognition: " + message)
        if success:
            self.global_hotkey_string = combo
            self.settings["global_hotkey"] = combo
            save_settings(self.settings)
        self._stop_hotkey_capture(reset_display=False)
        self.hotkey_capture_var.set(self.global_hotkey_string)
    return "break"


def _overlay_on_hotkey_capture_keyrelease(self, event) -> str:
    if not self.hotkey_listening:
        return "break"
    name = hotkey_from_event_keysym(event.keysym)
    if name in HOTKEY_MODIFIER_SET and name in self.hotkey_pressed_modifiers:
        self.hotkey_pressed_modifiers.discard(name)
        current = [m for m in HOTKEY_MODIFIER_ORDER if m in self.hotkey_pressed_modifiers]
        display = " + ".join(current) + (" + ..." if current else "Press new shortcut...")
        if self.hotkey_capture_target == "overlay":
            self.overlay_hotkey_capture_var.set(display)
        else:
            self.hotkey_capture_var.set(display)
    return "break"


def _on_overlay_opacity_slider(self, value) -> None:
    try:
        opacity = float(value)
    except Exception:
        opacity = float(DEFAULT_OVERLAY_OPACITY)
    opacity = max(0.10, min(1.00, opacity))
    self.overlay_opacity_var.set(opacity)
    self.overlay_opacity_text_var.set(f"{int(round(opacity * 100))}%")
    self.overlay_window.set_opacity(opacity)
    self.settings["overlay_opacity"] = opacity
    save_settings(self.settings)
    if self.current_run_id:
        self._append_log_line(self.current_run_id, timestamp_now(), f"Overlay opacity set to {opacity:.2f}.")


def _on_completion_toast_toggle(self) -> None:
    enabled = bool(self.completion_toast_enabled_var.get())
    self.settings["completion_toast_enabled"] = enabled
    save_settings(self.settings)
    if not enabled:
        self.completion_toast_window.hide()
    if self.current_run_id:
        state = "enabled" if enabled else "disabled"
        self._append_log_line(self.current_run_id, timestamp_now(), f"Completion toast {state}.")


def _on_overlay_text_only_toggle(self) -> None:
    if not ENABLE_OVERLAY_TEXT_ONLY_FEATURE:
        self.overlay_text_only_var.set(False)
        return
    enabled = bool(self.overlay_text_only_var.get())
    self.settings["overlay_text_only_enabled"] = enabled
    save_settings(self.settings)
    if self.overlay_window.is_visible():
        self._refresh_overlay_image()
    if self.current_run_id:
        state = "enabled" if enabled else "disabled"
        self._append_log_line(self.current_run_id, timestamp_now(), f"Overlay text-only mode {state}.")


def _overlay_open_hotkey_popover(self) -> None:
    if self.hotkey_popover is not None:
        return
    popover = tk.Toplevel(self.root)
    popover.transient(self.root)
    popover.resizable(False, False)
    popover.configure(bg="#0f141b")
    popover.title("Hotkeys")
    try:
        popover.attributes("-topmost", True)
    except Exception:
        pass
    self.hotkey_popover = popover

    panel = tk.Frame(popover, bg="#111823", highlightthickness=1, highlightbackground="#243244", bd=0)
    panel.pack(fill="both", expand=True)
    self.hotkey_popover_content = panel

    body = tk.Frame(panel, bg="#111823", bd=0)
    body.pack(fill="both", expand=True, padx=10, pady=10)

    def add_section(title: str, var: tk.StringVar, button_attr_name: str, callback_target: str, button_text: str):
        tk.Label(body, text=title, bg="#111823", fg="#9fb0c9", anchor="w", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 2))
        tk.Label(body, textvariable=var, bg="#0d1117", fg="#f6f8fc", anchor="w", justify="left", padx=8, pady=6, font=("Segoe UI", 10, "bold")).pack(fill="x", pady=(0, 6))
        controls = tk.Frame(body, bg="#111823", bd=0)
        controls.pack(fill="x", pady=(0, 10))
        btn = tk.Button(
            controls,
            text=button_text,
            command=lambda target=callback_target: self._toggle_hotkey_capture(target),
            bg="#1e66ff",
            fg="#ffffff",
            activebackground="#2b74ff",
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        btn.pack(side="left")
        setattr(self, button_attr_name, btn)

    add_section("Recognition hotkey", self.hotkey_capture_var, "hotkey_listen_button", "recognition", "Set hotkey")
    add_section("Overlay hotkey", self.overlay_hotkey_capture_var, "overlay_hotkey_listen_button", "overlay", "Set overlay hotkey")

    tk.Label(body, text="Overlay opacity", bg="#111823", fg="#9fb0c9", anchor="w", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 2))
    opacity_row = tk.Frame(body, bg="#111823", bd=0)
    opacity_row.pack(fill="x", pady=(0, 10))
    opacity_scale = tk.Scale(
        opacity_row,
        from_=0.10,
        to=1.00,
        resolution=0.01,
        orient="horizontal",
        variable=self.overlay_opacity_var,
        command=self._on_overlay_opacity_slider,
        bg="#111823",
        fg="#f6f8fc",
        highlightthickness=0,
        troughcolor="#223043",
        activebackground="#1e66ff",
        bd=0,
        length=220,
    )
    opacity_scale.pack(side="left", fill="x", expand=True)
    tk.Label(opacity_row, textvariable=self.overlay_opacity_text_var, bg="#111823", fg="#f6f8fc", width=5, anchor="e", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(8, 0))

    toast_check = tk.Checkbutton(
        body,
        text="Show 'Recognition Complete' toast overlay",
        variable=self.completion_toast_enabled_var,
        command=self._on_completion_toast_toggle,
        bg="#111823",
        fg="#f6f8fc",
        activebackground="#111823",
        activeforeground="#f6f8fc",
        selectcolor="#1a2431",
        anchor="w",
        justify="left",
        bd=0,
        highlightthickness=0,
        font=("Segoe UI", 9, "bold"),
    )
    toast_check.pack(fill="x", pady=(0, 10))

    if ENABLE_OVERLAY_TEXT_ONLY_FEATURE:
        overlay_text_check = tk.Checkbutton(
            body,
            text="Use text-only transparent image for overlay",
            variable=self.overlay_text_only_var,
            command=self._on_overlay_text_only_toggle,
            bg="#111823",
            fg="#f6f8fc",
            activebackground="#111823",
            activeforeground="#f6f8fc",
            selectcolor="#1a2431",
            anchor="w",
            justify="left",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9, "bold"),
        )
        overlay_text_check.pack(fill="x", pady=(0, 10))

    tk.Label(
        body,
        text=(
            "Click one of the buttons above, then press the combination. Press Esc to cancel.\n"
            + "The overlay hotkey toggles the selected result map on top of the last recognized monitor.\n"
            + "Use the slider below to control overlay opacity. Clicking the overlay hides it.\n"
            + ("Enable text-only mode to keep only labels in a transparent overlay.\n" if ENABLE_OVERLAY_TEXT_ONLY_FEATURE else "")
            + "The completion toast appears on the recognized monitor, stays for three seconds, then fades away."
        ),
        bg="#111823",
        fg="#9fb0c9",
        anchor="w",
        justify="left",
        wraplength=310,
        font=("Segoe UI", 8),
    ).pack(anchor="w", pady=(0, 6))
    tk.Label(body, textvariable=self.hotkey_status_var, bg="#111823", fg="#8fb2ff", anchor="w", justify="left", wraplength=310, font=("Segoe UI", 8, "bold")).pack(anchor="w")

    self.root.update_idletasks()
    popover.update_idletasks()
    width = max(340, popover.winfo_reqwidth())
    height = popover.winfo_reqheight()
    root_x = self.root.winfo_rootx()
    root_y = self.root.winfo_rooty()
    root_w = max(1, self.root.winfo_width())
    root_h = max(1, self.root.winfo_height())
    x = root_x + max(0, (root_w - width) // 2)
    y = root_y + max(0, (root_h - height) // 2)
    popover.geometry(f"{width}x{height}+{x}+{y}")
    popover.bind("<FocusOut>", self._on_hotkey_popover_focus_out)
    popover.protocol("WM_DELETE_WINDOW", self._close_hotkey_popover)
    self.hotkey_popover_bind_id = self.root.bind_all("<ButtonPress>", self._on_global_button_press_for_popover, add="+")
    popover.lift()
    popover.focus_set()


def _overlay_close_hotkey_popover(self) -> None:
    if self.hotkey_popover is None:
        return
    self._stop_hotkey_capture()
    if self.hotkey_popover_bind_id is not None:
        try:
            self.root.unbind_all("<ButtonPress>")
        except Exception:
            pass
        self.hotkey_popover_bind_id = None
    try:
        self.hotkey_popover.destroy()
    except Exception:
        pass
    self.hotkey_popover = None
    self.hotkey_popover_content = None


def _overlay_clear_dual_shot_pending(self, reset_button: bool = True) -> None:
    after_id = getattr(self, "dual_shot_timeout_after_id", None)
    if after_id is not None:
        try:
            self.root.after_cancel(after_id)
        except Exception:
            pass
    self.dual_shot_timeout_after_id = None
    self.dual_shot_waiting_for_second = False
    self.dual_shot_first_full_img = None
    self.dual_shot_monitor_index = 1
    if reset_button and self.current_cancel_event is None:
        try:
            self.run_button.configure(text="Start recognition")
        except Exception:
            pass


def _overlay_on_dual_shot_toggle(self) -> None:
    enabled = bool(self.dual_shot_enabled_var.get())
    self.settings["dual_shot_enabled"] = bool(enabled)
    save_settings(self.settings)
    if not enabled and bool(getattr(self, "dual_shot_waiting_for_second", False)):
        self._overlay_clear_dual_shot_pending(reset_button=True)
        self.status_var.set("Ready")
        self._set_status_badge_color("#18212d", "#dce6f7")


def _overlay_begin_run_with_captures(
    self,
    monitor_index: int,
    first_full_img: Optional[np.ndarray],
    second_full_img: Optional[np.ndarray],
    dual_mode: bool,
) -> None:
    monitor_label_value = "Primary / default"
    if int(monitor_index) != 1:
        for idx, monitor in self.monitor_entries:
            if idx == int(monitor_index):
                monitor_label_value = monitor_label(idx, monitor)
                break
    self.monitor_choice_var.set(monitor_label_value)

    self._next_run_capture_first_full_img = first_full_img
    self._next_run_capture_second_full_img = second_full_img
    self._next_run_dual_mode = bool(dual_mode and second_full_img is not None)
    self._next_run_dualshot_debug_dir = None

    if is_dual_shot_debug_enabled() and first_full_img is not None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = runtime_dir() / "tmp_dualshot" / f"run_{stamp}_{int(time.time() * 1000) % 1000:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(run_dir / "shot1.png"), first_full_img)
        if second_full_img is not None:
            cv2.imwrite(str(run_dir / "shot2.png"), second_full_img)
        self._next_run_dualshot_debug_dir = str(run_dir)

    self.last_result_monitor_index = int(monitor_index)
    _ORIG_GUI_START(self)


def _overlay_dual_shot_timeout(self, token: int) -> None:
    if int(token) != int(getattr(self, "dual_shot_timeout_token", 0)):
        return
    if not bool(getattr(self, "dual_shot_waiting_for_second", False)):
        return
    first = self.dual_shot_first_full_img
    monitor_index = int(getattr(self, "dual_shot_monitor_index", self._selected_monitor_index()))
    self._overlay_clear_dual_shot_pending(reset_button=False)
    self.status_var.set("Running")
    self._set_status_badge_color("#17315e", "#dbe7ff")
    self._overlay_begin_run_with_captures(
        monitor_index=monitor_index,
        first_full_img=first,
        second_full_img=None,
        dual_mode=False,
    )


def _overlay_start_or_restart_recognition(self) -> None:
    self.hide_overlay()
    try:
        self.completion_toast_window.hide()
    except Exception:
        pass
    if self.current_cancel_event is not None:
        self._overlay_clear_dual_shot_pending(reset_button=False)
        self.last_result_monitor_index = self._selected_monitor_index()
        _ORIG_GUI_START(self)
        return

    dual_enabled = bool(getattr(self, "dual_shot_enabled_var", tk.BooleanVar(value=False)).get())
    if not dual_enabled:
        self._overlay_clear_dual_shot_pending(reset_button=False)
        self.last_result_monitor_index = self._selected_monitor_index()
        self._overlay_begin_run_with_captures(
            monitor_index=self.last_result_monitor_index,
            first_full_img=None,
            second_full_img=None,
            dual_mode=False,
        )
        return

    selected_monitor_index = self._selected_monitor_index()
    if not bool(getattr(self, "dual_shot_waiting_for_second", False)):
        try:
            first_full_img = capture_monitor_bgr(selected_monitor_index)
        except Exception as exc:
            self.status_var.set("Failed")
            self._set_status_badge_color("#51212b", "#ffd9de")
            self.run_button.configure(text="Start recognition")
            if self.current_run_id:
                self._append_log_line(self.current_run_id, timestamp_now(), f"Dual-shot first capture failed: {exc}")
            return
        self.dual_shot_waiting_for_second = True
        self.dual_shot_first_full_img = first_full_img
        self.dual_shot_monitor_index = int(selected_monitor_index)
        self.dual_shot_timeout_token += 1
        token = int(self.dual_shot_timeout_token)
        after_id = getattr(self, "dual_shot_timeout_after_id", None)
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self.dual_shot_timeout_after_id = self.root.after(DUAL_SHOT_TIMEOUT_MS, lambda t=token: self._overlay_dual_shot_timeout(t))
        self.status_var.set("Awaiting second shot")
        self._set_status_badge_color("#3a2f17", "#ffe9b5")
        self.run_button.configure(text="Capture second shot")
        return

    monitor_index = int(getattr(self, "dual_shot_monitor_index", selected_monitor_index))
    first_full_img = self.dual_shot_first_full_img
    self._overlay_clear_dual_shot_pending(reset_button=False)
    try:
        second_full_img = capture_monitor_bgr(monitor_index)
    except Exception as exc:
        self.status_var.set("Running")
        self._set_status_badge_color("#17315e", "#dbe7ff")
        if self.current_run_id:
            self._append_log_line(self.current_run_id, timestamp_now(), f"Dual-shot second capture failed ({exc}); running fallback with first shot only.")
        self._overlay_begin_run_with_captures(
            monitor_index=monitor_index,
            first_full_img=first_full_img,
            second_full_img=None,
            dual_mode=False,
        )
        return
    self._overlay_begin_run_with_captures(
        monitor_index=monitor_index,
        first_full_img=first_full_img,
        second_full_img=second_full_img,
        dual_mode=True,
    )


def _overlay_worker_main(self, run_id: int, cancel_event: threading.Event, monitor_index: int) -> None:
    ctx = RunContext(run_id, cancel_event, self._queue_log)
    first_full_img = getattr(self, "_next_run_capture_first_full_img", None)
    second_full_img = getattr(self, "_next_run_capture_second_full_img", None)
    dual_mode = bool(getattr(self, "_next_run_dual_mode", False))
    debug_dir = getattr(self, "_next_run_dualshot_debug_dir", None)
    self._next_run_capture_first_full_img = None
    self._next_run_capture_second_full_img = None
    self._next_run_dual_mode = False
    self._next_run_dualshot_debug_dir = None
    try:
        if debug_dir:
            ctx.log(f"[Dual-shot debug] Saved trigger captures to: {debug_dir}")
        payload = run_recognition(
            ctx,
            monitor_index=monitor_index,
            preview_callback=lambda img: self._queue_preview(run_id, img),
            pre_captured_full_img=first_full_img,
            second_full_img=second_full_img,
            dual_shot_enabled=dual_mode,
            dynamic_render=os.environ.get('NR_DISABLE_DYNAMIC_RENDER') is None,
            dynamic_render_only=os.environ.get('NR_DISABLE_RENDER_FALLBACK') is None,
        )
        ctx.check_cancel()
        self._queue_result(run_id, payload)
    except CancelledError:
        self._queue_cancelled(run_id)
    except Exception as exc:
        self._queue_error(run_id, f"ERROR: {exc}")
    finally:
        self._queue_finished(run_id)


def _overlay_show_results(self, payload: dict) -> None:
    _ORIG_GUI_SHOW_RESULTS(self, payload)
    crop_box = payload.get("crop_box")
    self.last_crop_box = crop_box if isinstance(crop_box, CropBox) else None
    overlay_paths = payload.get("overlay_text_top_paths")
    self.current_overlay_text_image_paths = overlay_paths if isinstance(overlay_paths, list) else []
    self.last_result_payload = payload
    if self.overlay_window.is_visible():
        self._refresh_overlay_image()


def _overlay_thumbnail_click(self, index: int) -> None:
    _ORIG_GUI_THUMBNAIL_CLICK(self, index)
    if self.overlay_window.is_visible():
        self._refresh_overlay_image()


def _get_monitor_dict_for_overlay(self) -> Optional[dict]:
    for idx, monitor in self.monitor_entries:
        if idx == self.last_result_monitor_index:
            return monitor
    return None


def _refresh_overlay_image(self) -> None:
    if not self.current_image_paths or self.last_crop_box is None:
        self.hide_overlay()
        return
    monitor = self._get_monitor_dict_for_overlay()
    if monitor is None:
        self.hide_overlay()
        return

    selected_index = min(self.current_selected_index, len(self.current_image_paths) - 1)
    image_path: Path | Image.Image = self.current_image_paths[selected_index]
    if ENABLE_OVERLAY_TEXT_ONLY_FEATURE and hasattr(self, "overlay_text_only_var") and bool(self.overlay_text_only_var.get()):
        overlay_paths = getattr(self, "current_overlay_text_image_paths", [])
        if selected_index < len(overlay_paths):
            overlay_image = overlay_paths[selected_index]
            if overlay_image is not None:
                image_path = overlay_image
    pil_img = self._load_pil_image(image_path)

    width = max(1, int(self.last_crop_box.x1 - self.last_crop_box.x0))
    height = max(1, int(self.last_crop_box.y1 - self.last_crop_box.y0))
    screen_x = int(monitor.get("left", 0)) + int(self.last_crop_box.x0)
    screen_y = int(monitor.get("top", 0)) + int(self.last_crop_box.y0)

    prev_hwnd = get_foreground_window_handle()
    self.overlay_window.show_pil(
        pil_img=pil_img,
        x=screen_x,
        y=screen_y,
        width=width,
        height=height,
        previous_foreground_hwnd=prev_hwnd,
    )
    self._append_log_line(self.current_run_id, timestamp_now(), f"Overlay shown at ({screen_x},{screen_y}) size {width}x{height} for layout #{self.current_results[selected_index].number}.") if self.current_run_id else None


def show_overlay(self) -> None:
    if not self.current_results or not self.current_image_paths:
        if self.current_run_id:
            self._append_log_line(self.current_run_id, timestamp_now(), "Overlay requested, but there are no recognition results yet.")
        return
    if self.last_crop_box is None:
        if self.current_run_id:
            self._append_log_line(self.current_run_id, timestamp_now(), "Overlay requested, but the last recognition did not provide a crop box.")
        return
    self._refresh_overlay_image()


def hide_overlay(self) -> None:
    self.overlay_window.hide()


def toggle_overlay(self) -> None:
    if self.overlay_window.is_visible():
        self.hide_overlay()
    else:
        self.show_overlay()


def _show_completion_toast(self) -> None:
    if not bool(self.completion_toast_enabled_var.get()):
        return
    monitor = self._get_monitor_dict_for_overlay()
    if monitor is None:
        return
    prev_hwnd = get_foreground_window_handle()
    self.completion_toast_window.show_message(
        text=COMPLETION_TOAST_TEXT,
        monitor=monitor,
        hold_ms=COMPLETION_TOAST_HOLD_MS,
        fade_ms=COMPLETION_TOAST_FADE_MS,
        previous_foreground_hwnd=prev_hwnd,
    )
    if self.current_run_id:
        self._append_log_line(self.current_run_id, timestamp_now(), "Displayed completion toast overlay.")


def _overlay_process_queue(self) -> None:
    try:
        while True:
            item = self.message_queue.get_nowait()
            kind = item[0]
            if kind == "log":
                _, run_id, message, ts = item
                self._append_log_line(run_id, ts, message)
            elif kind == "error":
                _, run_id, message, ts = item
                self._append_log_line(run_id, ts, message)
                if run_id == self.current_run_id:
                    self.status_var.set("Failed")
                    self._set_status_badge_color("#51212b", "#ffd9de")
                    self.run_button.configure(text="Start recognition")
            elif kind == "cancelled":
                _, run_id = item
                if run_id == self.current_run_id:
                    self.status_var.set("Cancelled")
                    self._set_status_badge_color("#3a2f17", "#ffe9b5")
            elif kind == "preview":
                _, run_id, cropped_panel = item
                if run_id == self.current_run_id:
                    self._show_preview(cropped_panel)
            elif kind == "hotkey":
                self.start_or_restart_recognition()
            elif kind == "overlay_hotkey":
                self.toggle_overlay()
            elif kind == "result":
                _, run_id, payload = item
                if run_id == self.current_run_id:
                    self._show_results(payload)
                    self.status_var.set("Completed")
                    self._set_status_badge_color("#1c3c2b", "#d5ffe7")
                    self.run_button.configure(text="Start recognition")
                    self._show_completion_toast()
            elif kind == "finished":
                _, run_id = item
                if run_id == self.current_run_id:
                    self.current_cancel_event = None
                    if self.status_var.get() == "Running":
                        self.status_var.set("Completed")
                        self._set_status_badge_color("#1c3c2b", "#d5ffe7")
                        self.run_button.configure(text="Start recognition")
    except queue.Empty:
        pass
    self.root.after(100, self._process_queue)


def _overlay_on_close(self) -> None:
    try:
        self._overlay_clear_dual_shot_pending(reset_button=False)
    except Exception:
        pass
    try:
        self.hide_overlay()
    except Exception:
        pass
    try:
        self.completion_toast_window.destroy()
    except Exception:
        pass
    try:
        self.overlay_window.destroy()
    except Exception:
        pass
    try:
        self.overlay_hotkey_manager.stop()
    except Exception:
        pass
    _ORIG_GUI_CLOSE(self)


NightreignRecognizerGUI.__init__ = _overlay_gui_init
NightreignRecognizerGUI._initialize_hotkey_registration = _overlay_initialize_hotkeys
NightreignRecognizerGUI._toggle_hotkey_capture = _overlay_toggle_hotkey_capture
NightreignRecognizerGUI._overlay_toggle_hotkey_capture = _overlay_toggle_hotkey_capture
NightreignRecognizerGUI._overlay_start_hotkey_capture = _overlay_start_hotkey_capture
NightreignRecognizerGUI._overlay_stop_hotkey_capture = _overlay_stop_hotkey_capture
NightreignRecognizerGUI._start_hotkey_capture = _overlay_start_hotkey_capture
NightreignRecognizerGUI._stop_hotkey_capture = _overlay_stop_hotkey_capture
NightreignRecognizerGUI._on_hotkey_capture_keypress = _overlay_on_hotkey_capture_keypress
NightreignRecognizerGUI._on_hotkey_capture_keyrelease = _overlay_on_hotkey_capture_keyrelease
NightreignRecognizerGUI._on_overlay_opacity_slider = _on_overlay_opacity_slider
NightreignRecognizerGUI._on_completion_toast_toggle = _on_completion_toast_toggle
NightreignRecognizerGUI._on_overlay_text_only_toggle = _on_overlay_text_only_toggle
NightreignRecognizerGUI._on_dual_shot_toggle = _overlay_on_dual_shot_toggle
NightreignRecognizerGUI._overlay_dual_shot_timeout = _overlay_dual_shot_timeout
NightreignRecognizerGUI._overlay_begin_run_with_captures = _overlay_begin_run_with_captures
NightreignRecognizerGUI._overlay_clear_dual_shot_pending = _overlay_clear_dual_shot_pending
NightreignRecognizerGUI._open_hotkey_popover = _overlay_open_hotkey_popover
NightreignRecognizerGUI._close_hotkey_popover = _overlay_close_hotkey_popover
NightreignRecognizerGUI.start_or_restart_recognition = _overlay_start_or_restart_recognition
NightreignRecognizerGUI._worker_main = _overlay_worker_main
NightreignRecognizerGUI._show_results = _overlay_show_results
NightreignRecognizerGUI._on_thumbnail_click = _overlay_thumbnail_click
NightreignRecognizerGUI._process_queue = _overlay_process_queue
NightreignRecognizerGUI._on_close = _overlay_on_close
NightreignRecognizerGUI._get_monitor_dict_for_overlay = _get_monitor_dict_for_overlay
NightreignRecognizerGUI._refresh_overlay_image = _refresh_overlay_image
NightreignRecognizerGUI.show_overlay = show_overlay
NightreignRecognizerGUI.hide_overlay = hide_overlay
NightreignRecognizerGUI.toggle_overlay = toggle_overlay
NightreignRecognizerGUI._show_completion_toast = _show_completion_toast


def main() -> int:
    set_windows_app_user_model_id()
    root = tk.Tk()
    apply_window_icons(root)
    NightreignRecognizerGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
