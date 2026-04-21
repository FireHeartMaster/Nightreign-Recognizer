#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import math
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk

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

CACHE_FILENAME = "nightreign_ref_cache_square_gui.npz"
DISPLAY_SUBDIRS = [Path("renders/with_text"), Path("all_images/with_text")]
LEGACY_CACHE_SUBDIR = Path("all_images/no_text")
LOG_FILE_NAME = "recognition.log"
SETTINGS_FILE_NAME = "nightreign_settings.json"
DEFAULT_GLOBAL_HOTKEY = "Ctrl + Alt + M"
LARGE_EXE_ICON_FILE = "nightreign_sword_large_transparent_256.ico"
SMALL_WINDOW_ICON_FILE = "nightreign_sword_small_transparent_48.ico"
WINDOWS_APP_USER_MODEL_ID = "nightreign.recognizer.slots"


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
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_base_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
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


def resolve_display_paths(results: Sequence[RankedResult], ctx: RunContext, disp_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for result in results[:THUMB_COUNT]:
        try:
            paths.append(find_image_by_number(disp_dir, result.number))
        except FileNotFoundError:
            ctx.log(f"Warning: Could not find image #{result.number} in {disp_dir}.")
    return paths


def run_recognition(ctx: RunContext, monitor_index: int = 1, preview_callback: Optional[Callable[[np.ndarray], None]] = None) -> dict:
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
    top_paths = resolve_display_paths(results, ctx, disp_dir)
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
        self.root.title("Nightreign Map Recognizer - Tuned Slot Backend")
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

        self.monitor_entries = get_available_monitors()
        if not self.monitor_entries:
            raise RuntimeError("No connected displays were detected.")
        self.monitor_choice_var = tk.StringVar(value="Primary / default")

        self.current_results: List[RankedResult] = []
        self.current_image_paths: List[Path] = []
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
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

        self.logs_frame = tk.Frame(self.root, bg="#0d1117", width=390, highlightthickness=0, bd=0)
        self.logs_frame.grid(row=0, column=0, sticky="nsew")
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
        self.results_frame.grid_columnconfigure(1, weight=0)

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
        self.thumbs_frame.grid(row=0, column=1, sticky="ns", padx=0, pady=0)
        self.thumbs_frame.grid_propagate(False)

        self.thumb_labels: List[tk.Label] = []
        for idx in range(THUMB_COUNT):
            lbl = tk.Label(
                self.thumbs_frame,
                bg="#000000",
                bd=0,
                highlightthickness=0,
                anchor="center",
                cursor="hand2",
                text="",
                compound="center",
            )
            lbl.pack(side="top", fill="both", expand=True, padx=0, pady=0)
            lbl.bind("<Button-1>", lambda event, i=idx: self._on_thumbnail_click(i))
            self.thumb_labels.append(lbl)

        self.results_frame.bind("<Configure>", self._on_results_configure)
        self.thumbs_frame.bind("<Configure>", self._on_results_configure)
        self.main_image_canvas.bind("<Configure>", self._on_results_configure)
        self.main_image_canvas.bind("<MouseWheel>", self._on_main_canvas_mousewheel)
        self.main_image_canvas.bind("<Button-4>", self._on_main_canvas_mousewheel)
        self.main_image_canvas.bind("<Button-5>", self._on_main_canvas_mousewheel)
        self.main_image_canvas.bind("<Double-Button-1>", self._reset_main_zoom)
        self.main_image_canvas.bind("<ButtonPress-1>", self._on_main_canvas_button_press)
        self.main_image_canvas.bind("<B1-Motion>", self._on_main_canvas_drag)
        self.main_image_canvas.bind("<ButtonRelease-1>", self._on_main_canvas_button_release)

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
            payload = run_recognition(ctx, monitor_index=monitor_index, preview_callback=lambda img: self._queue_preview(run_id, img))
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

    def _load_pil_image(self, path: Path) -> Image.Image:
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

    def _show_preview(self, cropped_panel: np.ndarray) -> None:
        pil_img = self._pil_from_bgr(cropped_panel)
        self.current_results = []
        self.current_image_paths = []
        self.preview_photo = None
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
            self.current_selected_index = index
            self._refresh_best_labels()
            self._refresh_result_images()

    def _on_results_configure(self, _event=None) -> None:
        self._refresh_main_canvas_image()
        self._refresh_result_images()

    def _on_root_configure(self, _event=None) -> None:
        total_width = self.results_frame.winfo_width()
        if total_width > 0:
            thumb_width = max(170, int(total_width * 0.20))
            self.thumbs_frame.configure(width=thumb_width)
        self._refresh_main_canvas_image()
        self._refresh_result_images()

    def _refresh_result_images(self) -> None:
        if not self.current_image_paths:
            return

        selected_index = min(self.current_selected_index, len(self.current_image_paths) - 1)
        main_path = self.current_image_paths[selected_index]
        main_img = self._load_pil_image(main_path)
        self._set_main_pil_image(main_img, reset_zoom=False)

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
                )
                if idx == selected_index:
                    label.configure(bg="#111824")
                else:
                    label.configure(bg="#000000")
            else:
                self.thumb_photos[idx] = None
                label.configure(image="", text="", bg="#000000")



# --- Slot-based recognition backend (v16) ---
SLOT_PACKAGE_FILENAME = "nightreign_geom_proto_v4_slots.npz"
SLOT_LANDMARK_TYPES = ["Church", "Fort", "Great Church", "Main Encampment", "Ruins", "Sorcerer's Rise"]
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
    earth_rank = classify_shifting_earth_slot(panel_bgr, package)
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



def run_recognition(ctx: RunContext, monitor_index: int = 1, preview_callback: Optional[Callable[[np.ndarray], None]] = None) -> dict:
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
    top_paths = resolve_display_paths(results, ctx, disp_dir)
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


def run_recognition(ctx: RunContext, monitor_index: int = 1, preview_callback: Optional[Callable[[np.ndarray], None]] = None) -> dict:
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
        t_capture = step_start(ctx, "Capture screenshot for preview")
        full_img = backend.capture_monitor_bgr(monitor_index)
        step_end(ctx, "Capture screenshot for preview", t_capture)
        cv2.imwrite(str(screenshot_path), full_img)

        # Build the GUI preview from the initial crop, before later normalization/refinement.
        preview_panel = None
        try:
            crop_circle, crop_method = backend.detect_map_circle(full_img)
            if crop_method == "blue_ring":
                preview_box = backend.clip_box(
                    backend.estimate_panel_box(crop_circle, backend.DEFAULT_PANEL_SCALE),
                    full_img.shape[1],
                    full_img.shape[0],
                )
                preview_panel = full_img[preview_box.y0:preview_box.y1, preview_box.x0:preview_box.x1].copy()
            else:
                preview_box = backend.refine_panel_box(full_img, crop_circle)
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
        top_paths = resolve_display_paths(ranked_results, ctx, disp_dir)

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
    "slot_patch_model_v1.pt",
    "slot_ai_test_output/slot_patch_model_v1.pt",
]


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


_AI_MODEL_CACHE = None
_AI_DEVICE_CACHE = None


def choose_torch_device() -> torch.device:
    global _AI_DEVICE_CACHE
    if _AI_DEVICE_CACHE is None:
        _AI_DEVICE_CACHE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _AI_DEVICE_CACHE


def find_ai_model_path(ctx: Optional[RunContext] = None) -> Path:
    candidates: List[Path] = []
    for name in AI_MODEL_FILENAMES:
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
                ctx.log(f"Using AI slot model path: {cand.resolve()}")
            return cand

    searched = "\n".join(str(p) for p in unique)
    raise FileNotFoundError(f"AI slot model not found. Looked in:\n{searched}")


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


def ai_classify_slots_for_earth(
    panel_bgr: np.ndarray,
    backend,
    package,
    earth_name: str,
    ai_bundle: dict,
    confidence_threshold: float,
    margin_threshold: float,
) -> Tuple[List[dict], Dict[str, float]]:
    dx_norm, dy_norm, align_response = backend.estimate_global_translation(panel_bgr, package, earth_name)
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
        probs = infer_slot_probs_ai(model, labels, patch, device)
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_prob = ordered[0]
        second_prob = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = best_prob - second_prob

        if best_prob < confidence_threshold or margin < margin_threshold:
            final_label = "unknown"
        else:
            final_label = best_label

        slots_out.append({
            "slot_id": int(slot["slot_id"]),
            "slot_name": str(slot["slot_name"]),
            "label": final_label,
            "best_prob": float(best_prob),
            "second_prob": float(second_prob),
            "margin": float(margin),
            "probs": probs,
            "x_norm": float(cx / w),
            "y_norm": float(cy / h),
            "nominal_x_norm": float(slot["x_norm"]),
            "nominal_y_norm": float(slot["y_norm"]),
            "radius_norm": float(slot["radius_norm"]),
            "weight": float(slot.get("weight", 1.0)),
        })

    return slots_out, {
        "dx_norm": float(dx_norm),
        "dy_norm": float(dy_norm),
        "align_response": float(align_response),
    }


def score_layout_ai(slots_out: Sequence[dict], layout_slot_labels: Sequence[str], earth_score: float) -> Tuple[float, dict]:
    eps = 1e-6
    total = 0.0
    exact_weight = 0.0
    unknown_weight = 0.0
    hard_mismatch_weight = 0.0

    for slot_obs, expected in zip(slots_out, layout_slot_labels):
        probs = slot_obs["probs"]
        p = max(eps, float(probs.get(str(expected), eps)))
        weight = float(slot_obs.get("weight", 1.0))
        total += weight * math.log(p)

        observed = str(slot_obs["label"])
        if observed == str(expected):
            exact_weight += weight
        elif observed == "unknown":
            unknown_weight += weight
        else:
            hard_mismatch_weight += weight

    total += 2.0 * len(slots_out) * float(earth_score)
    details = {
        "exact_match_weight": float(exact_weight),
        "hard_mismatch_weight": float(hard_mismatch_weight),
        "unknown_weight": float(unknown_weight),
    }
    return float(total), details


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


def run_recognition(ctx: RunContext, monitor_index: int = 1, preview_callback: Optional[Callable[[np.ndarray], None]] = None) -> dict:
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
    package = backend.RecognitionPackage(package_path)
    ai_bundle = load_ai_model_bundle(ctx)

    ai_confidence_threshold = float(tuned_cfg.get("ai_confidence_threshold", 0.55))
    ai_margin_threshold = float(tuned_cfg.get("ai_margin_threshold", 0.08))
    ctx.log(f"Using AI confidence_threshold={ai_confidence_threshold:.3f}")
    ctx.log(f"Using AI margin_threshold={ai_margin_threshold:.3f}")
    ctx.log("Using trained AI slot classifier for slot recognition.")
    backend.log = lambda msg: ctx.log(str(msg))

    with tempfile.TemporaryDirectory(prefix="nightreign_gui_ai_") as td:
        td_path = Path(td)
        crop_out = td_path / "recognition_panel_used.png"
        ai_debug_out = td_path / "recognition_panel_used_with_slot_labels_ai.png"
        screenshot_path = td_path / "captured_full_screenshot.png"

        t_capture = step_start(ctx, "Capture screenshot for preview")
        full_img = backend.capture_monitor_bgr(monitor_index)
        step_end(ctx, "Capture screenshot for preview", t_capture)
        cv2.imwrite(str(screenshot_path), full_img)

        preview_panel = None
        try:
            crop_circle, crop_method = backend.detect_map_circle(full_img)
            if crop_method == "blue_ring":
                preview_box = backend.clip_box(
                    backend.estimate_panel_box(crop_circle, backend.DEFAULT_PANEL_SCALE),
                    full_img.shape[1],
                    full_img.shape[0],
                )
                preview_panel = full_img[preview_box.y0:preview_box.y1, preview_box.x0:preview_box.x1].copy()
            else:
                preview_box = backend.refine_panel_box(full_img, crop_circle)
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

        t_rec = step_start(ctx, "Run AI slot recognizer")

        crop_box_obj = None
        crop_refine_info = None

        circle, crop_method = backend.detect_map_circle(full_img)
        if crop_method == "blue_ring":
            crop_box = backend.clip_box(
                backend.estimate_panel_box(circle, backend.DEFAULT_PANEL_SCALE),
                full_img.shape[1],
                full_img.shape[0],
            )
            raw_panel_bgr = full_img[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
            panel_bgr = backend.normalize_panel_for_recognition(raw_panel_bgr)
            earth_rank = backend.classify_shifting_earth(panel_bgr, package)
            crop_box_obj = CropBox(crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1)
        else:
            initial_box = backend.refine_panel_box(full_img, circle)
            crop_box, raw_panel_bgr, panel_bgr, metrics, earth_rank, crop_refine_info = backend.refine_crop_box_by_earth_score(full_img, initial_box, package)
            crop_method = "gray_hough_refined"
            crop_box_obj = CropBox(crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1)

        save_bgr(crop_out, panel_bgr)

        ctx.log("Shifting Earth ranking:")
        for name, score, details in earth_rank[:5]:
            ctx.log(f"  {name:<14} score={score:.6f} low_corr={details['low_corr']:.4f} grad_corr={details['grad_corr']:.4f}")

        candidate_earths = [earth_rank[0][0]]
        if len(earth_rank) > 1 and (earth_rank[0][1] - earth_rank[1][1]) < float(backend.EARTH_SECONDARY_THRESHOLD):
            candidate_earths.append(earth_rank[1][0])

        all_results: List[dict] = []
        best_slots_debug = None
        best_debug_earth = None
        observations_by_earth: dict = {}

        for earth_name in candidate_earths:
            earth_score = next(score for name, score, _ in earth_rank if name == earth_name)
            slots_out, align = ai_classify_slots_for_earth(
                panel_bgr=panel_bgr,
                backend=backend,
                package=package,
                earth_name=earth_name,
                ai_bundle=ai_bundle,
                confidence_threshold=ai_confidence_threshold,
                margin_threshold=ai_margin_threshold,
            )
            unknown_count = sum(1 for s in slots_out if s["label"] == "unknown")
            ctx.log(f"AI slots for earth {earth_name}: dx={align['dx_norm']:+.4f} dy={align['dy_norm']:+.4f} resp={align['align_response']:.4f} unknown={unknown_count}/{len(slots_out)}")

            observations_by_earth[earth_name] = {"align": align, "slots": slots_out}
            if best_slots_debug is None:
                best_slots_debug = slots_out
                best_debug_earth = earth_name

            for layout in package.layouts_by_earth[earth_name]:
                total_score, details = score_layout_ai(slots_out, layout["slot_labels"], earth_score)
                all_results.append({
                    "layout_number": int(layout["layout_number"]),
                    "shifting_earth": earth_name,
                    "total_score": float(total_score),
                    "earth_score": float(earth_score),
                    "details": details,
                })

        all_results.sort(key=lambda item: item["total_score"], reverse=True)
        step_end(ctx, "Run AI slot recognizer", t_rec)

        if best_slots_debug is not None:
            ai_dbg = draw_ai_debug_image(panel_bgr, best_slots_debug)
            save_bgr(ai_debug_out, ai_dbg)
            ctx.log(f"Saved AI slot debug image to {ai_debug_out}")

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
                ]
                if exact_w is not None:
                    parts.append(f"exactW={float(exact_w):.2f}")
                if mism_w is not None:
                    parts.append(f"mismW={float(mism_w):.2f}")
                if unk_w is not None:
                    parts.append(f"unkW={float(unk_w):.2f}")
                ctx.log("  " + " ".join(parts))
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
        top_paths = resolve_display_paths(ranked_results, ctx, disp_dir)

        payload = {
            "crop_method": crop_method,
            "crop_refinement": crop_refine_info,
            "earth_ranking": [{"name": n, "score": s, "details": d} for n, s, d in earth_rank],
            "candidate_earths": candidate_earths,
            "best_earth_for_debug": best_debug_earth,
            "observations_by_earth": observations_by_earth,
            "results": all_results,
            "model_path": str(ai_bundle["model_path"]),
            "ai_confidence_threshold": ai_confidence_threshold,
            "ai_margin_threshold": ai_margin_threshold,
        }

        ctx.log(f"Resolved {len(top_paths)} display image(s) for preview/result view.")
        ctx.log(f"GUI total execution time: {fmt_seconds(time.perf_counter() - total_start)}")

        return {
            "results": ranked_results,
            "ordered_numbers": ordered_numbers,
            "crop_box": crop_box_obj,
            "top_paths": top_paths,
            "cropped_panel": preview_panel if preview_panel is not None else panel_bgr,
            "backend_payload": payload,
            "tuned_config": tuned_cfg,
            "ai_model_path": str(ai_bundle["model_path"]),
        }


def main() -> int:
    set_windows_app_user_model_id()
    root = tk.Tk()
    apply_window_icons(root)
    NightreignRecognizerGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
