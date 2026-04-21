import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

# --------------------------------------------------
# Config: adjust these paths if needed
# --------------------------------------------------
BACKGROUNDS_DIR = Path(r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\maps")
ICONS_DIR = Path(r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\POI_icons")
OUTPUT_DIR = Path(r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\fixed_icon_tool_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# World/image coordinate system from your renderer
IMAGE_W = 1536
IMAGE_H = 1536
WORLD_MIN_X = -256.0
WORLD_MIN_Z = -256.0
WORLD_MAX_X = 1280.0
WORLD_MAX_Z = 1280.0

# UI display size
DISPLAY_W = 1024
DISPLAY_H = 1024

# Right-click erase sensitivity in screen pixels
ERASE_RADIUS_PX = 24


def file_safe_name(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def image_to_world(px: float, py: float) -> tuple[float, float]:
    x = WORLD_MIN_X + (px / IMAGE_W) * (WORLD_MAX_X - WORLD_MIN_X)
    z = WORLD_MIN_Z + ((IMAGE_H - py) / IMAGE_H) * (WORLD_MAX_Z - WORLD_MIN_Z)
    return x, z


def world_to_image(x: float, z: float) -> tuple[float, float]:
    px = (float(x) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X) * IMAGE_W
    py = IMAGE_H - ((float(z) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z) * IMAGE_H)
    return px, py


class FixedIconTool:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Nightreign Fixed Icon Tool")
        self.root.geometry("1500x1100")

        self.backgrounds = self.discover_backgrounds()
        self.icons = self.discover_icons()
        self.icon_images_full: dict[str, Image.Image] = {}
        self.icon_thumbs: dict[str, ImageTk.PhotoImage] = {}
        self.entries_by_map: dict[str, list[dict]] = {name: [] for name in self.backgrounds}
        self.undo_stack: list[dict] = []

        self.current_map_name = tk.StringVar(value=self.backgrounds[0] if self.backgrounds else "")
        self.selected_icon_name = tk.StringVar(value=self.icons[0] if self.icons else "")
        self.text_value = tk.StringVar(value="")
        self.include_text_var = tk.BooleanVar(value=True)
        self.output_style_var = tk.StringVar(value="python_dict")

        self.current_background_original: Image.Image | None = None
        self.current_background_display: Image.Image | None = None
        self.current_tk_background: ImageTk.PhotoImage | None = None
        self.current_scale_x = DISPLAY_W / IMAGE_W
        self.current_scale_y = DISPLAY_H / IMAGE_H

        self.build_ui()
        self.load_icon_cache()
        if self.backgrounds:
            self.load_background(self.backgrounds[0])
            self.refresh_entry_list()

        self.root.bind("<Control-z>", self.undo_last_action)

    def discover_backgrounds(self) -> list[str]:
        if not BACKGROUNDS_DIR.exists():
            return []
        names = []
        for p in sorted(BACKGROUNDS_DIR.glob("*.png")):
            names.append(p.stem.replace("_", " "))
        return names

    def discover_icons(self) -> list[str]:
        if not ICONS_DIR.exists():
            return []
        names = []
        for p in sorted(ICONS_DIR.glob("*.png")):
            names.append(p.stem.replace("_", " "))
        return names

    def build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 8))

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # Controls
        ttk.Label(left, text="Shifting Earth / rareMap").pack(anchor="w")
        self.map_combo = ttk.Combobox(left, textvariable=self.current_map_name, values=self.backgrounds, state="readonly", width=30)
        self.map_combo.pack(fill="x", pady=(0, 8))
        self.map_combo.bind("<<ComboboxSelected>>", self.on_map_changed)

        ttk.Label(left, text="Icon").pack(anchor="w")
        self.icon_combo = ttk.Combobox(left, textvariable=self.selected_icon_name, values=self.icons, state="readonly", width=30)
        self.icon_combo.pack(fill="x", pady=(0, 8))

        ttk.Label(left, text="Text").pack(anchor="w")
        ttk.Entry(left, textvariable=self.text_value, width=32).pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(left, text="Include text in exported entry", variable=self.include_text_var).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Export style").pack(anchor="w")
        ttk.Radiobutton(left, text="Python dict snippet", variable=self.output_style_var, value="python_dict").pack(anchor="w")
        ttk.Radiobutton(left, text="JSON", variable=self.output_style_var, value="json").pack(anchor="w", pady=(0, 8))

        button_row = ttk.Frame(left)
        button_row.pack(fill="x", pady=(0, 8))
        ttk.Button(button_row, text="Undo", command=self.undo_last_action).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(button_row, text="Export TXT", command=self.export_txt).pack(side="left", fill="x", expand=True)

        button_row2 = ttk.Frame(left)
        button_row2.pack(fill="x", pady=(0, 8))
        ttk.Button(button_row2, text="Clear current map", command=self.clear_current_map).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(button_row2, text="Save JSON backup", command=self.save_json_backup).pack(side="left", fill="x", expand=True)

        help_text = (
            "Left click: add icon\n"
            "Right click: erase nearest icon\n"
            "Ctrl+Z: undo\n\n"
            "Export writes a snippet you can paste into FIXED_POINTS_BY_RAREMAP.\n"
            "If text is filled and 'Include text' is checked, the entry will include a 'text' field."
        )
        ttk.Label(left, text=help_text, justify="left").pack(anchor="w", pady=(8, 8))

        ttk.Label(left, text="Entries on current map").pack(anchor="w")
        self.entry_list = tk.Listbox(left, width=46, height=28)
        self.entry_list.pack(fill="both", expand=True)

        # Canvas
        self.canvas = tk.Canvas(right, width=DISPLAY_W, height=DISPLAY_H, bg="#202020", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def load_icon_cache(self) -> None:
        for icon_name in self.icons:
            path = ICONS_DIR / f"{file_safe_name(icon_name)}.png"
            try:
                img = Image.open(path).convert("RGBA")
                self.icon_images_full[icon_name] = img
            except Exception:
                continue

    def on_map_changed(self, _event=None) -> None:
        map_name = self.current_map_name.get().strip()
        self.load_background(map_name)
        self.refresh_entry_list()

    def load_background(self, map_name: str) -> None:
        path = BACKGROUNDS_DIR / f"{file_safe_name(map_name)}.png"
        if not path.exists():
            self.current_background_original = None
            self.current_background_display = None
            self.current_tk_background = None
            self.canvas.delete("all")
            self.status_var.set(f"Missing background: {path}")
            return

        img = Image.open(path).convert("RGBA")
        self.current_background_original = img
        self.current_background_display = img.resize((DISPLAY_W, DISPLAY_H), Image.Resampling.LANCZOS)
        self.current_tk_background = ImageTk.PhotoImage(self.current_background_display)
        self.current_scale_x = DISPLAY_W / IMAGE_W
        self.current_scale_y = DISPLAY_H / IMAGE_H
        self.redraw_canvas()
        self.status_var.set(f"Loaded background: {map_name}")

    def redraw_canvas(self) -> None:
        self.canvas.delete("all")
        if self.current_tk_background is not None:
            self.canvas.create_image(0, 0, anchor="nw", image=self.current_tk_background)

        map_name = self.current_map_name.get().strip()
        for idx, entry in enumerate(self.entries_by_map.get(map_name, [])):
            self.draw_entry(entry, idx)

    def draw_entry(self, entry: dict, idx: int) -> None:
        px, py = world_to_image(entry["x"], entry["z"])
        sx = px * self.current_scale_x
        sy = py * self.current_scale_y

        icon_name = entry["icon"]
        icon_img = self.icon_images_full.get(icon_name)
        thumb_size = 28
        if icon_img is not None:
            thumb = icon_img.copy()
            thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            tk_thumb = ImageTk.PhotoImage(thumb)
            key = f"{icon_name}_{idx}_{sx:.1f}_{sy:.1f}"
            self.icon_thumbs[key] = tk_thumb
            self.canvas.create_image(sx, sy, image=tk_thumb)
        else:
            self.canvas.create_oval(sx - 8, sy - 8, sx + 8, sy + 8, outline="yellow", width=2)

        self.canvas.create_rectangle(sx - 2, sy - 2, sx + 2, sy + 2, fill="red", outline="")

        if entry.get("text", ""):
            self.canvas.create_text(sx, sy - 18, text=entry["text"], fill="white", font=("Arial", 10, "bold"))

    def on_left_click(self, event) -> None:
        map_name = self.current_map_name.get().strip()
        icon_name = self.selected_icon_name.get().strip()
        if map_name == "" or icon_name == "":
            self.status_var.set("Select a map and an icon first")
            return

        image_x = event.x / self.current_scale_x
        image_y = event.y / self.current_scale_y
        world_x, world_z = image_to_world(image_x, image_y)

        entry = {
            "x": round(world_x, 3),
            "z": round(world_z, 3),
            "icon": icon_name,
            "text": self.text_value.get().strip(),
            "include_text": bool(self.include_text_var.get()),
        }

        self.entries_by_map.setdefault(map_name, []).append(entry)
        self.undo_stack.append({"action": "add", "map": map_name, "entry": entry})
        self.redraw_canvas()
        self.refresh_entry_list()
        self.status_var.set(f"Added {icon_name} at X={entry['x']}, Z={entry['z']}")

    def on_right_click(self, event) -> None:
        map_name = self.current_map_name.get().strip()
        entries = self.entries_by_map.get(map_name, [])
        if not entries:
            return

        nearest_idx = None
        nearest_dist = None

        for idx, entry in enumerate(entries):
            px, py = world_to_image(entry["x"], entry["z"])
            sx = px * self.current_scale_x
            sy = py * self.current_scale_y
            dist = math.hypot(event.x - sx, event.y - sy)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx

        if nearest_idx is None or nearest_dist is None or nearest_dist > ERASE_RADIUS_PX:
            self.status_var.set("No nearby icon to erase")
            return

        removed = entries.pop(nearest_idx)
        self.undo_stack.append({"action": "remove", "map": map_name, "entry": removed, "index": nearest_idx})
        self.redraw_canvas()
        self.refresh_entry_list()
        self.status_var.set(f"Removed {removed['icon']} at X={removed['x']}, Z={removed['z']}")

    def undo_last_action(self, _event=None) -> None:
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return

        action = self.undo_stack.pop()
        map_name = action["map"]
        entries = self.entries_by_map.setdefault(map_name, [])

        if action["action"] == "add":
            for i in range(len(entries) - 1, -1, -1):
                if entries[i] is action["entry"] or entries[i] == action["entry"]:
                    entries.pop(i)
                    break
            self.status_var.set("Undid add")
        elif action["action"] == "remove":
            index = action.get("index", len(entries))
            entries.insert(min(index, len(entries)), action["entry"])
            self.status_var.set("Undid remove")

        self.redraw_canvas()
        self.refresh_entry_list()

    def clear_current_map(self) -> None:
        map_name = self.current_map_name.get().strip()
        old_entries = list(self.entries_by_map.get(map_name, []))
        if not old_entries:
            return
        self.entries_by_map[map_name] = []
        self.undo_stack.append({"action": "restore_many", "map": map_name, "entries": old_entries})
        self.redraw_canvas()
        self.refresh_entry_list()
        self.status_var.set(f"Cleared {map_name}")

    def refresh_entry_list(self) -> None:
        self.entry_list.delete(0, tk.END)
        map_name = self.current_map_name.get().strip()
        for idx, entry in enumerate(self.entries_by_map.get(map_name, []), start=1):
            text_part = f", text={entry['text']}" if entry.get("text") else ""
            self.entry_list.insert(
                tk.END,
                f"{idx:02d}. {entry['icon']} @ X={entry['x']:.3f}, Z={entry['z']:.3f}{text_part}"
            )

    def format_python_dict_snippet(self) -> str:
        lines = ["FIXED_POINTS_BY_RAREMAP = {"]
        for map_name in sorted(self.entries_by_map.keys()):
            entries = self.entries_by_map[map_name]
            if not entries:
                continue
            lines.append(f'    "{map_name}": [')
            for entry in entries:
                pieces = [
                    f'"x": {entry["x"]}',
                    f'"z": {entry["z"]}',
                    f'"icon": "{entry["icon"]}"',
                ]
                if entry.get("include_text") and entry.get("text", "") != "":
                    pieces.append(f'"text": "{entry["text"]}"')
                lines.append("        {" + ", ".join(pieces) + "},")
            lines.append("    ],")
        lines.append("}")
        return "\n".join(lines)

    def format_json_text(self) -> str:
        payload = {}
        for map_name in sorted(self.entries_by_map.keys()):
            entries = self.entries_by_map[map_name]
            if not entries:
                continue
            payload[map_name] = []
            for entry in entries:
                out = {
                    "x": entry["x"],
                    "z": entry["z"],
                    "icon": entry["icon"],
                }
                if entry.get("include_text") and entry.get("text", "") != "":
                    out["text"] = entry["text"]
                payload[map_name].append(out)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def export_txt(self) -> None:
        default_name = OUTPUT_DIR / "fixed_points_export.txt"
        path = filedialog.asksaveasfilename(
            title="Export TXT",
            initialdir=str(OUTPUT_DIR),
            initialfile=default_name.name,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        if self.output_style_var.get() == "json":
            content = self.format_json_text()
        else:
            content = self.format_python_dict_snippet()

        Path(path).write_text(content, encoding="utf-8")
        self.status_var.set(f"Exported: {path}")
        messagebox.showinfo("Export complete", f"Saved:\n{path}")

    def save_json_backup(self) -> None:
        path = OUTPUT_DIR / "fixed_points_backup.json"
        payload = self.entries_by_map
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"Saved backup: {path}")


def main() -> None:
    root = tk.Tk()
    app = FixedIconTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
