import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

# --------------------------------------------------
# Paths
# --------------------------------------------------
json_path = Path(r"C:\Users\eduar\Documents\CodeProjects\Nightreign_smithbox_new_project\ParamEditorExports\LotResultSmallBaseAndSpot_grouped_by_patternId_4.json")
backgrounds_dir = Path(r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\maps")
icons_dir = Path(r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\POI_icons")
output_dir = Path(r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\renders_new")

output_dir.mkdir(parents=True, exist_ok=True)

missing_icons_txt = output_dir / "missing_icons.txt"
used_icons_txt = output_dir / "used_icons.txt"

MISSING_ICON_FILES = set()
USED_ICON_FILES = set()

# --------------------------------------------------
# Coordinate system
# Image size: 1536 x 1536
# World bottom-left: (-256, -256)
# World center: (512, 512)
# World top-right: (1280, 1280)
# --------------------------------------------------
IMAGE_W = 1536
IMAGE_H = 1536
WORLD_MIN_X = -256
WORLD_MIN_Z = -256
WORLD_MAX_X = 1280
WORLD_MAX_Z = 1280

ICON_SCALE = 1.8
BASE_ICON_SIZE = 48

SKIP_TEXT_WHEN_ICON_MISSING = False

# --------------------------------------------------
# Which patternIds to render
# None means render all
# --------------------------------------------------
# PATTERN_IDS_TO_RENDER = set(range(1005, 1010))
PATTERN_IDS_TO_RENDER = {1099, 1100, 1105, 1110, 1115, 1116}

# Examples:
# PATTERN_IDS_TO_RENDER = {1005, 1006, 1007, 1008, 1009}
# PATTERN_IDS_TO_RENDER = set(range(1005, 1010))

FIXED_POINTS_BY_RAREMAP = {
   #  "Limveld": [
   #      {"x": 441.0, "z": 213.0, "icon": "Field Boss Red"},
   #      {"x": 500.0, "z": 300.0, "icon": "Church"},
   #  ],
   #  "Crater": [
   #      {"x": 620.0, "z": 410.0, "icon": "Castle"},
   #  ],
   #  "Noklateo": [
   #      {"x": 700.0, "z": 520.0, "icon": "Night Horde"},
   #  ],
   "Great Hollow": [
   #  {"x": 143.604, "z": 392.848, "icon": "Ruins"},
    {"x": 386.488, "z": 926.106, "icon": "Church"},
    {"x": 576.452, "z": 619.449, "icon": "Church"},
   ],
}

TEXT_TO_NOT_WRITE = {
    "Church",
    "Sorcerer's Rise",
}

def file_safe_name(name):
    return str(name).strip().replace(" ", "_")

def wrap_text_by_words(text, max_chars=16):
    text = str(text).strip()
    if text == "":
        return ""

    lines = text.splitlines()
    wrapped_lines = []

    for line in lines:
        line = line.strip()
        if line == "":
            wrapped_lines.append("")
            continue

        wrapped = textwrap.fill(
            line,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False
        )
        wrapped_lines.append(wrapped)

    return "\n".join(wrapped_lines)

def world_to_image(x, z):
    """
    Convert world-space (X, Z) to image-space (px, py).
    X grows left->right.
    Z grows bottom->top, so Y is inverted for the image.
    """
    px = (float(x) - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X) * IMAGE_W
    py = IMAGE_H - ((float(z) - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z) * IMAGE_H)

    px = max(0, min(IMAGE_W - 1, int(round(px))))
    py = max(0, min(IMAGE_H - 1, int(round(py))))
    return px, py

def load_font(size):
    """
    Tries a few common fonts, then falls back to PIL default.
    """
    candidates = [
        "arialbd.ttf",
        "Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def draw_label(draw, center_x, center_y, text, font, fill="white", stroke_fill="black", stroke_width=1, max_chars=16):
    wrapped_text = wrap_text_by_words(text, max_chars=max_chars)

    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped_text,
        font=font,
        align="center",
        spacing=4,
        stroke_width=stroke_width,
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = center_x - text_w / 2
    y = center_y - text_h / 2

    draw.multiline_text(
        (x, y),
        wrapped_text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        spacing=4,
        align="center",
    )

def get_label_bbox(draw, center_x, center_y, text, font, stroke_width=1, max_chars=16):
    wrapped_text = wrap_text_by_words(text, max_chars=max_chars)

    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped_text,
        font=font,
        align="center",
        spacing=4,
        stroke_width=stroke_width,
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = center_x - text_w / 2
    y = center_y - text_h / 2

    return wrapped_text, (x, y, x + text_w, y + text_h)

def boxes_overlap(a, b, padding=6):
    return not (
        a[2] + padding <= b[0] or
        b[2] + padding <= a[0] or
        a[3] + padding <= b[1] or
        b[3] + padding <= a[1]
    )

def find_non_overlapping_position(
    draw,
    center_x,
    center_y,
    text,
    font,
    placed_boxes,
    stroke_width=1,
    max_chars=16,
    padding=6,
    step=10,
    max_radius=180,
):
    # Try the original position first
    wrapped_text, bbox = get_label_bbox(
        draw, center_x, center_y, text, font,
        stroke_width=stroke_width,
        max_chars=max_chars
    )

    if not any(boxes_overlap(bbox, old_box, padding=padding) for old_box in placed_boxes):
        return center_x, center_y, bbox

    # Then search outward in rings
    for radius in range(step, max_radius + step, step):
        candidate_offsets = [
            (0, -radius),
            (radius, 0),
            (0, radius),
            (-radius, 0),
            (radius, -radius),
            (radius, radius),
            (-radius, radius),
            (-radius, -radius),
        ]

        for dx, dy in candidate_offsets:
            test_x = center_x + dx
            test_y = center_y + dy

            _, test_bbox = get_label_bbox(
                draw, test_x, test_y, text, font,
                stroke_width=stroke_width,
                max_chars=max_chars
            )

            if not any(boxes_overlap(test_bbox, old_box, padding=padding) for old_box in placed_boxes):
                return test_x, test_y, test_bbox

    # Fallback: keep original position if no free spot found
    return center_x, center_y, bbox

def paste_centered(base_image, overlay, center_x, center_y):
    x = int(center_x - overlay.width / 2)
    y = int(center_y - overlay.height / 2)
    base_image.paste(overlay, (x, y), overlay)

def render_boss(base_img, icon_img, boss_name, x_world, z_world, label_queue, label_dx=0, label_dy=0):
    if boss_name is None:
        return
    boss_name = str(boss_name).strip()
    if boss_name == "":
        return

    if x_world in ("", None) or z_world in ("", None):
        return

    px, py = world_to_image(x_world, z_world)

    paste_centered(base_img, icon_img, px, py)

    label_queue.append((px + label_dx, py + label_dy, boss_name))

# --------------------------------------------------
# NEW: icon cache and row rendering
# --------------------------------------------------
ICON_CACHE = {}

def normalize_row_icon_name(icon_name):
    icon_name = str(icon_name).strip()
    if icon_name == "":
        return ""

    # For now, treat both variants as the same icon
   #  lowered = icon_name.lower()
   #  if lowered in {"field boss", "field boss red"}:
   #      return "Field Boss"

    return icon_name

def get_row_icon(icon_name):
    icon_name = normalize_row_icon_name(icon_name)
    if icon_name == "":
        return None

    cache_key = icon_name
    if cache_key in ICON_CACHE:
        return ICON_CACHE[cache_key]

    # Special mapping: use Field_Boss_Red.png for Gold Merchant
    if icon_name == "Gold Merchant":
        icon_path = icons_dir / "Field_Boss_Red.png"
    else:
        icon_path = icons_dir / f"{file_safe_name(icon_name)}.png"

    if not icon_path.exists():
        print(f"Missing row icon: {icon_path.name}")
        MISSING_ICON_FILES.add(icon_path.name)
        ICON_CACHE[cache_key] = None
        return None

    USED_ICON_FILES.add(icon_path.name)

    icon_img = Image.open(icon_path).convert("RGBA")

    target_size = int(BASE_ICON_SIZE * ICON_SCALE)

    orig_w, orig_h = icon_img.size
    scale = target_size / max(orig_w, orig_h)

    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    icon_img = icon_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    ICON_CACHE[cache_key] = icon_img
    return icon_img



def render_fixed_points(base_img, rare_map, top_icon_queue):
    fixed_points = FIXED_POINTS_BY_RAREMAP.get(str(rare_map).strip(), [])

    for point in fixed_points:
        icon_name = str(point.get("icon", "")).strip()
        x_world = point.get("x", None)
        z_world = point.get("z", None)

        if icon_name == "" or x_world is None or z_world is None:
            continue

        icon_img = get_row_icon(icon_name)
        if icon_img is None:
            continue

        px, py = world_to_image(x_world, z_world)

        # If you want Basement Boss fixed points to stay on top too
        if icon_name == "Basement Boss":
            top_icon_queue.append((icon_img, px, py))
        else:
            paste_centered(base_img, icon_img, px, py)

def render_row_item(base_img, row, label_queue, top_icon_queue):
    icon_name = row.get("smallBaseLocationName", "")
    label_text = row.get("smallBaseVariationName", "")
    x_world = row.get("X", "")
    z_world = row.get("Z", "")

    icon_name = "" if icon_name is None else str(icon_name).strip()
    label_text = "" if label_text is None else str(label_text).strip()

    if icon_name == "":
        return
    if x_world in ("", None) or z_world in ("", None):
        return

    px, py = world_to_image(x_world, z_world)

    # Try to load icon, but do not force an early return unless the flag says so
    icon_img = get_row_icon(icon_name)

    if icon_img is not None:
        # Special case:
        # If this is Field Boss / Field Boss Red at world position X=441, Z=213.0,
        # move both icon and text left by icon width and down by icon height.
        try:
            x_val = float(x_world)
            z_val = float(z_world)
        except (TypeError, ValueError):
            x_val = None
            z_val = None

        if (
            icon_name in {"Field Boss", "Field Boss Red"}
            and x_val is not None
            and z_val is not None
            and abs(x_val - 441.0) < 1e-6
            and abs(z_val - 213.0) < 1e-6
        ):
            px -= icon_img.width / 2
            py += icon_img.height / 2

        # Basement Boss should render above all other icons
        if icon_name == "Basement Boss":
            px += icon_img.width
            top_icon_queue.append((icon_img, px, py))
        else:
            paste_centered(base_img, icon_img, px, py)

    elif SKIP_TEXT_WHEN_ICON_MISSING:
        return

    text_to_draw = label_text if label_text != "" else icon_name

    if text_to_draw != "" and icon_name not in TEXT_TO_NOT_WRITE:
        label_queue.append((px, py, text_to_draw))

def should_render_pattern(pattern_id):
    if PATTERN_IDS_TO_RENDER is None:
        return True

    try:
        pattern_id = int(pattern_id)
    except (TypeError, ValueError):
        return False

    return pattern_id in PATTERN_IDS_TO_RENDER

# --------------------------------------------------
# Load input JSON
# --------------------------------------------------
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# --------------------------------------------------
# Load shared icon and font
# --------------------------------------------------
boss_icon_path = icons_dir / "Night_location.png"
if not boss_icon_path.exists():
    raise FileNotFoundError(f"Missing boss icon: {boss_icon_path}")

USED_ICON_FILES.add(boss_icon_path.name)

boss_icon = Image.open(boss_icon_path).convert("RGBA")

target_size = int(BASE_ICON_SIZE * ICON_SCALE)

orig_w, orig_h = boss_icon.size
scale = target_size / max(orig_w, orig_h)

new_w = max(1, int(round(orig_w * scale)))
new_h = max(1, int(round(orig_h * scale)))

boss_icon = boss_icon.resize((new_w, new_h), Image.Resampling.LANCZOS)

title_font = load_font(28)
boss_font = load_font(22)

# --------------------------------------------------
# Render one image per patternId
# --------------------------------------------------
for entry in data:
    pattern_id = entry.get("patternId", "")
    rare_map = str(entry.get("rareMap", "")).strip()

    if not should_render_pattern(pattern_id):
        continue

    if rare_map == "":
        print(f"Skipping pattern {pattern_id}: rareMap is empty")
        continue

    background_path = backgrounds_dir / f"{file_safe_name(rare_map)}.png"
    if not background_path.exists():
        print(f"Skipping pattern {pattern_id}: missing background {background_path.name}")
        continue

    bg = Image.open(background_path).convert("RGBA")
    if bg.size != (IMAGE_W, IMAGE_H):
        bg = bg.resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(bg)
    label_queue = []
    top_icon_queue = [] 
    # Optional title
    title = f"Pattern {pattern_id} - {rare_map}"
    label_queue.append((20, 20, title, title_font)) 
    # Render fixed points for this Shifting Earth / rareMap
    render_fixed_points(
       base_img=bg,
       rare_map=rare_map,
       top_icon_queue=top_icon_queue,
    ) 
    # Render normal row entries
    for row in entry.get("rows", []):
       render_row_item(
          base_img=bg,
          row=row,
          label_queue=label_queue,
          top_icon_queue=top_icon_queue,
       )

    # Boss 1
    render_boss(
        base_img=bg,
        icon_img=boss_icon,
        boss_name=f"Night 1\n{entry.get('bossId1Name', '')}".strip(),
        x_world=entry.get("playArea1_X", ""),
        z_world=entry.get("playArea1_Z", ""),
        label_queue=label_queue,
        label_dx=0,
        label_dy=0,
    )

    # Boss 2
    render_boss(
        base_img=bg,
        icon_img=boss_icon,
        boss_name=f"Night 2\n{entry.get('bossId2Name', '')}".strip(),
        x_world=entry.get("playArea2_X", ""),
        z_world=entry.get("playArea2_Z", ""),
        label_queue=label_queue,
        label_dx=0,
        label_dy=0,
    )

    for icon_img, px, py in top_icon_queue:
      paste_centered(bg, icon_img, px, py)

    placed_label_boxes = []

    for item in label_queue:
        if len(item) == 4:
            x, y, text, font = item
        else:
            x, y, text = item
            font = boss_font

        final_x, final_y, final_bbox = find_non_overlapping_position(
            draw=draw,
            center_x=x,
            center_y=y,
            text=text,
            font=font,
            placed_boxes=placed_label_boxes,
            stroke_width=1,
            max_chars=16,
            padding=6,
            step=10,
            max_radius=180,
        )

        draw_label(draw, final_x, final_y, text, font)
        placed_label_boxes.append(final_bbox)

    out_path = output_dir / f"pattern_{int(pattern_id):04d}_{rare_map}.png"
    bg.save(out_path)
    print(f"Created: {out_path}")

with open(missing_icons_txt, "w", encoding="utf-8") as f:
    for name in sorted(MISSING_ICON_FILES):
        f.write(f"{name}\n")

with open(used_icons_txt, "w", encoding="utf-8") as f:
    for name in sorted(USED_ICON_FILES):
        f.write(f"{name}\n")

print(f"Saved missing icon list: {missing_icons_txt}")
print(f"Saved used icon list: {used_icons_txt}")

print("Done.")