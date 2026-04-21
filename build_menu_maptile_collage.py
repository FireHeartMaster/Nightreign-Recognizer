#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image

# Expected filename pattern:
# MENU_MapTile_L0_YY_XX_webp.webp
# Corrected placement rule:
#   column = YY (left -> right)
#   row    = max_xx - XX (bottom -> top)
FILENAME_RE = re.compile(
    r"^MENU_MapTile_L(?P<level>\d+)_(?P<yy>\d{2})_(?P<xx>\d{2})_webp\.webp$",
    re.IGNORECASE,
)


TileKey = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a collage from MENU_MapTile tiles in an input folder."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing files like MENU_MapTile_L0_YY_XX_webp.webp",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("collage.png"),
        help="Output image path. Default: collage.png",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Tile level to include. Default: 0",
    )
    parser.add_argument(
        "--background",
        default=None,
        help="Optional background color, e.g. '#000000' or 'white'. Defaults to transparent for RGBA output, black otherwise.",
    )
    return parser.parse_args()


def collect_tiles(input_folder: Path, level: int) -> Dict[TileKey, Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    tiles: Dict[TileKey, Path] = {}
    for path in sorted(input_folder.iterdir()):
        if not path.is_file():
            continue

        match = FILENAME_RE.match(path.name)
        if not match:
            continue

        file_level = int(match.group("level"))
        if file_level != level:
            continue

        yy = int(match.group("yy"))
        xx = int(match.group("xx"))
        tiles[(yy, xx)] = path

    if not tiles:
        raise ValueError(
            f"No matching L{level} files found in {input_folder}. "
            f"Expected names like MENU_MapTile_L{level}_YY_XX_webp.webp"
        )

    return tiles


def infer_grid_size(tiles: Dict[TileKey, Path]) -> Tuple[int, int, int, int]:
    ys = [yy for yy, _ in tiles]
    xs = [xx for _, xx in tiles]
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)
    return min_y, max_y, min_x, max_x


def choose_canvas_mode(output_path: Path, sample: Image.Image) -> str:
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "RGB"
    if sample.mode == "RGBA":
        return "RGBA"
    return "RGB"


def build_collage(
    tiles: Dict[TileKey, Path],
    output_path: Path,
    background: str | None,
) -> Path:
    sample_path = next(iter(tiles.values()))
    with Image.open(sample_path) as sample:
        tile_w, tile_h = sample.size
        sample_mode = sample.mode

    min_y, max_y, min_x, max_x = infer_grid_size(tiles)
    cols = max_y - min_y + 1
    rows = max_x - min_x + 1

    canvas_mode = choose_canvas_mode(output_path, Image.new(sample_mode, (1, 1)))
    if canvas_mode == "RGBA":
        bg = (0, 0, 0, 0) if background is None else background
    else:
        bg = "black" if background is None else background

    canvas = Image.new(canvas_mode, (cols * tile_w, rows * tile_h), bg)

    for (yy, xx), path in tiles.items():
        col = yy - min_y
        row = max_x - xx  # corrected rule: XX increases bottom -> top

        with Image.open(path) as tile:
            if canvas_mode == "RGB" and tile.mode != "RGB":
                tile = tile.convert("RGB")
            elif canvas_mode == "RGBA" and tile.mode != "RGBA":
                tile = tile.convert("RGBA")

            canvas.paste(tile, (col * tile_w, row * tile_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"} and canvas.mode != "RGB":
        canvas = canvas.convert("RGB")
    canvas.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    tiles = collect_tiles(args.input_folder, args.level)
    output_path = build_collage(tiles, args.output, args.background)
    print(f"Saved collage to: {output_path}")


if __name__ == "__main__":
    main()
