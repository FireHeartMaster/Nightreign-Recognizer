import argparse
import json
from pathlib import Path
from typing import Dict

import torch

import layout_classifier_lib as lib


def infer_one(model_path: Path, image_path: Path) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size, meta = lib.load_checkpoint(model_path, device)
    result = lib.infer_proba(
        model=model,
        image_path=image_path,
        class_names=class_names,
        image_size=image_size,
        device=device,
    )
    result["model_path"] = str(model_path.resolve())
    result["class_names"] = class_names
    result["image_size"] = int(image_size)
    result["meta"] = meta
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer Slot 2 class for a 200x200 icon crop.")
    parser.add_argument("--model", default="slot2_status_cnn.pt", help="Checkpoint path.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    image_path = Path(args.input).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not image_path.exists():
        raise SystemExit(f"Input image not found: {image_path}")

    result = infer_one(model_path=model_path, image_path=image_path)
    text = json.dumps(result, indent=2)
    print(text)

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved JSON to: {out_path}")


if __name__ == "__main__":
    main()
