import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


PANEL_SCALE_FOR_RADIUS = 1.12
INPUT_SIZE = 256


def list_panel_sources() -> List[Path]:
    roots = [
        Path("renders_new/no_text"),
        Path("renders/recognizer_no_text"),
        Path("renders/no_text"),
    ]
    out: List[Path] = []
    for root in roots:
        if root.exists():
            out.extend(sorted(root.glob("layout_*.png")))
    if not out:
        raise RuntimeError("No panel sources found in renders folders.")
    return out


def _rand_canvas_size() -> Tuple[int, int]:
    options = [(1920, 1080), (2560, 1440), (3440, 1440), (1680, 1050)]
    return random.choice(options)


def _load_img(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def synthetic_sample(panel_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    w, h = _rand_canvas_size()
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    base = np.random.randint(10, 60)
    canvas[:] = (base, base + random.randint(0, 10), base + random.randint(5, 30))

    # Use map textures as coarse background noise.
    bg = _load_img(random.choice(panel_paths))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
    mix = random.uniform(0.15, 0.45)
    canvas = cv2.addWeighted(canvas, 1.0 - mix, bg, mix, 0.0)

    side = int(random.uniform(0.58, 0.78) * min(w, h))
    cx = int(random.uniform(0.66, 0.84) * w)
    cy = int(random.uniform(0.42, 0.60) * h)
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    x1 = x0 + side
    y1 = y0 + side

    panel = _load_img(random.choice(panel_paths))
    panel = cv2.resize(panel, (side, side), interpolation=cv2.INTER_CUBIC)
    canvas[y0:y1, x0:x1] = panel

    # Light blur/noise augmentation.
    if random.random() < 0.5:
        k = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), random.uniform(0.5, 1.2))
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(3, 10), canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Circle target in screenshot pixel space.
    true_cx = x0 + side * 0.5 + random.uniform(-0.015, 0.015) * side
    true_cy = y0 + side * 0.5 + random.uniform(-0.015, 0.015) * side
    true_r = side / (2.0 * PANEL_SCALE_FOR_RADIUS) * random.uniform(0.98, 1.02)

    # Normalize labels to [0,1] domain.
    label = np.array(
        [
            true_cx / float(w),
            true_cy / float(h),
            true_r / float(min(w, h)),
        ],
        dtype=np.float32,
    )
    return canvas, label


class CircleRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def preprocess_for_model(img_bgr: np.ndarray) -> np.ndarray:
    x = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return x


def train_model(model_out: Path, steps: int, batch_size: int, lr: float) -> None:
    panel_paths = list_panel_sources()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircleRegressor().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for step in range(1, steps + 1):
        xs = []
        ys = []
        for _ in range(batch_size):
            img, label = synthetic_sample(panel_paths)
            xs.append(preprocess_for_model(img))
            ys.append(label)
        xb = torch.from_numpy(np.stack(xs)).to(device)
        yb = torch.from_numpy(np.stack(ys)).to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == 1:
            print(f"step {step}/{steps} loss={float(loss.detach().cpu()):.6f}", flush=True)

    payload = {
        "state_dict": model.state_dict(),
        "input_size": INPUT_SIZE,
        "panel_scale_for_radius": PANEL_SCALE_FOR_RADIUS,
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_out)
    print(f"saved model: {model_out}")


def predict_circle(model_path: Path, image_path: Path, out_overlay: Path | None = None) -> Tuple[float, float, float]:
    img = _load_img(image_path)
    h, w = img.shape[:2]
    payload = torch.load(model_path, map_location="cpu")
    model = CircleRegressor()
    model.load_state_dict(payload["state_dict"])
    model.eval()

    x = preprocess_for_model(img)
    xb = torch.from_numpy(x[None, ...])
    with torch.no_grad():
        pred = model(xb).cpu().numpy()[0]
    cx = float(pred[0] * w)
    cy = float(pred[1] * h)
    r = float(pred[2] * min(w, h))

    if out_overlay is not None:
        vis = img.copy()
        cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"pred cx={cx:.1f} cy={cy:.1f} r={r:.1f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        out_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_overlay), vis)
        print(f"saved overlay: {out_overlay}")
    return cx, cy, r


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experimental circle center/radius regressor (test only).")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model-out", type=Path, default=Path("tmp_circle_regressor_test/circle_regressor_v1.pt"))
    t.add_argument("--steps", type=int, default=1200)
    t.add_argument("--batch-size", type=int, default=24)
    t.add_argument("--lr", type=float, default=1e-3)

    q = sub.add_parser("predict")
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--image", type=Path, required=True)
    q.add_argument("--overlay-out", type=Path, default=Path("tmp_circle_regressor_test/pred_overlay.png"))
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "train":
        train_model(args.model_out, args.steps, args.batch_size, args.lr)
        return 0
    if args.cmd == "predict":
        cx, cy, r = predict_circle(args.model, args.image, args.overlay_out)
        print(f"predicted_circle cx={cx:.3f} cy={cy:.3f} r={r:.3f}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
