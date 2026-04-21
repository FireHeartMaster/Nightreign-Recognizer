import argparse
import threading
from pathlib import Path

import nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay as v38


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v38 recognition headlessly on a saved image.")
    parser.add_argument("input", type=Path, help="Path to the input image to analyze.")
    parser.add_argument("--monitor-index", type=int, default=1, help="Fake monitor index to use.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input image not found: {args.input}")

    image = v38.load_bgr(args.input)

    def fake_log(run_id: int, message: str) -> None:
        print(message)

    ctx = v38.RunContext(run_id=1, cancel_event=threading.Event(), log_func=fake_log)
    v38.get_available_monitors = lambda: [
        (
            args.monitor_index,
            {"width": image.shape[1], "height": image.shape[0], "left": 0, "top": 0},
        )
    ]

    result = v38.run_recognition(
        ctx,
        monitor_index=args.monitor_index,
        pre_captured_full_img=image,
    )

    print("\n=== TIMINGS ===")
    for key, value in result["backend_payload"]["timings"].items():
        print(f"{key}: {value:.4f}s")

    print("\n=== TOP RESULTS ===")
    for idx, item in enumerate(result["results"][:5], start=1):
        details = item.details if hasattr(item, "details") else item["details"]
        print(f"{idx}: layout {item.number} earth={details.get('shifting_earth')} score={item.score:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
