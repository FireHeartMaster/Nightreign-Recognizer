#!/usr/bin/env python3
import os
import sys
import cv2
import time
import threading
from pathlib import Path

# Set environment variables for GH model-match flags
os.environ["GH_USE_PARALLEL_PANEL_CLASSIFIER"] = "1"
os.environ["GH_LAYOUT_MODEL_MATCH_MODE"] = "models"

# Add the current directory to sys.path if needed
sys.path.insert(0, str(Path(__file__).parent))

from nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay import RunContext, run_recognition

def main():
    image_path = r"C:\Users\eduar\Documents\CodeProjects\Nightreign-Recognizer\1183_.png"
    
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return 1
    
    print(f"Loaded image: {image_path}, shape: {img.shape}")
    
    # Create a RunContext with a dummy cancel event
    cancel_event = threading.Event()
    ctx = RunContext(run_id=1, cancel_event=cancel_event, log_func=lambda run_id, msg: print(f"[LOG] {msg}"))
    
    # Run recognition
    start_time = time.perf_counter()
    try:
        payload = run_recognition(ctx, monitor_index=1, pre_captured_full_img=img)
        end_time = time.perf_counter()
        
        print(f"\nRecognition completed in {end_time - start_time:.3f} seconds")
        
        # Print results
        results = payload["results"]
        print(f"\nTop {len(results)} results:")
        for i, r in enumerate(results[:20], start=1):  # Top 20
            res = r.details  # r.details is the dict
            print(f"{i:>2}. #{r.number:<3} score={r.score:.6f} earth={res.get('shifting_earth', 'N/A'):<14} exactW={res.get('exact_match_weight', 0):.2f} mismW={res.get('hard_mismatch_weight', 0):.2f} unkW={res.get('unknown_weight', 0):.2f}")
        
        print(f"\nTotal execution time: {end_time - start_time:.3f}s")
        
    except Exception as e:
        print(f"Error during recognition: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())