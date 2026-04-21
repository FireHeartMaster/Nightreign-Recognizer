import os
import sys
import threading
import cv2
import json

sys.path.insert(0, r'C:/Users/eduar/Documents/CodeProjects/Nightreign-Recognizer')
os.environ['GH_USE_PARALLEL_PANEL_CLASSIFIER'] = '1'
os.environ['GH_LAYOUT_MODEL_MATCH_MODE'] = 'models'
from nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay import RunContext, run_recognition

img = cv2.imread(r'C:/Users/eduar/Documents/CodeProjects/Nightreign-Recognizer/1183_.png')
ctx = RunContext(1, threading.Event(), lambda run_id, msg: None)
payload = run_recognition(ctx, monitor_index=1, pre_captured_full_img=img)

results = payload.get('results', [])
timings = payload.get('timings', {})

# Print top 5 results
print('\n=== TOP 5 RESULTS ===')
for idx, r in enumerate(results[:5], 1):
    res = r.details if hasattr(r, 'details') else r
    print(f'{idx}. Layout {res["layout_number"]} ({res["shifting_earth"]}) score={res["total_score"]:.6f}')
    print(f'   earth_score={res["earth_score"]:.6f}, gh_model_layout_prior={res.get("great_hollow_model_layout_prior", 0.0):.6f}, gh_status_prior={res.get("great_hollow_status_prior", 0.0):.6f}')

# Print timings
print('\n=== TIMING BREAKDOWN (seconds) ===')
for key in sorted(timings.keys()):
    if key != 'total_run':
        print(f'{key}: {timings[key]:.4f}s')
print(f'\nTOTAL RUN TIME: {timings.get("total_run", 0.0):.4f}s')

# Verify flags
print(f'\n=== FLAGS ===')
print(f'GH_USE_PARALLEL_PANEL_CLASSIFIER: {os.environ.get("GH_USE_PARALLEL_PANEL_CLASSIFIER")}')
print(f'GH_LAYOUT_MODEL_MATCH_MODE: {os.environ.get("GH_LAYOUT_MODEL_MATCH_MODE")}')
print(f'Image loaded: {img is not None}')
