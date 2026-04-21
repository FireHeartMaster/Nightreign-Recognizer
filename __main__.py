#!/usr/bin/env python3
"""Wrapper to start the app with better error reporting."""
import sys
import traceback

try:
    print("Starting app initialization...", file=sys.stderr, flush=True)
    
    # Import the main module
    from nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay import main
    
    print("Calling main()...", file=sys.stderr, flush=True)
    main()
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
