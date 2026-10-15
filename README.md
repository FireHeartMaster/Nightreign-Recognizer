# Nightreign Recognizer

Nightreign Recognizer is a Windows desktop tool that identifies the currently displayed **Elden Ring Nightreign** map variation from a screenshot. It captures the selected monitor, isolates the in-game map panel, combines computer-vision and machine-learning classifiers to rank matching map layouts, and displays a completed reference map. It can also place the result in a configurable transparent overlay for use while playing.

This repository is both the application source and its research workspace. The current packaged application is **v41**; its main GUI source retains its historical v39 filename. Many older `v*`, `experimental_*`, `tmp_*`, training, rendering, and evaluation files preserve the iterations that led to it.

> This is an independent fan project. It is not affiliated with or endorsed by FromSoftware, Bandai Namco, or the Nightreign Map Router project.

## What the application does

- Captures the primary or selected monitor with `mss`.
- Locates and crops the circular Nightreign map panel with OpenCV.
- Classifies the active map region (the Shifting Earth).
- Classifies fixed landmark slots and region-specific visual states.
- Ranks candidate pattern/layout IDs using the observed slots and learned layout models.
- Detects supporting details such as Evergaols and status icons.
- Renders or loads a completed map with POIs, labels, and night-location information.
- Shows the best result and alternatives in a Tkinter GUI.
- Provides a configurable global recognition hotkey and transparent overlay hotkey.
- Supports completion notifications, overlay opacity, text-only overlays, and an optional dual-shot workflow.

The default hotkeys in source are:

- Recognition: `Ctrl + Alt + M`
- Show/hide overlay: `Ctrl + Alt + O`
- Hide overlay only: `Ctrl + Alt + H` (does nothing when the overlay is hidden)

All three shortcuts can be reassigned in the hotkey settings. The hide-only shortcut is saved between sessions and works while another window has focus.

Saved values in `nightreign_settings.json` override these defaults.

## How recognition works

The current pipeline is hybrid rather than a single end-to-end neural network:

1. **Screen capture** — the GUI captures one monitor and keeps the same frame for both preview and inference.
2. **Map-panel extraction** — Hough-circle detection searches the right side of the screenshot, estimates the surrounding panel, refines the crop, and normalizes it to the expected geometry.
3. **Shifting Earth classification** — the default v2 earth selector uses a serialized scikit-learn model. A CNN/legacy terrain scorer remains available as a fallback or experimental path.
4. **Slot inference** — PyTorch classifiers inspect known landmark positions for churches, forts, ruins, great churches, the main encampment, Sorcerer's Rises, empty slots, and region-specific variants.
5. **Layout inference** — per-earth layout classifiers and the slot recognition package score compatible layout/pattern IDs. Great Hollow has additional parallel slot classifiers and specialized rules because its variations are more ambiguous.
6. **Detail recognition** — dedicated models and template matching detect Evergaols, landmark status variants, and map icons.
7. **Result composition** — the highest-ranked layouts are associated with reference renders. `render_bosses_on_backgrounds.py` adds POIs, labels, night bosses, and optional text-only overlay content.
8. **Presentation** — the GUI displays the result, alternatives, logs, a completion toast, and an optional transparent click-through overlay.

The older backend in `recognize_nightreign_map_proto_v12_debugplus_radiusscale_fix.py` documents the geometric foundation clearly: it compares normalized terrain references, classifies six landmark families, refines uncertain slots through layout consensus, and computes a weighted score for every compatible layout. The current GUI extends that base with the newer learned earth, slot, status, and per-earth layout classifiers.

## Current entry points

| File | Role |
| --- | --- |
| `nightreign_map_recognizer_gui_ai_v39_completion_toast_overlay.py` | Current desktop application and runtime orchestrator (historical filename) |
| `nightreign_v41.spec` | Current PyInstaller build definition and packaged asset manifest |
| `build_flavor.py` | Selects the `v41` or `DEMO` build flavor and whitelist behavior |
| `recognize_nightreign_map_proto_v12_debugplus_radiusscale_fix.py` | Geometric recognizer loaded by the GUI as its compatibility/backend layer |
| `render_bosses_on_backgrounds.py` | Builds completed reference images and transparent text overlays |
| `render_nightreign_router_map.py` | Production batch renderer for map layouts |
| `render_nightreign_router_map_proto.py` | Renderer that can also export the `.npz` recognition package |
| `icon_recognizer_v4.py` | OpenCV template-based icon/status recognition |
| `Testing/run_parallel_panel_classifiers.py` | Runs the specialized slot/status classifiers used by the current pipeline |

`nightreign_map_recognizer_gui.py`, `recognize_nightreign_map_proto.py`, and the many numbered files are earlier stable points or experiments. `__main__.py` still imports the v38 GUI and should be treated as a legacy wrapper; run the current GUI file listed above directly.

## Repository layout

```text
Nightreign-Recognizer/
├── nightreign_map_recognizer_gui_ai_v39_completion_toast_overlay.py
├── nightreign_v41.spec
├── build_flavor.py
├── recognize_nightreign_map_proto_v12_debugplus_radiusscale_fix.py
├── render_nightreign_router_map.py
├── render_nightreign_router_map_proto.py
├── render_bosses_on_backgrounds.py
├── icon_recognizer_v4.py
├── nightreign_geom_proto_v4_slots.npz
├── nightreign_best_tuned_config*.json
├── nightreign_settings.json
├── reference_material/pattern_layouts/   # layout metadata copied from Router data
├── public/ and src/                      # selected Nightreign Map Router assets/source
├── maps/ and POI_icons/                  # runtime map tiles and icons
├── great_hollow_pattern_sets/            # Great Hollow pattern metadata
├── Train_Earth_Selector_v2/              # earth-selector training/inference work
├── Testing/                              # active and historical slot classifiers
├── renders*/                             # generated reference-map sets
├── *_ai_test_output*/                    # serialized PyTorch models and experiments
├── tmp_*/                                # evaluation runs, debug output, and prototypes
├── build/ and dist/                      # local PyInstaller output
└── README_v38_DETAILED.md                # detailed v38 algorithms and data reference
```

The repository is intentionally artifact-heavy. A large part of it consists of trained model weights, generated images, evaluation reports, extracted game tables, and one-off scripts used to investigate difficult layouts. For application work, begin with the files in the table above and the paths listed in `nightreign_v41.spec`.

## Requirements

The current application is Windows-oriented. Its global hotkeys, transparent overlay behavior, taskbar identity, and PyInstaller build use Windows APIs. The v41 release was built and tested with **Python 3.13.1** and **pip 24.3.1**.

Core runtime dependencies inferred from the active source are:

- Python 3
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- `mss`
- PyTorch and torchvision
- scikit-learn
- Tkinter (included with the standard Windows Python installer)

PyInstaller is additionally required to create the executable. Some build metadata also names `torchaudio`, although the tested v41 environment does not install it and the main runtime path does not import it.

[`requirements.txt`](requirements.txt) pins the complete runtime and PyInstaller dependency set from the environment used to build v41. It uses the official PyTorch CUDA 12.6 package index to reproduce the packaged GPU-enabled build.

Create the environment and install the locked dependencies with:

```powershell
# Confirm that the Python installation includes its required Tcl/Tk files.
py -3.13 -c "import tkinter as tk; print(tk.Tcl().eval('info patchlevel'))"

py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install pip==24.3.1
python -m pip install -r requirements.txt
```

The Tcl/Tk check should print a version such as `8.6.15`. If it reports that `init.tcl` cannot be found, repair or reinstall Python with the **Tcl/Tk and IDLE** component before building; Tkinter and its Tcl/Tk runtime are supplied by Python rather than `pip`.

The application automatically uses CUDA when `torch.cuda.is_available()` is true and otherwise runs on CPU. A different PyTorch build can be installed deliberately for another CUDA version or for CPU-only use, but that environment will differ from the locked v41 build environment.

## Running from source

Run commands from the repository root because model and data discovery includes repository-relative paths.

```powershell
python nightreign_map_recognizer_gui_ai_v39_completion_toast_overlay.py
```

Typical use:

1. Start the application and select the monitor containing Nightreign.
2. Open the in-game map and leave it unobstructed.
3. Click **Start recognition** or use the configured recognition hotkey.
4. Check that the captured-map preview contains the complete map panel.
5. Review the selected layout and top alternatives.
6. Use the overlay hotkey to show or hide the result over the game.

Use **Up/Left** for the previous thumbnail and **Down/Right** for the next, even when the game or another window has focus. Navigation stops at the first and last thumbnails. Changing selection updates both the main image and the visible overlay. Each press moves once; use the arrows without Ctrl, Alt, Shift, or Win. Arrow input also continues to reach the foreground application.

The GUI writes runtime messages to `recognition.log` and persists user preferences in `nightreign_settings.json`.

For an end-user-oriented walkthrough, see `README_end_user_exe.md`.

## Building the Windows application

The v41 spec packages the GUI, icons, map assets, recognition metadata, scikit-learn models, PyTorch weights, specialized slot classifiers, and the Torch DLL runtime hook.

1. Set the desired flavor in `build_flavor.py`:

   ```python
   BUILD_FLAVOR = "v41"  # or "DEMO"
   ```

2. Build from the repository root:

   ```powershell
   pyinstaller nightreign_v41.spec
   ```

3. Find the one-directory application under:

   ```text
   dist/NightreignRecognizer_v41/
   ```

The demo flavor builds as `NightreignRecognizer_DEMO` and enables a pattern-ID whitelist by default. `PATTERN_ID_WHITELIST_OVERRIDE` in `build_flavor.py` can explicitly force that filter on or off.

## Rendering and regenerating recognition data

The renderers reproduce Nightreign Map Router-style maps from the copied layout JSON, map tiles, POI icons, coordinate mappings, and selected TypeScript rendering rules in `src/`.

Render all maps with and without labels:

```powershell
python render_nightreign_router_map.py --mode both --out-dir renders
```

Render selected layouts:

```powershell
python render_nightreign_router_map.py --layout 1005 1007 --text on --ext webp
```

Export the geometric slot-recognition package and per-layout metadata:

```powershell
python render_nightreign_router_map_proto.py `
  --mode both `
  --export-recognition-package nightreign_geom_proto_v4_slots.npz `
  --export-recognition-dir recognition_metadata_v4_slots
```

The renderers expect these copied Router paths to remain available:

- `reference_material/pattern_layouts/`
- `public/assets/maps/`
- `public/POI_icons/`
- `src/app/_components/MapCanvas.tsx`
- `src/utils/poiIconMapping.ts`
- `src/utils/poiLocationMapping.ts`

Additional scripts such as `generate_layout_json_from_game_csvs.py`, `build_mappatternset_bundle_json.py`, and the `render_*game_csv*` family transform extracted game CSV tables into layout JSON, grouped pattern data, and rendered training/reference images.

## How the project was created

The repository history and retained artifacts show an iterative, experiment-driven development process:

1. **Router data was copied locally.** Selected Nightreign Map Router TypeScript files, coordinate data, icons, SVGs, map tiles, and layout JSON became the rendering reference inside this repository.
2. **Python renderers reproduced the map.** `render_nightreign_router_map.py` mirrors important `MapCanvas.tsx` filters, coordinate mappings, icon rules, and label placement so complete reference images can be generated offline.
3. **A geometric recognizer established the pipeline.** Early prototypes used Hough-circle panel extraction, terrain/template correlation, fixed landmark slots, and weighted layout scoring. The renderer exported `nightreign_geom_proto_v4_slots.npz` to connect known layouts to recognition-time geometry.
4. **Models were trained for ambiguous stages.** The project added PyTorch slot, status, and Evergaol CNNs; a scikit-learn earth selector; per-earth layout classifiers; and specialized Great Hollow classifiers. The numerous `experimental_*`, `Testing/`, `Train_Earth_Selector_v2/`, and `tmp_*` paths are the surviving training and evaluation work.
5. **The GUI evolved through numbered snapshots.** The progression from the unversioned GUI through hybrid, tuned, AI, overlay, logging, animation, toast, v38, and v39 files records features being layered onto the recognizer without discarding working checkpoints.
6. **PyInstaller turned the workspace into a distributable app.** The `.spec` files enumerate the exact model and data assets required by each release. `build_flavor.py` later centralized demo/full-build behavior.

Git records the initial repository on **2 April 2026**, the first base application/assets in the next commit, a major GUI/CLI and Great Hollow checkpoint on **16 April**, and rapid overlay and optimization work through **25 April 2026**. The configured origin is `FireHeartMaster/Nightreign-Recognizer` on GitHub.

## Development notes

The [archived v38-era technical reference](docs/legacy/technical-reference-v38.md) preserves the previous README, including its detailed algorithms, CSV schemas, configuration examples, and troubleshooting guidance. It is historical and contains outdated information; verify it against the current source before use.

- There is no conventional automated unit-test suite. Validation is primarily performed through smoke scripts, batch evaluations, saved JSON reports, debug overlays, and screenshot datasets.
- Most `tmp_*` directories are experiment outputs rather than runtime dependencies. Do not assume one is disposable without checking `nightreign_v41.spec` and the current GUI source; several trained models currently live under historically named temporary directories.
- The active GUI file contains multiple generations of functions, with later definitions intentionally overriding earlier compatibility implementations. Search for the final definition when tracing behavior.
- `nightreign_settings.json`, log files, generated renders, model outputs, and local build directories may change during normal use.
- Existing detailed documents are mostly v38-era references. `README_v38_DETAILED.md` remains useful for formulas, tuning parameters, and CSV relationships, but the current GUI source and v41 spec are authoritative for the packaged application.

## Data and legal notice

This repository contains or expects game-derived tables, map images, icons, and data copied from another project. No license file is currently present. Before redistributing the source, assets, trained models, or packaged executable, verify the licensing and redistribution terms for every third-party and game-derived component.
