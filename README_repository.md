# Nightreign Map Recognition and Rendering Toolkit

## Overview

This repository contains four main deliverables:

1. **Production renderer** for generating finished reference maps from the Nightreign Map Router data that has already been copied into this repository.
2. **Production GUI recognizer** for identifying the current in-game map from a live screenshot and showing the best completed reference image.
3. **Prototype renderer / package generator** for exporting metadata packages used by experimental recognizers.
4. **Prototype recognizer** for testing new recognition strategies, currently centered on a slot-based landmark model.

The project evolved through many versioned development files. The repository should keep the stable, non-versioned filenames listed below.

---

## Canonical scripts kept in the repository

### 1. `render_nightreign_router_map.py`
Production renderer.

Use this to:
- render reference maps from the Nightreign Map Router data copied into this repository
- output `png`, `jpg`, or `webp`
- generate no-text or text-on maps
- batch-render all layouts
- create numbered filenames for recognition image sets

This renderer now assumes the required Nightreign Map Router files already exist **inside this repository**. The `--repo` flag is optional and only needed if you want to point at a different root.

### 2. `nightreign_map_recognizer_gui.py`
Production GUI recognizer.

Use this as the main end-user application.

Current intended capabilities:
- monitor selection
- stable map crop from the live screen
- slots-based recognition backend
- top-result image plus top alternatives
- zoom and click-drag pan in the main image area
- configurable global hotkey from inside the app
- settings popover
- support for packaged `.webp` display images
- support for packaged icon files and Windows taskbar/window icon handling

### 3. `render_nightreign_router_map_proto.py`
Prototype renderer / package generator.

Use this to export experimental recognition packages.
The latest prototype line generates slot-based metadata packages for the prototype recognizer and the slots-based GUI recognizer.

Like the production renderer, it now assumes the copied Nightreign Map Router assets are already present inside this repository. The `--repo` flag is optional.

### 4. `recognize_nightreign_map_proto.py`
Prototype recognizer.

Use this to test experimental recognition approaches without touching the production GUI.

The latest prototype line is slot-based and supports:
- live monitor capture or input image files
- optional debug overlays
- slot labeling
- layout ranking inside the predicted Shifting Earth

---

## Recognition architecture

### Production GUI recognizer
The current GUI recognizer is intended to use a **slot-based recognition package** rather than the older full-image similarity cache.

High-level flow:
1. capture the selected display
2. detect and crop the map panel
3. classify the Shifting Earth from terrain
4. classify local landmark slots
5. compare the observed slot pattern against the packaged slot metadata
6. rank candidate layouts
7. display the best detailed map image and the top alternatives

The GUI keeps the polished interaction layer from the older recognizer line:
- screen selection
- preview of the cropped screenshot first
- final detailed reference image afterward
- zooming and dragging
- hotkey support
- settings popover

### Prototype recognizer
The prototype recognizer exists to test new recognition ideas faster and with better debug visibility.

The latest prototype line uses:
- Shifting Earth classification from the cropped panel
- a finite set of landmark slots
- local slot classification for:
  - Church
  - Fort
  - Great Church
  - Main Encampment
  - Ruins
  - Sorcerer's Rise
- comparison against per-layout slot occupancy metadata
- support for real empty slots, not just missed detections

---

## Landmark set that determines layout variations

For a fixed Shifting Earth, the important discriminative landmark families are:

- Church
- Fort
- Great Church
- Main Encampment
- Ruins
- Sorcerer's Rise

Other icons can still be rendered to make the final map complete, but they are not required for identifying the layout variation in the prototype recognition model.

---

## External data and assets expected by the project

### A. Files copied from Nightreign Map Router
The renderer scripts expect the following Nightreign Map Router content to exist **inside this repository**, preserving the same relative structure as the original repository:

- `reference_material/pattern_layouts`
- `public/assets/maps`
- `public/POI_icons`
- `src/app/_components/MapCanvas.tsx`
- `src/utils/poiIconMapping.ts`
- `src/utils/poiLocationMapping.ts`

These are used to:
- load layout JSON files
- load map tiles
- load canonical POI coordinates
- load icon/type mappings
- mirror key router rendering rules

### B. Display image set
The GUI recognizer displays finished map images from:

- `all_images/with_text`

Current practical convention:
- use `.webp` display images for packaged builds

### C. Recognition package
The slots-based recognizer expects:

- `nightreign_geom_proto_v4_slots.npz`

This package is generated by `render_nightreign_router_map_proto.py`.

### D. Icon files for the packaged Windows app
Current practical icon files:

- `nightreign_sword_large_transparent_256.ico`
- `nightreign_sword_small_transparent_48.ico`

Recommended use:
- large icon for the executable resource
- small icon for the running window/taskbar icon, with fallback to the large icon

---

## Recommended repository structure

```text
Nightreign-Recognizer/
  README_repository.md
  README_end_user_exe.md

  render_nightreign_router_map.py
  nightreign_map_recognizer_gui.py
  render_nightreign_router_map_proto.py
  recognize_nightreign_map_proto.py

  nightreign_geom_proto_v4_slots.npz

  nightreign_sword_large_transparent_256.ico
  nightreign_sword_small_transparent_48.ico

  all_images/
    with_text/
      *.webp

  reference_material/
    pattern_layouts/
      layout_*.json

  public/
    assets/
      maps/
        ...
    POI_icons/
      ...

  src/
    app/
      _components/
        MapCanvas.tsx
    utils/
      poiIconMapping.ts
      poiLocationMapping.ts
```

---

## Typical workflows

### 1. Render full reference images
Use the production renderer directly from this repository root.

Example:

```bash
python render_nightreign_router_map.py --text on --name-mode numbered --ext webp
```

If you want to point at a different copied router root, you can still do:

```bash
python render_nightreign_router_map.py --repo "C:/path/to/alternate/root" --text on --name-mode numbered --ext webp
```

### 2. Generate the slot recognition package
Use the prototype renderer/package generator:

```bash
python render_nightreign_router_map_proto.py --text off --name-mode numbered --ext webp --export-recognition-package "nightreign_geom_proto_v4_slots.npz" --export-recognition-dir "recognition_metadata_v4_slots"
```

### 3. Run the production GUI recognizer
```bash
python nightreign_map_recognizer_gui.py
```

### 4. Run the prototype recognizer with debug output
```bash
python recognize_nightreign_map_proto.py --package "nightreign_geom_proto_v4_slots.npz" --monitor 1 --debug
```

---

## How the renderer now locates its data

Both renderer scripts now resolve their repository root like this:
- use `--repo` if it was provided explicitly
- otherwise use the folder containing the script itself

That means the copied Nightreign Map Router folders can live directly inside this repository, and the most common commands no longer need a separate `--repo` argument.

---

## Packaging

A typical PyInstaller command for the production GUI recognizer is:

```bat
python -m PyInstaller --noconfirm --clean --windowed --name NightreignRecognizerSlots --icon "nightreign_sword_large_transparent_256.ico" --add-data "nightreign_geom_proto_v4_slots.npz:." --add-data "all_images/with_text/*.webp:all_images/with_text" --add-data "nightreign_sword_large_transparent_256.ico:." --add-data "nightreign_sword_small_transparent_48.ico:." nightreign_map_recognizer_gui.py
```

Notes:
- the executable icon uses the large `.ico`
- the running app window/taskbar icon is set in code, preferring the small `.ico`
- Windows icon caching can make Explorer keep showing an old icon after rebuilding; rebuilding the icon cache may be necessary

---

## Notes for future development

### Renderer
The production renderer is stable enough for everyday use and can:
- generate finished map images
- export `png`, `jpg`, or `webp`
- produce numbered batches for recognition assets

The prototype renderer should remain the place for:
- metadata export
- recognition-package generation
- future slot-catalog improvements

A future cleanup step could merge package-generation options into the production renderer so only one renderer script is needed.

### GUI recognizer
The GUI recognizer should remain the user-facing application.

Useful future improvements:
- better in-app debug mode for slot observations
- side-by-side per-slot diff for top candidate layouts
- improved packaging polish
- settings export/import
- clearer confidence and failure reporting

### Prototype recognizer
The prototype recognizer should continue to be the testbed for:
- better local slot classification
- improved handling of `empty` versus `unknown`
- alternative slot scoring models
- debug overlays and reports

---

## Commercialization considerations

If the GUI recognizer later becomes a commercial product, review carefully:
- whether bundled map assets may be redistributed
- whether recognition packages derived from game data can be bundled directly
- licensing and rights related to any shipped images, icons, or extracted metadata
- whether the product should require the user to provide their own game-derived assets or packages

---

## Summary

This repository should be treated as having four stable entry points:

- `render_nightreign_router_map.py`
- `nightreign_map_recognizer_gui.py`
- `render_nightreign_router_map_proto.py`
- `recognize_nightreign_map_proto.py`

The production path is:
- render complete reference maps
- package the slot recognition data
- run the GUI recognizer

The prototype path is:
- experiment with new recognition strategies
- inspect debug overlays and reports
- evolve the recognition backend before promoting changes into the GUI.
