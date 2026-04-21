# Nightreign Recognizer v38 Deep Technical README

This document is a code-driven reference for the current `v38` stack in this repository.

Scope:
- `nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py`
- `render_bosses_on_backgrounds.py`
- CSV synthesis utilities:
  - `generate_layout_json_from_game_csvs.py`
  - `build_mappatternset_bundle_json.py`
- headless/debug runners:
  - `run_v38_headless.py`
  - `run_v38_on_image.py`
- build config:
  - `nightreign_v38.spec`
  - `pyinstaller_command.txt`

This README is intentionally detailed and math-first.

---

## 1. End-to-End Architecture

The v38 recognizer pipeline is:

1. Capture screen image (or use pre-captured image).
2. Detect map circle and derive map panel crop.
3. Normalize panel and run earth/slot inference.
4. Score candidate layouts with probabilistic + prior terms.
5. Rank outputs and render top-5 display maps.
6. Optionally generate text-only overlay variants.
7. Show main image + top-5 thumbnails + optional overlay.

Important runtime behavior:
- Dynamic render outputs are cached in `tmp_dynamic_render_cache_v38` and then reused by file path.
- Main/top-5 display are path-based after first render.
- Overlay text-only is feature-gated by `ENABLE_OVERLAY_TEXT_ONLY_FEATURE`.

---

## 2. Math and Scoring Details

## 2.1 Coordinate Transforms

World-space to image-space (renderer):

Given:
- `WORLD_MIN_X=-256`, `WORLD_MAX_X=1280`
- `WORLD_MIN_Z=-256`, `WORLD_MAX_Z=1280`
- `IMAGE_W=IMAGE_H=1536`

Transform:

- `px = ((x - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X)) * IMAGE_W`
- `py = IMAGE_H - ((z - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z)) * IMAGE_H`

`py` is inverted because world `Z` grows upward while image `Y` grows downward.

Normalized panel coordinate helper (recognizer):
- `x_norm = (x_world - WORLD_MIN_X) / (WORLD_MAX_X - WORLD_MIN_X)`
- `y_norm = 1 - ((z_world - WORLD_MIN_Z) / (WORLD_MAX_Z - WORLD_MIN_Z))`

---

## 2.2 Circle and Panel Localization

The recognizer tries two Hough stages:

1. Blue-ring detection (preferred stable anchor).
2. Grayscale Hough fallback with stabilization toward a known fallback circle.

Candidate scoring uses distance from fallback center and radius bonus:
- `score = |cx-fx|/W + |cy-fy|/H - k*(r/min(H,W))`

After selecting a circle, panel box is estimated from:
- `side = 2 * r * panel_scale`

Panel edge refinement:
- Sobel `gradx`, `grady` over grayscale image.
- Search around estimated edges and maximize mixed brightness/gradient score.
- Final crop enforced square.

---

## 2.3 Legacy Similarity Branch (Image Correlation Math)

Preprocessing:
- grayscale resize to `CMP_SIZE=512`
- Gaussian blur
- gradient magnitude branch via Sobel + blur
- pHash via DCT thresholding

Similarity components:
- `low_corr`: normalized cross-correlation on low-frequency grayscale
- `grad_corr`: normalized cross-correlation on gradient magnitude
- `hsim`: hash similarity = `1 - hamming_distance/hash_len`
- `low_mae`, `grad_mae`: mean absolute errors after normalization

Final score:

`score = 0.42*low_corr + 0.26*grad_corr + 0.22*hsim + 0.06*(1/(1+low_mae)) + 0.04*(1/(1+grad_mae))`

---

## 2.4 Layout Probability Model (Main v38 AI Path)

For a layout, slot evidence uses weighted log-likelihood:

- For slot `i`, expected class probability `p_i` and weight `w_i`
- Accumulate `sum_i w_i * log(p_i + eps)`
- Normalize by total weight:
  - `normalized_log_prob = total_log / max(1, total_weight)`

Layout final score:

`layout_score = normalized_log_prob`
`+ 0.75 * earth_score`
`+ geometry_prior`
`+ lost_slot_prior`
`+ hard_constraint_prior`
`+ evergaol_prior`
`+ great_hollow_status_prior`
`+ great_hollow_model_layout_prior`
`+ layout_infer_prior`

This is implemented in `score_layout_ai(...)`.

---

## 2.5 Priors

### 2.5.1 Geometry Prior

Computed from earth-level alignment response and slot-count regime.

Key behavior:
- Compact-slot earths (`<=10`) are penalized if far below best standard-earth alignment.
- Standard-slot earths (`>=20`) can gain mild bonus if above best compact alignment.

### 2.5.2 Lost-Limveld Slot Prior

Derives Limveld slots absent in a candidate earth and checks Limveld observations:
- Observed `empty` on expected-lost slots => positive prior.
- Strong non-empty evidence where slot should be absent => negative prior.

### 2.5.3 Hard Constraint Prior

Great Hollow-focused penalty if forbidden Limveld slots appear occupied.

### 2.5.4 Evergaol Prior

At predefined world anchors, compute present probability `p`.

Slot prior:
- If expected present: `0.09 * (p - 0.35)`
- If expected absent: `0.10 * (0.35 - p)`

Earth prior is sum over all anchors.

### 2.5.5 Great Hollow Status Prior

For each GH anchor:
- Convert best score/margin to confidence:
  - `score_strength = clamp((best_score-0.42)/0.28, 0, 1)`
  - `margin_strength = clamp(margin/0.12, 0, 1)`
  - `confidence = 0.55*score_strength + 0.45*margin_strength`

Prior per anchor:
- Match expected status: `+ (0.035 + 0.085*confidence)`
- Wrong-but-allowed status: `- (0.030 + 0.110*confidence)`
- Invalid status: `- (0.180 + 0.100*confidence)`

### 2.5.6 Great Hollow Layout-Model Prior

Uses model observations for GH statuses/slot labels.

Strength examples:
- Binary strength:
  - `p = clamp((best_prob-0.50)/0.50,0,1)`
  - `m = clamp(margin/0.35,0,1)`
  - `strength = 0.65*p + 0.35*m`
- Multiclass strength:
  - baseline `1/class_count`
  - combines top probability and top-vs-second margin

Magnitude:
- `magnitude = weight * (0.30 + 0.70*strength)`
- Match => `+magnitude`, mismatch => `-magnitude`

### 2.5.7 Earth Cascade Ranking

For Great Hollow:

`cascade = 1.35*base + 1.10*geom + 1.45*hard + 1.15*ever + 0.10*gh_status_support`

For non-GH:

`cascade = 1.25*base + (geom + 1.35*lost + 1.35*ever)`

Then GH gating/boost logic applies:
- raw dominance gate
- structural gate
- optional score boost if gate passes

---

## 2.6 Renderer Slot Geometry Math

In `render_bosses_on_backgrounds.py`, recognition package metadata is derived by:

1. Collecting landmark points per earth/layout.
2. Building slot clusters with radius threshold `SLOT_CLUSTER_RADIUS_NORM`.
3. Merging nearby clusters (`0.72 * SLOT_CLUSTER_RADIUS_NORM`).
4. Slot radius:
   - `radius_norm = clamp(max_dist*1.35 + 0.004, SLOT_MIN_RADIUS_NORM, SLOT_MAX_RADIUS_NORM)`
5. Assigning labels by nearest-slot matching with tolerance:
   - `tol = slot_radius * SLOT_ASSIGN_TOLERANCE_FACTOR`
6. Slot discriminativeness weighting via entropy:
   - `weight = 0.75 + 1.0 * normalized_entropy`

This package is saved to `nightreign_geom_proto_v4_slots.npz`.

---

## 3. CSV Table Correlation and Data Lineage

## 3.1 Primary Layout Synthesis (`generate_layout_json_from_game_csvs.py`)

Input CSVs:
- `LotResultPlayAreaParam.csv`
- `LotResultSmallBaseAndSpot.csv`
- `LotResultMapPatternFlag.csv`

Logical relations:
- `LotResultPlayAreaParam.ID` is treated as layout ID.
- `LotResultPlayAreaParam.patternId` joins to:
  - `LotResultSmallBaseAndSpot.patternId`
  - `LotResultMapPatternFlag.patternId`

Output:
- one `layout_XXX.json` per layout ID
- includes:
  - Shifting Earth
  - Night bosses/circles/spawn
  - major/minor/event/merchant slots
  - per-entry metadata (`attachId`, `smallBaseMapId`, `variationId`, etc.)

Category inference is currently primarily from `attachId` range + known IDs, with mapId-name mapping support.

## 3.2 PatternSet Modifier Correlation (`build_mappatternset_bundle_json.py`)

Input CSVs:
- `LotResultPlayAreaParam.csv`
- `LotResultMapPatternFlag.csv`
- `MapPatternSet.csv`

Relations:
- `patternId` groups flag rows.
- `patternSetId` from grouped flags should map to exactly one `MapPatternSet.ID`.
- Modifier bundle:
  - realized modifiers come from nonzero `LotResultMapPatternFlag.modifier`
  - active patternset fields are fields in `MapPatternSet` whose numeric value appears in realized modifiers.

Output per pattern:
- `pattern_XXXX_mappatternset_bundle.json`
- includes source rows + active fields + matched MapPatternSet row.

## 3.3 Renderer Input Correlation

Renderer consumes grouped per-pattern JSON:
- currently `LotResultSmallBaseAndSpot_grouped_by_patternId_14.json`

Each pattern entry supplies:
- `patternId`
- `rareMap` (earth/map family)
- `rows` (POI instances with world coordinates and location/variation fields)

These rows are transformed into rendered landmarks and recognition metadata, which then feed GUI scoring.

---

## 4. Flag Catalog

This section separates:
- in-code flags (source constants/switches),
- environment flags (runtime),
- command-line flags (script entrypoints),
- build flags (PyInstaller).

## 4.1 In-Code Flags (Core)

Definition used here: in-code flags are source-level constants/toggles that change runtime behavior without requiring a code-path rewrite.

### `nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py`

- `ENABLE_PATTERN_ID_WHITELIST_FILTER = True`
  - Restricts layout scoring to `ALLOWED_PATTERN_IDS`.
- `ENABLE_OVERLAY_TEXT_ONLY_FEATURE = True`
  - Master gate for text-only overlay feature and checkbox visibility.
- `USE_EARTH_SELECTOR_V2_BY_DEFAULT`
  - Derived from env defaulting logic; controls whether Earth Selector v2 is used by default.
- `USE_EARTH_CASCADE_SELECTION`
  - Derived from env defaulting logic; controls whether cascade-based earth selection path is active.
- `USE_LEGACY_EARTH_SCORER_BY_DEFAULT`
  - Derived from env defaulting logic; when true, learned earth classifier path is bypassed.
- `DEFAULT_OVERLAY_TEXT_ONLY_ENABLED = True`
  - Default persisted toggle state when setting key missing.
- `DEFAULT_COMPLETION_TOAST_ENABLED = True`
  - Default completion toast behavior.
- `GH_USE_PARALLEL_PANEL_CLASSIFIER_DEFAULT = True`
  - Default parallel classifier toggle when env override is absent.
- `GREAT_HOLLOW_STATUS_CLASSIFIER_MODE_DEFAULT = "models"` (`classic|models`)
- `GREAT_HOLLOW_LAYOUT_MODEL_MATCH_MODE_DEFAULT = "models"` (`classic|models`)
- `GH_SLOT4_CLASSIFIER_MODE_DEFAULT = "prototype"` (`classic|prototype`)
- `GH_SLOT4_EMPTY_GATE_THRESHOLD_DEFAULT = 0.55`

### `render_bosses_on_backgrounds.py`

- `PATTERN_IDS_TO_RENDER = None`
  - `None` = render all.
- `SKIP_TEXT_WHEN_ICON_MISSING = False`
  - If true, suppresses text labels when icon assets are missing.
- `RENDER_ANY_TEXT = True`
  - Master in-code text rendering toggle in normal render path.
- `HIDE_ALL_BOSSES_AND_ALL_TEXT` (env-backed)
- `RENDERER_DEBUG` (env-backed)
- `UNDER_VARIANT_RENDER_ENABLED` (env-backed)
- `RENDER_ONLY_UNDER_VARIANTS` (env-backed)
- `UNDER_ONLY_TARGET_GREAT_CHURCH` (env-backed)

---

## 4.2 Environment Flags (Runtime)

All env vars below are read by current code and influence runtime behavior.

### GUI / recognition (`nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py`)

- `NR_DISABLE_DYNAMIC_RENDER`
  - If set (any value), dynamic render is disabled and fallback files are used.
  - If unset, dynamic render is enabled.
- `NR_DISABLE_RENDER_FALLBACK`
  - If set, render fallback lookup is disabled.
  - If unset, fallback image lookup is allowed.
- `NR_USE_EARTH_SELECTOR_V2`
  - Default behavior uses v2 when value is not `"0"`.
- `NR_ENABLE_EARTH_CASCADE`
  - Enable earth cascade ranking when `"1"`.
- `NR_USE_EARTH_CASCADE_SELECTION`
  - Alternative env to enable cascade when `"1"`.
- `NR_USE_LEARNED_EARTH_CLASSIFIER`
  - If `"0"`, forces legacy earth scorer path.
- `NR_SHOW_DUAL_SHOT_CHECKBOX`
  - If present, dual-shot checkbox is shown in UI.
- `GH_SLOT4_CLASSIFIER_MODE`
  - Accepted: `classic`, `prototype`.
- `GH_SLOT4_EMPTY_GATE_THRESHOLD`
  - Float clamped to `[0,1]`.
- `GH_STATUS_CLASSIFIER_MODE`
  - Accepted: `classic`, `models`.
- `GH_LAYOUT_MODEL_MATCH_MODE`
  - Accepted: `classic`, `models`.
- `GH_USE_PARALLEL_PANEL_CLASSIFIER`
  - Truthy set (`1,true,yes,on,y`) enables parallel panel classifier.
- `NIGHTREIGN_DUALSHOT_DEBUG`
  - Truthy enables dual-shot debug dump.
- `NIGHTREIGN_DEBUG`
  - Fallback truthy debug toggle used by dual-shot debug.

Internal temporary env keys used during GH status path (set/restored inside function):
- `ICON_RECOGNIZER_LEGACY_CROP_LEFT`
- `ICON_RECOGNIZER_LEGACY_CROP_RIGHT`
- `ICON_RECOGNIZER_LEGACY_CROP_TOP`
- `ICON_RECOGNIZER_LEGACY_CROP_BOTTOM`

### Renderer (`render_bosses_on_backgrounds.py`)

- `NR_HIDE_ALL_BOSSES_AND_ALL_TEXT`
  - `"1"` enables no-text/no-boss mode.
- `NR_RENDERER_DEBUG`
  - `"1"` enables debug-print hot-path logs.
- `NR_RENDER_GH_UNDER_VARIANTS`
  - `"1"` enables under-variant GH rendering.
- `NR_RENDER_ONLY_GH_UNDER_VARIANTS`
  - `"1"` renders only under-variants.
- `NR_UNDER_ONLY_TARGET_GREAT_CHURCH`
  - `"1"` constrains under logic to target Great Church slot path.

---

## 4.3 Command-Line Flags (Scripts)

Coverage note: this section includes all argparse-based entrypoints in the scoped files plus the current PyInstaller command wrapper file.

## `run_v38_headless.py`

- `input` (positional): image path
- `--monitor-index` (int, default `1`)

## `generate_layout_json_from_game_csvs.py`

- `--input-dir` (default `.`)
- `--output-dir` (default `./pattern_layouts_generated`)
- `--patterns` (IDs and/or ranges like `1005-1009`)
- `--known-mapid-csv` (optional mapId->name mapping CSV)

## `build_mappatternset_bundle_json.py`

- `--input-dir` (required)
- `--output-dir` (required)
- `--pattern-id` (optional single pattern)

## `run_v38_on_image.py`

- No CLI flags in current file; image path is hardcoded in source.

---

## 4.4 Build Flags (PyInstaller)

Current build entry:
- `pyinstaller nightreign_v38.spec`

Key options in `nightreign_v38.spec`:
- `console=False` (windowed app; no console window)
- `debug=False`
- `upx=True`
- `runtime_hooks=['hook_torch_dlls.py']`
- extensive `datas=[...]` for models/maps/icons/testing assets

---

## 5. Runtime State and Persisted Settings

Settings file:
- `nightreign_settings.json`

Stored keys used by GUI:
- `global_hotkey`
- `overlay_hotkey`
- `overlay_opacity`
- `completion_toast_enabled`
- `overlay_text_only_enabled`
- `dual_shot_enabled`
- `log_panel_open`

Dynamic render cache directory:
- `tmp_dynamic_render_cache_v38/`
  - `main/layout_XXXX.png`
  - `overlay_text/layout_XXXX.png`

---

## 6. Known Heavy Paths (Performance Reality)

After a full run, heaviness often comes from:
- resident Torch models (CPU RAM and possibly GPU VRAM),
- retained backend payloads and top result structures,
- image caches and Tk image objects,
- periodic queue/hotkey polling loops (small but continuous).

Heavy click regressions were historically tied to repeated full thumbnail strip rebuilds and expensive per-pixel RGBA conversions. Current code now:
- avoids full 5-thumb rebuild on simple thumbnail selection,
- uses disk-backed dynamic render paths,
- keeps feature-gated fast paths when text-only overlay feature is disabled.

---

## 7. Practical Flag Recipes

## Disable dynamic rendering (force pre-rendered/fallback images)

PowerShell:

```powershell
$env:NR_DISABLE_DYNAMIC_RENDER='1'
python .\nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py
```

## Disable fallback (render-only mode)

```powershell
$env:NR_DISABLE_RENDER_FALLBACK='1'
python .\nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py
```

## Enable earth cascade

```powershell
$env:NR_ENABLE_EARTH_CASCADE='1'
python .\nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py
```

## Renderer debug prints on

```powershell
$env:NR_RENDERER_DEBUG='1'
python .\render_bosses_on_backgrounds.py
```

## GH model-path toggles

```powershell
$env:GH_STATUS_CLASSIFIER_MODE='models'   # or classic
$env:GH_LAYOUT_MODEL_MATCH_MODE='models'  # or classic
$env:GH_USE_PARALLEL_PANEL_CLASSIFIER='1' # or 0
python .\nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py
```

---

## 8. Maintenance Notes

- The v38 file includes legacy and overridden sections from iterative development. Prefer editing the active bottom-layer patched methods/functions used by current monkey-patching.
- If changing scoring math, adjust both:
  - score computation
  - logging/output explanations
- If changing CSV synthesis logic, update both:
  - `generate_layout_json_from_game_csvs.py`
  - renderer assumptions about grouped fields and categories.
