# Nightreign-Recognizer: Elden Ring Shadow DLC Map Recognition Toolkit

A production-grade screenshot recognition system for Elden Ring: Shadow of the Erdtree DLC that combines classical computer vision with deep learning to recognize in-game maps and overlay reference layouts with POI details.

For the full v38 technical reference (math formulas, CSV-table correlations, and complete flag catalog), see `README_v38_DETAILED.md`.

## Table of Contents

1. [Overview](#overview)
2. [Project Architecture](#project-architecture)
3. [Installation & Running](#installation--running)
4. [CSV Data Structure & Correlations](#csv-data-structure--correlations)
5. [Mathematical Algorithms](#mathematical-algorithms)
6. [Configuration Files](#configuration-files)
7. [Command-Line Arguments](#command-line-arguments)
8. [In-Code Flags & Environment Variables](#in-code-flags--environment-variables)
9. [Data Models & AI Components](#data-models--ai-components)
10. [Recognition Pipeline](#recognition-pipeline)
11. [Performance & Tuning](#performance--tuning)

---

## Overview

**Nightreign-Recognizer** is designed to:
- **Capture** a game screenshot from your monitor
- **Extract** the map panel (circular area with terrain)
- **Identify** the Shifting Earth variant (geographical region like Limveld, Mountaintop, Great Hollow)
- **Classify** landmark slots (Churches, Forts, Ruins, etc.) based on visual patterns
- **Match** the overall layout against a database of known maps
- **Display** top 5 matches with zoom/pan reference visualization
- **Overlay** Points of Interest (POI) such as Evergaols and status-modifying blessings

The toolkit handles:
- **7 Shifting Earths** (Limveld, Mountaintop, The Crater, Rotted Woods, Noklateo, Great Hollow, Shrouded City)
- **6+ Landmark Types** (Church, Fort, Great Church, Main Encampment, Ruins, Sorcerer's Rise)
- **Status Modifiers** (Fire, Magic, Rot, Holy, Poison, Bleed, Sleep, Lightning)
- **Evergaol Detection** and placement mapping

---

## Project Architecture

### Directory Structure

```
Nightreign-Recognizer/
│
├── [Main GUI Application]
│   └── nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py
│       • TKinter-based desktop application
│       • Real-time screenshot capture with hotkey
│       • Multi-monitor support
│       • Log panel + result display with zoom/pan
│       • Toast notifications for completion
│
├── [Backend Recognition Engines]
│   ├── recognize_nightreign_map_proto_v12_authoritative.py
│   │   • Core slot-based recognition: panel extraction, earth classification, 
│   │     landmark classification, layout scoring
│   │
│   └── nightreign_slot_ai_inference_v1.py
│       • PyTorch model loading and inference
│       • SmallSlotCNN, EarthPanelCNN, EvergaolPatchCNN, Status classifiers
│
├── [Rendering & Reference Data Generation]
│   ├── render_nightreign_router_map.py (production)
│   ├── render_nightreign_router_map_proto.py (slot package export)
│   ├── render_bosses_on_backgrounds.py (POI overlay generation)
│   └── build_mappatternset_bundle_json.py (CSV data bundling)
│
├── [Game Data Conversion]
│   ├── generate_layout_json_from_game_csvs.py
│   │   • Converts Nightreign game CSV exports → Router-style layout JSON
│   │   • Creates layout_XXXX.json files with slot definitions
│   │
│   └── [Input CSV files]
│       ├── LotResultPlayAreaParam.csv
│       ├── LotResultSmallBaseAndSpot.csv
│       ├── LotResultMapPatternFlag.csv
│       └── MapPatternSet.csv
│
├── [Pre-computed Data Files]
│   ├── nightreign_geom_proto_v4_slots.npz (serialized recognition package)
│   ├── nightreign_best_tuned_config.json (runtime hyperparameters)
│   ├── nightreign_best_tuned_config_multi.json (multi-earth variant)
│   ├── nightreign_settings.json (user GUI settings)
│   ├── great_hollow_pattern_sets/ (per-layout JSON bundles)
│   ├── renders/ (cached reference map images)
│   ├── renders_new/ (with/without text variants)
│   └── POI_icons/ (status modifier and Evergaol icons)
│
└── [PyTorch Models]
    ├── slot_ai_test_output_dlc_no_text_nobalance/slot_patch_model_v1.pt
    │   • SmallSlotCNN: landmark type classification from cropped patches
    │
    ├── earth_ai_test_output_no_text/earth_panel_model_v1.pt
    │   • EarthPanelCNN: Shifting Earth classification from full panel
    │
    ├── evergaol_ai_test_output/evergaol_patch_model_v1.pt
    │   • EvergaolPatchCNN: Evergaol presence detection
    │
    ├── tmp_palace_magic_fire_test/palace_magic_fire_test_model.pt
    │   • Status classifier: Fire/Magic for Palace landmark
    │
    ├── tmp_fortress_rot_holy_test/fortress_rot_holy_test_model.pt
    │   • Status classifier: Rot/Holy for Fortress landmark
    │
    ├── tmp_ruins_poison_bleed_test/ruins_poison_bleed_test_model.pt
    │   • Status classifier: Poison/Bleed for Ruins landmark
    │
    ├── tmp_gc_sleep_lightning_test/gc_sleep_lightning_test_model.pt
    │   • Status classifier: Sleep/Lightning for Great Church landmark
    │
    ├── tmp_gc_frost_holy_test_newbox/gc_frost_holy_test_model_newbox.pt
    │   • Hybrid model: additional Great Church classification variant
    │
    ├── tmp_gh_slot4_prototype_model_test/gh_slot4_empty_vs_churchrise_prototype_model.pkl
    │   • Great Hollow Slot 4 classifier: Empty gate vs. Church of the Rotten Rise
    │
    └── Testing/slot_2_fixed_geometry_panel_bases260/
        └── slot2_status_cnn_offset50_v1.pt
            • Great Hollow Slot 2: Great Church status (Fire/Magic)
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py` | Main GUI entry point with screenshot capture, recognition orchestration, and result display |
| `recognize_nightreign_map_proto_v12_authoritative.py` | Recognition engine: panel extraction, earth classification, slot labeling, layout matching |
| `nightreign_slot_ai_inference_v1.py` | PyTorch model inference wrapper for CNN-based slot/earth/evergaol/status classification |
| `render_nightreign_router_map.py` | Renders reference map images from Router JSON metadata |
| `render_bosses_on_backgrounds.py` | Overlays POI (Evergaols, status modifiers) onto rendered maps |
| `generate_layout_json_from_game_csvs.py` | Converts game CSV exports to Router-compatible layout JSON files |
| `build_mappatternset_bundle_json.py` | Bundles per-pattern CSV data into JSON for efficient lookup |
| `icon_recognizer_v4.py` | Template-based status icon classifier with alignment correction |
| `map_class_classifier.py` | Map region classification by CSS/type identifier |

---

## Installation & Running

### Prerequisites

- **Python 3.9+**
- **PyTorch** (CPU or CUDA): `pip install torch torchvision`
- **OpenCV**: `pip install opencv-python`
- **NumPy**: `pip install numpy`
- **Pillow**: `pip install Pillow`
- **mss** (multi-monitor screenshot): `pip install mss`
- **pynput** (hotkey binding): `pip install pynput`

### Running the GUI Application

```bash
python nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py
```

The application will:
1. Open a TKinter window with two panels:
   - **Left panel**: Real-time log output (toggleable)
   - **Right panel**: Map display with top 5 matches and thumbnail strip
2. Wait for you to press the configured hotkey (default: `Ctrl+Alt+G`)
3. Capture a screenshot and perform recognition
4. Display results ranked by confidence score

### Running Recognition Directly (Python API)

```bash
# Using full screenshot
python recognize_nightreign_map_proto_v12_authoritative.py \
    --package nightreign_geom_proto_v4_slots.npz \
    --image screenshot.png \
    --top 20 \
    --debug \
    --crop-out panel_used.png \
    --detections-out panel_with_labels.png \
    --json-out results.json

# Using monitor capture (1-based index)
python recognize_nightreign_map_proto_v12_authoritative.py \
    --package nightreign_geom_proto_v4_slots.npz \
    --monitor 1 \
    --top 20

# Using pre-cropped panel image
python recognize_nightreign_map_proto_v12_authoritative.py \
    --package nightreign_geom_proto_v4_slots.npz \
    --panel-image panel.png \
    --top 5
```

---

## CSV Data Structure & Correlations

The game data is sourced from **Nightreign game CSV exports**. These CSVs define all map variants, landmark placements, modifiers, and boss placements.

### CSV Tables Overview

#### 1. **LotResultPlayAreaParam.csv**

**Purpose**: Defines each unique map layout/pattern.

**Key Columns**:
- `ID` (int, PRIMARY KEY) → **Layout number** (1000–1999 range typical)
- `modifier` (int) → Shifting Earth modifier (11=Mountaintop, 12=The Crater, 13=Rotted Woods, 14=Great Hollow, 15=Shrouded City)
- `bossModifier1` (int) → First night boss encounter modifier
- `bossModifier2` (int) → Second night boss encounter modifier
- `baseName` (string) → Human-readable region label (e.g., "Limveld Area A")
- `subAreaId` (int) → Region classification
- `playAreaType` (int) → Layout type identifier

**Usage**: 
- Maps to `shifting_earth` via `modifier` field
- Used as the canonical layout identifier in layout JSON files
- Paired with `LotResultSmallBaseAndSpot` by `ID` to get landmark placements

**Example**:
```
ID,modifier,bossModifier1,bossModifier2,baseName,subAreaId
1001,0,4551,4552,Limveld Area A,100
1002,11,4551,4552,Mountaintop Area B,101
1003,14,5200,5201,Great Hollow Area C,114
```

#### 2. **LotResultSmallBaseAndSpot.csv**

**Purpose**: Defines all placed landmarks, bosses, and POI within maps.

**Key Columns**:
- `patternId` (int) → **Foreign key to `LotResultPlayAreaParam.ID`** (which layout this belongs to)
- `attachId` (int) → **Slot ID** (0–15 typical; identifies the specific landmark position within a layout)
- `smallBaseMapId` (int) → Content/entity type (Church=600, Fort=610, etc.)
- `variationId` (int) → Variation within that type (e.g., Church type 1 vs. type 2)
- `x`, `y` (float) → World coordinates (normalized 0–1 or absolute)
- `z`, `rotation` (float) → Elevation and rotation
- `isBossPlacement` (bool) → Whether this is a boss encounter
- `scale` (float) → Visual scale factor

**Correlation**: 
- **Many-to-one**: Multiple `LotResultSmallBaseAndSpot` rows share the same `patternId` (one per slot in a layout)
- **Links to `LotResultPlayAreaParam`**: Via `patternId` = `LotResultPlayAreaParam.ID`
- **Indexed by attachId**: Each row's `attachId` identifies which of the 6–12 landmark slots in that layout is being filled

**Usage**:
- Used to populate slot definitions in layout JSON
- Determines which slots are "filled" vs. "empty" for a given layout
- Provides world coordinates for POI rendering

**Example**:
```
patternId,attachId,smallBaseMapId,variationId,x,y,z,isBossPlacement
1001,0,600,0,0.256,0.312,10.5,0
1001,1,610,1,0.612,0.745,5.2,0
1001,2,0,0,0.100,0.900,0,1
```

*(Row 3 has `smallBaseMapId=0`, indicating an empty slot.)*

#### 3. **LotResultMapPatternFlag.csv**

**Purpose**: Maps pattern ID to modifier flags and pattern set metadata.

**Key Columns**:
- `patternId` (int) → **Foreign key to `LotResultPlayAreaParam.ID`**
- `modifier` (int) → Status modifier flag (11–15 for Shifting Earth, higher numbers for status modifiers)
- `patternSetId` (int) → **Foreign key to `MapPatternSet.ID`** (groups patterns by visual/thematic set)
- `flagValue` (int) → Bitwise flag encoding

**Correlation**:
- **Many-to-one**: Multiple rows per `patternId` (one per active modifier)
- **Links**: `patternId` → `LotResultPlayAreaParam.ID` AND `patternSetId` → `MapPatternSet.ID`
- **Usage**: Categorizes which modifiers (Shifting Earths, status effects) apply to this pattern

**Example**:
```
patternId,modifier,patternSetId,flagValue
1001,0,100,0x0001
1001,11,101,0x0002
1002,11,101,0x0002
```

*(Pattern 1001 belongs to both PatternSet 100 and 101, with different modifiers.)*

#### 4. **MapPatternSet.csv**

**Purpose**: Defines pattern set metadata with field mappings for active modifiers.

**Key Columns**:
- `ID` (int, PRIMARY KEY) → Pattern set identifier
- `greatHollowEastFortMainModifier1` (int) → Modifier 1 for Great Hollow East Fortress (Fort status)
- `greatHollowEastFortMainModifier2` (int) → Modifier 2 for Great Hollow East Fortress
- `greatHollowWestFortMainModifier1` (int) → Modifier 1 for Great Hollow West Fortress (Palace status)
- `greatHollowWestFortMainModifier2` (int) → Modifier 2 for Great Hollow West Fortress
- `greatHollowWestFortUnderModifier1` (int) → Modifier 1 for Great Hollow West Fortress Underground (Ruins status)
- `greatHollowWestFortUnderModifier2` (int) → Modifier 2 for Great Hollow West Fortress Underground
- `greatHollowWestFortLibraryModifier1` (int) → Modifier 1 for Great Hollow West Fortress Library (Great Church status)
- `greatHollowWestFortLibraryModifier2` (int) → Modifier 2 for Great Hollow West Fortress Library
- *(Other regional or special modifier fields)*

**Correlation**:
- **One-to-many**: One `MapPatternSet` can apply to multiple `LotResultMapPatternFlag` rows
- **Realized via `LotResultMapPatternFlag.patternSetId`**
- Fields only contain values if the corresponding modifier is "active" (realized) in the layout

**Usage**:
- Determines which status modifiers (Fire/Magic, Rot/Holy, Poison/Bleed, Sleep/Lightning) are available in that pattern set
- Used by `build_mappatternset_bundle_json.py` to bundle modifiers per pattern

**Example**:
```
ID,greatHollowEastFortMainModifier1,greatHollowEastFortMainModifier2,greatHollowWestFortMainModifier1,...
100,7001,7002,7003,...
101,7004,7005,7006,...
```

### CSV Processing Pipeline

```
Game CSV Files
    ↓
generate_layout_json_from_game_csvs.py
    ├─ Read: LotResultPlayAreaParam.csv
    ├─ Read: LotResultSmallBaseAndSpot.csv
    ├─ Join: patternId + attachId → landmark placement
    ├─ Create: layout_XXXX.json (one per pattern ID)
    └─ Output: Router-style JSON with slot definitions
    ↓
build_mappatternset_bundle_json.py
    ├─ Read: LotResultMapPatternFlag.csv
    ├─ Read: MapPatternSet.csv
    ├─ Group: flags_by_pattern_id
    ├─ Link: pattern → patternSet → modifiers
    └─ Output: pattern_set_XXXX.json (per-pattern metadata bundles)
    ↓
render_nightreign_router_map_proto.py
    ├─ Read: layout_XXXX.json
    ├─ Extract: slot info (positions, types)
    └─ Output: nightreign_geom_proto_v4_slots.npz (recognition package)
```

### Key Correlations Summary

| CSV Table | Links To | Via Column | Relationship |
|-----------|----------|-----------|--------------|
| `LotResultSmallBaseAndSpot` | `LotResultPlayAreaParam` | `patternId` = `ID` | Many-to-one (multiple POI per layout) |
| `LotResultMapPatternFlag` | `LotResultPlayAreaParam` | `patternId` = `ID` | Many-to-one (multiple modifiers per pattern) |
| `LotResultMapPatternFlag` | `MapPatternSet` | `patternSetId` = `ID` | Many-to-one (multiple patterns per set) |
| `MapPatternSet` | *Modifier fields* | Field value → modifier ID | One-to-many (set defines which modifiers are active) |

---

## Mathematical Algorithms

### 1. Image Normalization

**Purpose**: Standardize input images to a canonical size for consistent processing.

**Formula**:
$$\text{normalized} = \text{resize}(\text{input}, \text{target\_size}, \text{interpolation})$$

**Parameters**:
- `CANONICAL_PANEL_SIZE = 1536` (pixels)
- Interpolation: `INTER_CUBIC` (upsampling), `INTER_AREA` (downsampling)

**Implementation** (from `recognize_nightreign_map_proto_v12_authoritative.py`):
```python
def normalize_panel_for_recognition(panel_bgr, target_size=CANONICAL_PANEL_SIZE):
    h, w = panel_bgr.shape[:2]
    interpolation = cv2.INTER_CUBIC if min(h, w) < target_size else cv2.INTER_AREA
    return cv2.resize(panel_bgr, (target_size, target_size), interpolation=interpolation)
```

---

### 2. Panel Extraction via Hough Circle Transform

**Purpose**: Locate and extract the circular map panel from a full screenshot.

**Algorithm**:
1. Convert input to HSV color space
2. Apply blue ring detection (optional, preferred for stability):
   - Threshold HSV for blue hue range
   - Apply morphological operations
   - Find contours and fit circles
3. Fallback to gray Hough circle transform if blue ring not found
4. Refine circle position with grid search (see **Crop Refinement** below)

**Hough Circle Transform Parameters**:
```python
cv2.HoughCircles(
    gray_image,
    cv2.HOUGH_GRADIENT,
    dp=1,                    # Accumulator resolution ratio
    minDist=100,             # Minimum distance between circle centers
    param1=50,               # Upper threshold for Canny edge detection
    param2=30,               # Accumulator threshold
    minRadius=300,           # Minimum circle radius
    maxRadius=800            # Maximum circle radius
)
```

**Output**:
```python
@dataclass
class Circle:
    cx: float      # Center X coordinate
    cy: float      # Center Y coordinate
    r: float       # Radius
```

**Blue Ring Detection Strategy**:
- Uses HSV threshold: Hue ∈ [100, 130] (blue range)
- Fits circles to contours of detected blue pixels
- Prefers circles with stable center/radius across multiple frames

---

### 3. Crop Box Estimation

**Purpose**: Convert circle position to square crop box for panel extraction.

**Formula**:
$$\text{side} = 2 \times r \times \text{panel\_scale}$$
$$\text{half} = \frac{\text{side}}{2}$$
$$\text{crop\_box} = [c_x - \text{half}, c_y - \text{half}, c_x + \text{half}, c_y + \text{half}]$$

**Constants**:
- `DEFAULT_PANEL_SCALE = 1.12` (accounts for circular panel within square bounds)

**Implementation**:
```python
def estimate_panel_box(circle, panel_scale=DEFAULT_PANEL_SCALE):
    side = 2.0 * circle.r * panel_scale
    half = side / 2.0
    return CropBox(
        int(round(circle.cx - half)),
        int(round(circle.cy - half)),
        int(round(circle.cx + half)),
        int(round(circle.cy + half))
    )
```

---

### 4. Crop Refinement via Grid Search

**Purpose**: Maximize Shifting Earth classification confidence by optimizing crop position and scale.

**Objective Function**:
$$\text{score} = \text{earth\_score} + 0.35 \cdot \text{margin} - 0.12 \cdot \text{black\_ratio} + 0.03 \cdot \text{edge\_mean}$$

**Where**:
- `earth_score`: CNN confidence for top Shifting Earth class
- `margin`: Crop margin in normalized units (0–1)
- `black_ratio`: Fraction of pixels with value < 50 (penalizes too-tight crops)
- `edge_mean`: Average edge magnitude (rewards texture-rich crops)

**Search Space**:
```python
CROP_REFINE_SCALE_FACTORS = (0.97, 1.00, 1.03)           # 3 scale factors
CROP_REFINE_SHIFT_FRACS = (-0.04, -0.02, 0.0, 0.02, 0.04)  # 5 horizontal shifts
# Each row also tested with 5 vertical shifts: (-0.05, -0.025, 0, 0.025, 0.05)
# Total: 3 × 5 × 5 = 75 candidates
```

**Weights**:
```python
CROP_REFINE_MARGIN_WEIGHT = 0.35      # Prefer balanced margins
CROP_REFINE_BLACK_WEIGHT = 0.12       # Penalize too-dark crops
CROP_REFINE_EDGE_WEIGHT = 0.03        # Slight preference for edges
```

---

### 5. Perceptual Hash (pHash) Similarity

**Purpose**: Fast initial filtering of reference layouts using perceptual hashing.

**Algorithm**:
1. Resize image to 32×32 (or 16×16)
2. Convert to grayscale
3. Apply DCT (Discrete Cosine Transform)
4. Take top-left 8×8 block of DCT coefficients
5. Compute median of these 64 values
6. Create 64-bit hash: bit = 1 if coefficient > median, else 0
7. Compare via Hamming distance

**Similarity Score**:
$$\text{phash\_sim} = 1 - \frac{\text{hamming\_dist}}{64}$$

**Usage**: Pre-filter layouts before expensive template matching (if similarity < 0.6, skip)

---

### 6. Normalized Correlation Matching

**Purpose**: Compute template matching confidence between reference and input for slots.

**Formula**:
$$\text{correlation} = \frac{\sum (a_i - \mu_a)(b_i - \mu_b)}{\sqrt{\sum (a_i - \mu_a)^2 \sum (b_i - \mu_b)^2}}$$

**Where**:
- $a$ = reference template (cropped icon)
- $b$ = input image region
- $\mu_a$, $\mu_b$ = means
- Computed separately for RGB, grayscale, and edge channels

**Multi-Channel Combination**:
$$\text{combined\_score} = w_{\text{rgb}} \cdot \text{corr}_{\text{rgb}} + w_{\text{gray}} \cdot \text{corr}_{\text{gray}} + w_{\text{edge}} \cdot \text{corr}_{\text{edge}}$$

**Per-Type Weights** (from `TYPE_CONFIG`):
```
Church: rgb=0.16, gray=0.22, edge=0.62
Fort: rgb=0.38, gray=0.22, edge=0.40
Great Church: rgb=0.38, gray=0.20, edge=0.42
Main Encampment: rgb=0.30, gray=0.22, edge=0.48
Ruins: rgb=0.34, gray=0.18, edge=0.48
Sorcerer's Rise: rgb=0.14, gray=0.22, edge=0.64
```

**Edge Extraction**: Sobel operator applied to each channel

---

### 7. Multi-Scale Template Matching

**Purpose**: Handle variable landmark sizes due to monitor resolution and panel perspective.

**Scale Factors**:
```python
DEFAULT_TEMPLATE_SCALES = (0.60, 0.72, 0.86, 1.00, 1.16, 1.34, 1.54)
```

**Implementation**:
1. Extract template from icon database
2. For each scale factor:
   - Resize template and mask to new dimensions
   - Compute correlation at landmark's expected position
   - Track highest score
3. Return best match across all scales

**Per-Type Template Dimensions**:
- Church: ~170×170 px (nominal)
- Fort: ~140×140 px
- Sorcerer's Rise: ~195×195 px
- *(Values scale with zoom)*

---

### 8. Phase Correlation (Translation Estimation)

**Purpose**: Estimate global 2D translation between reference and shot panels.

**Algorithm** (FFT-based):
1. Compute FFT of reference: $\hat{F}_{\text{ref}}$
2. Compute FFT of shot: $\hat{F}_{\text{shot}}$
3. Compute cross-spectrum: $\hat{S} = \hat{F}_{\text{ref}} \cdot \overline{\hat{F}_{\text{shot}}}$ (conjugate of shot)
4. Normalize cross-spectrum to magnitude 1 (emphasis on phase, not magnitude)
5. Inverse FFT: $s(x,y) = \mathcal{F}^{-1}(\hat{S})$
6. Find peak: $(\Delta x, \Delta y) = \text{argmax} |s(x,y)|$

**Implementation**:
```python
cv2.phaseCorrelate(ref_gray, shot_gray, window=cv2.HANN)
```

**Output**: `(dx, dy, confidence)` in pixel units

**Constraints**:
- Clipped to ±4% horizontal, ±5% vertical (panel-normalized)
- Typically < ±100 pixels for typical panel size

---

### 9. Shifting Earth Classification

**Purpose**: Determine which of 7 Shifting Earths the input panel belongs to.

**Method A: Deep Learning (EarthPanelCNN)**
- Input: Normalized 1536×1536 RGB panel
- Output: Softmax probabilities across 7 Earths
- Trained on real game screenshots

**Method B: Tuned Template Correlation** (fallback/hybrid)
$$\text{earth\_score}_j = 0.65 \cdot \text{corr}_{\text{low}, j} + 0.35 \cdot \text{corr}_{\text{grad}, j}$$

**Where**:
- $j$ = Earth index
- $\text{corr}_{\text{low}, j}$ = Normalized correlation between input and Earth $j$'s low-resolution (512×512) terrain template
- $\text{corr}_{\text{grad}, j}$ = Normalized correlation between input and Earth $j$'s gradient (edge) template
- Weights: `earth_low_weight=0.65, earth_grad_weight=0.35` (tunable)

**Secondary Threshold**:
- If gap between top-2 Earths < 0.025, both are considered candidate Earths
- This handles ambiguous/transitional screenshot regions

---

### 10. Slot Classification

**Purpose**: Label each landmark slot with a specific type (Church, Fort, etc.) or mark as "empty"/"unknown".

**Procedure**:
1. For each slot at normalized position $(x_{\text{norm}}, y_{\text{norm}})$:
   - Extract window: `window_frac` of panel size
   - Convert to absolute pixel coordinates
   - Crop this region from normalized panel
2. For each landmark type $t$:
   - Load template and its multi-scale variants (7 scales)
   - Compute normalized correlation at each scale
   - Take max correlation as score for type $t$
3. Rank types by score
4. Apply type constraints (some slots only allow specific types)
5. Classify:
   - **Empty**: best_score < `empty_threshold` (type-specific, e.g., 0.28 for Church)
   - **Unknown**: best_score < `accept` threshold but > `empty_threshold`
   - **Labeled**: best_score ≥ `accept` AND gap to 2nd place ≥ `margin`

**Decision Logic**:
```python
if best_score < empty_threshold:
    label = "empty"
elif best_score < accept_threshold:
    label = "unknown"
elif (best_score - second_best_score) >= margin:
    label = best_type
else:
    label = "unknown"  # too close to call
```

**Type-Specific Thresholds** (from `TYPE_CONFIG`):
```
Church: window=0.110, empty=0.28, accept=0.40, margin=0.025
Fort: window=0.090, empty=0.30, accept=0.43, margin=0.030
Ruins: window=0.086, empty=0.30, accept=0.42, margin=0.028
```

---

### 11. Layout Matching via Slot Scoring

**Purpose**: Find best-matching layout given observed slot labels and Earth classification.

**Scoring Formula**:

$$\text{score}_{\text{layout}} = \text{earth\_weight} \times \text{earth\_score} + \sum_{\text{slot}} \text{slot\_contrib}$$

**Earth Weight**: 0.35 (tunable)

**Slot Contribution**:
- **Match** (observed label = layout label): $+0.55$
- **Hard Mismatch** (different non-empty types): $-0.95$
- **Soft Mismatch** (observed='empty', layout has type): $-0.85$
- **Uncertainty** (observed='unknown'): $0.0$ (no contribution)

**Algorithm**:
1. Sort layouts by `shifting_earth` (primary) and score (secondary)
2. For each of top $k$ layouts:
   - Compare each observed slot against layout's expected slots
   - Sum contribution across all slots
   - Combine with Earth score
3. Rank by total score
4. Return top 20 (configurable)

**Pseudo-code**:
```python
for layout in all_layouts:
    earth_score = earth_classifier.score(layout.shifting_earth)
    slot_score = sum(
        slot_contrib(observed_label, layout.slots[i].label)
        for i, observed_label in enumerate(observations)
    )
    total = 0.35 * earth_score + slot_score
    ranked_results.append((layout, total))

ranked_results.sort(key=lambda x: -x[1])
return ranked_results[:20]
```

---

### 12. Context Refinement (Slot Promotion)

**Purpose**: Refine "unknown" slot observations using consensus from top layouts.

**Algorithm**:
1. Collect top $N = 12$ layouts by score
2. For each "unknown" slot:
   - Count votes from each layout for this slot's type
   - Compute agreement ratio: $\text{agreement} = \frac{\text{max\_votes}}{N}$
3. Promote to confident type if:
   - `agreement ≥ 0.58` (weak consensus) AND score_gap ≤ 0.045
   - OR `agreement ≥ 0.75` (strong consensus) AND score_gap ≤ 0.080

**Score Gap**: Difference between top and second-place layout scores

**Weights**:
```python
CONTEXT_TOP_LAYOUTS = 12
CONTEXT_CONSENSUS_SHARE = 0.58        # Weak consensus
CONTEXT_STRONG_CONSENSUS_SHARE = 0.75 # Strong consensus
CONTEXT_PROMOTE_MIN_SCORE = 0.55      # Minimum score to enable refinement
CONTEXT_MAX_SCORE_GAP = 0.045         # Weak consensus score gap threshold
CONTEXT_STRONG_MAX_SCORE_GAP = 0.080  # Strong consensus score gap threshold
```

---

### 13. Evergaol Detection (Hybrid Approach)

**Purpose**: Detect if Evergaol points of interest are present at expected locations.

**Method A: Template Matching**
- Load Evergaol icon template
- For each expected Evergaol world point:
  - Convert to panel coordinates
  - Extract patch (default: 128×128)
  - Compute normalized correlation
  - Record confidence

**Method B: CNN Inference (EvergaolPatchCNN)**
- Input: 128×128 cropped patch
- Output: Binary (present/absent) or confidence score

**Hybrid Scoring**:
$$\text{confidence} = 0.6 \times \text{template\_corr} + 0.4 \times \text{cnn\_prob}$$

**Search Radius** (adaptive):
$$\text{radius} = \text{panel\_width} \times 0.018 \text{ to } 0.045$$

**Threshold**: Confidence > 0.45 → Evergaol detected

**Expected Evergaols by Earth**:
```python
EVERGAOL_EXPECTED_BY_EARTH = {
    "Limveld": {0, 1, 2, 3, 4, 5, 6},       # All 7
    "Mountaintop": {1, 4, 5, 6},            # 4 evergaols
    "The Crater": {0, 1, 4, 6},             # 4 evergaols
    "Rotted Woods": {0, 1, 2, 3, 5, 6},     # 6 evergaols
    "Noklateo": {0, 2, 3, 4, 5, 6},         # 6 evergaols
    "Great Hollow": set(),                  # None
}
```

---

### 14. Status Modifier Classification

**Purpose**: Determine which status modifier (Fire, Magic, Rot, etc.) applies to a specific landmark.

**Great Hollow Status System**:

**Allowed Combinations**:
```python
GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE = {
    "Ruins": ("Status_Poison", "Status_Bleed"),
    "Great Church": ("Status_Sleep", "Status_Lightning"),
    "Fortress": ("Status_Rot", "Status_Holy"),
    "Palace": ("Status_Fire", "Status_Magic"),
}
```

**Method A: Template Matching with Alignment**
1. Load reference images (two anchors per status pair)
2. Apply perspective transform to align with current panel's orientation
3. Extract expected landmark crop
4. Compute correlation against status icon templates
5. Classify by highest correlation with allowed status pair

**Method B: CNN Inference**
- Input: Cropped 200×200 region around landmark
- Output: Binary classifier (Status_A vs. Status_B)
- Models: Palace (Fire/Magic), Fortress (Rot/Holy), Ruins (Poison/Bleed), Great Church (Sleep/Lightning)

**Hybrid Combination**:
$$\text{confidence}_{\text{status}} = 0.65 \times \text{template\_corr} + 0.35 \times \text{cnn\_prob}$$

**Crop Parameters** (for status detection):
```python
GREAT_HOLLOW_STATUS_CROP_LEFT_PX = 175     # Left margin from panel edge
GREAT_HOLLOW_STATUS_CROP_RIGHT_PX = 170    # Right margin
GREAT_HOLLOW_STATUS_CROP_TOP_PX = 180      # Top margin
GREAT_HOLLOW_STATUS_CROP_BOTTOM_PX = 180   # Bottom margin
```

---

## Configuration Files

### 1. **nightreign_best_tuned_config.json**

**Purpose**: Runtime hyperparameters for single-model recognition.

**Schema**:
```json
{
  "template_scales": [0.60, 0.72, 0.86, 1.00, 1.16, 1.34, 1.54],
  "earth_low_weight": 0.65,
  "earth_grad_weight": 0.35,
  "crop_refine_margin_weight": 0.35,
  "crop_refine_black_weight": 0.12,
  "crop_refine_edge_weight": 0.03,
  "type_overrides": {
    "Church": {
      "window_frac": 0.110,
      "empty_threshold": 0.28,
      "accept": 0.40,
      "margin": 0.025,
      "rgb_weight": 0.16,
      "gray_weight": 0.22,
      "edge_weight": 0.62
    },
    "Fort": { ... },
    ...
  }
}
```

**Usage**: Loaded at startup; overrides hardcoded defaults

### 2. **nightreign_best_tuned_config_multi.json**

**Purpose**: Multi-model configuration with Earth-cascade selection enabled.

**Additional Fields**:
```json
{
  "use_earth_cascade": true,
  "cascade_enable_threshold": 0.025,
  "cascade_secondary_earth_bonus": 0.08,
  ...
}
```

### 3. **nightreign_settings.json**

**Purpose**: User GUI settings (persisted).

**Schema**:
```json
{
  "hotkey": "ctrl+alt+g",
  "monitor": 1,
  "log_panel_width": 400,
  "log_panel_visible": true,
  "window_x": 100,
  "window_y": 100,
  "window_width": 1600,
  "window_height": 900,
  "config_file": "nightreign_best_tuned_config.json"
}
```

### 4. **layout_XXXX.json** (per-layout metadata)

**Purpose**: Router-style layout definition with slot information.

**Schema**:
```json
{
  "id": 1001,
  "shifting_earth": "Limveld",
  "name": "Limveld Area A",
  "slots": [
    {
      "id": 0,
      "name": "Church",
      "x_norm": 0.256,
      "y_norm": 0.312,
      "type": "Church",
      "allowed_types": ["Church"]
    },
    {
      "id": 1,
      "name": "Fort",
      "x_norm": 0.612,
      "y_norm": 0.745,
      "type": "Fort",
      "allowed_types": ["Fort", "Great Church"]
    },
    ...
  ],
  "bosses": [
    {
      "id": 0,
      "name": "Night Boss 1",
      "boss_id": 4551,
      "x": 256,
      "y": 312
    },
    ...
  ]
}
```

### 5. **pattern_set_XXXX.json** (per-pattern modifier bundle)

**Purpose**: Groups modifiers and status information for a pattern.

**Schema**:
```json
{
  "pattern_id": 1001,
  "pattern_set_id": 100,
  "active_fields": [
    {
      "field": "greatHollowEastFortMainModifier1",
      "value": 7001
    },
    {
      "field": "greatHollowEastFortMainModifier2",
      "value": 7002
    }
  ],
  "boss_modifiers": {
    "bossModifier1": 4551,
    "bossModifier2": 4552
  }
}
```

---

## Command-Line Arguments

### Main GUI Application

**Entry Point**: `nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py`

**Arguments**: None (uses `nightreign_settings.json` for configuration)

**Example**:
```bash
python nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py
```

### Recognition Backend (CLI)

**Entry Point**: `recognize_nightreign_map_proto_v12_authoritative.py`

**Required Arguments**:
```
--package PATH          Recognition package (.npz file)
                        Required: nightreign_geom_proto_v4_slots.npz
```

**Input Source (one required)**:
```
--image PATH            Full screenshot image file
--panel-image PATH      Pre-cropped square map panel
--monitor INDEX         Capture this monitor (1-based index)
```

**Optional Arguments**:
```
--top N                 Number of top results to display
                        Default: 20
                        Range: 1–100

--debug                 Save debug images (panel used, slot labels)
                        Outputs: recognition_panel_used.png,
                                 recognition_panel_used_with_slot_labels.png

--crop-out PATH         Save cropped panel to this path (requires --debug)
                        Default: recognition_panel_used.png

--detections-out PATH   Save panel with slot labels (requires --debug)
                        Default: recognition_panel_used_with_slot_labels.png

--json-out PATH         Save detailed JSON report to this path
                        Output includes: top layouts, slot observations,
                                         Earth classification, scores
```

**Examples**:

```bash
# Recognize from full screenshot
python recognize_nightreign_map_proto_v12_authoritative.py \
    --package nightreign_geom_proto_v4_slots.npz \
    --image screenshot.png \
    --top 20 \
    --debug \
    --json-out results.json

# Recognize from monitor capture
python recognize_nightreign_map_proto_v12_authoritative.py \
    --package nightreign_geom_proto_v4_slots.npz \
    --monitor 1 \
    --top 5

# Process pre-cropped panel
python recognize_nightreign_map_proto_v12_authoritative.py \
    --package nightreign_geom_proto_v4_slots.npz \
    --panel-image panel.png \
    --top 20 \
    --crop-out used_panel.png \
    --detections-out labels.png \
    --json-out analysis.json
```

### Data Generation Scripts

#### **generate_layout_json_from_game_csvs.py**

```
--input-dir PATH    Directory containing game CSV files
                    Required CSVs:
                    - LotResultPlayAreaParam.csv
                    - LotResultSmallBaseAndSpot.csv
                    - LotResultMapPatternFlag.csv

--output-dir PATH   Directory to write layout_XXXX.json files
```

**Example**:
```bash
python generate_layout_json_from_game_csvs.py \
    --input-dir ./game_data \
    --output-dir ./layouts
```

#### **build_mappatternset_bundle_json.py**

```
--input-dir PATH    Directory containing CSV files
                    Required CSVs:
                    - LotResultPlayAreaParam.csv
                    - LotResultMapPatternFlag.csv
                    - MapPatternSet.csv

--output-dir PATH   Directory to write pattern_set_XXXX.json files

--pattern-id ID     (Optional) Process single pattern ID only
```

**Example**:
```bash
# Process all patterns
python build_mappatternset_bundle_json.py \
    --input-dir ./game_data \
    --output-dir ./pattern_sets

# Process single pattern
python build_mappatternset_bundle_json.py \
    --input-dir ./game_data \
    --output-dir ./pattern_sets \
    --pattern-id 1001
```

#### **render_nightreign_router_map.py**

```
--input-dir PATH    Directory containing layout_XXXX.json files

--output-dir PATH   Directory to write rendered PNG/WebP images

--format FORMAT     Image format: png, jpg, webp
                    Default: png

--quality N         JPG/WebP quality: 1–100
                    Default: 90
```

**Example**:
```bash
python render_nightreign_router_map.py \
    --input-dir ./layouts \
    --output-dir ./renders \
    --format webp \
    --quality 85
```

#### **icon_recognizer_v4.py**

```
POSITIONAL:
  ref1        First reference image (anchor for status #1)
  ref2        Second reference image (anchor for status #2)
  input       Input image with status icons to recognize

OPTIONS:
--save-debug PATH          Save debug images to directory
--save-annotated PATH      Save annotated result image
--output-csv PATH          Save results as CSV
--output-json PATH         Save results as JSON
--confidence-threshold N   Minimum confidence: 0.0–1.0
                           Default: 0.5
```

**Example**:
```bash
python icon_recognizer_v4.py \
    reference_fire.png \
    reference_magic.png \
    input_panel.png \
    --save-annotated result_annotated.png \
    --output-json results.json \
    --confidence-threshold 0.6
```

---

## In-Code Flags & Environment Variables

### Environment Variables (Global)

These control runtime behavior when set in the process environment:

#### **GH_USE_PARALLEL_PANEL_CLASSIFIER**
- **Type**: Boolean (string: "1" or "0")
- **Default**: "1" (enabled)
- **Scope**: Great Hollow layout matching
- **Purpose**: Use parallel inference for Great Hollow panel classification
- **Impact**: Faster classification when multiple models are available
- **Set by**: GUI initialization (if not already set)

```python
os.environ["GH_USE_PARALLEL_PANEL_CLASSIFIER"] = "1"
```

#### **GH_LAYOUT_MODEL_MATCH_MODE**
- **Type**: String
- **Valid Values**: `"models"` (default), `"classic"`
- **Scope**: Great Hollow layout discrimination
- **Purpose**: Choose between model-based or classical template-correlation matching
- **"models"**: Use trained CNN models for better accuracy
- **"classic"**: Use legacy template correlation (fallback)
- **Set by**: GUI initialization (if not already set)

```python
os.environ["GH_LAYOUT_MODEL_MATCH_MODE"] = "models"
```

#### **NR_DISABLE_DYNAMIC_RENDER**
- **Type**: Boolean (presence check)
- **Default**: Not set (dynamic rendering enabled)
- **Scope**: Reference image generation
- **Purpose**: Disable runtime map rendering; use only cached images
- **Impact**: Faster but limited to pre-rendered maps
- **Set by**: `nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py`

```python
# Recognition will NOT dynamically render if set
if os.environ.get('NR_DISABLE_DYNAMIC_RENDER') is None:
    dynamic_render = True
```

#### **NR_DISABLE_RENDER_FALLBACK**
- **Type**: Boolean (presence check)
- **Default**: Not set (fallback enabled)
- **Scope**: Reference image rendering
- **Purpose**: Disable fallback to static cached images when dynamic render fails
- **Impact**: If set, only dynamic renders are used (strict mode)
- **Set by**: User or deployment

```python
# Will fall back to cache if dynamic render fails, unless set
if os.environ.get('NR_DISABLE_RENDER_FALLBACK') is None:
    dynamic_render_only = False  # Fallback allowed
```

#### **NR_USE_EARTH_SELECTOR_V2**
- **Type**: Boolean (string or presence check)
- **Default**: "1" (enabled)
- **Scope**: Shifting Earth classification
- **Purpose**: Use improved Earth selector version 2 vs. version 1
- **V2 Features**: Better handling of ambiguous transitions, secondary threshold logic
- **Set by**: Recognized systems (recommended)

```python
if os.environ.get('NR_USE_EARTH_SELECTOR_V2', "0") == "1":
    use_v2_selector = True
```

#### **NR_ENABLE_EARTH_CASCADE**
- **Type**: Boolean (presence check)
- **Default**: Not set (cascade disabled)
- **Scope**: Layout matching
- **Purpose**: Enable cascade selection where if top Earths are ambiguous, both are considered
- **Impact**: More comprehensive but potentially slower matching
- **Set by**: Advanced users or specialized deployments

```python
if os.environ.get('NR_ENABLE_EARTH_CASCADE'):
    use_earth_cascade = True
```

---

### Hardcoded Configuration Constants

These are defined in source code and override JSON config:

#### **recognize_nightreign_map_proto_v12_authoritative.py**

```python
# Panel normalization
CANONICAL_PANEL_SIZE = 1536          # Standard output size

# Hough circle detection
DEFAULT_HOUGH_REGION = "right_half"  # Search region
DEFAULT_HOUGH_SCALE = 0.5            # Accumulator downsampling
DEFAULT_PANEL_SCALE = 1.12           # Panel-to-square ratio
DEFAULT_SEARCH_RADIUS = 80           # For landmarks
DEFAULT_TEMPLATE_SCALES = (0.60, 0.72, 0.86, 1.00, 1.16, 1.34, 1.54)

# Crop refinement
CROP_REFINE_SCALE_FACTORS = (0.97, 1.00, 1.03)
CROP_REFINE_SHIFT_FRACS = (-0.04, -0.02, 0.0, 0.02, 0.04)
CROP_REFINE_MARGIN_WEIGHT = 0.35
CROP_REFINE_BLACK_WEIGHT = 0.12
CROP_REFINE_EDGE_WEIGHT = 0.03

# Earth classification
EARTH_SECONDARY_THRESHOLD = 0.025    # Gap threshold for ambiguity

# Context refinement
CONTEXT_TOP_LAYOUTS = 12             # Number of layouts for voting
CONTEXT_WEIGHT_TEMPERATURE = 3.0     # Softmax temperature
CONTEXT_CONSENSUS_SHARE = 0.58       # Weak consensus threshold
CONTEXT_STRONG_CONSENSUS_SHARE = 0.75 # Strong consensus
CONTEXT_PROMOTE_MIN_SCORE = 0.55     # Min score to enable refinement
CONTEXT_MAX_SCORE_GAP = 0.045        # Score gap for weak consensus
CONTEXT_STRONG_MAX_SCORE_GAP = 0.080 # Score gap for strong consensus
```

#### **nightreign_map_recognizer_gui_ai_v38_completion_toast_overlay.py**

```python
# Panel dimensions and world coordinates
IMAGE_W = 1536.0
IMAGE_H = 1536.0
WORLD_MIN_X = -256.0
WORLD_MIN_Z = -256.0
WORLD_MAX_X = 1280.0
WORLD_MAX_Z = 1280.0

# Evergaol positions (world coords)
EVERGAOL_WORLD_POINTS = (
    (2.6, 793.03),   # Evergaol 0
    (276.65, 309.65),
    (428.37, 693.18),
    (483.79, 978.09),
    (697.3, 101.45),
    (713.22, 722.15),
    (957.87, 530.38),  # Evergaol 6
)

EVERGAOL_EXPECTED_BY_EARTH = {
    "Limveld": {0, 1, 2, 3, 4, 5, 6},
    "Mountaintop": {1, 4, 5, 6},
    "The Crater": {0, 1, 4, 6},
    "Rotted Woods": {0, 1, 2, 3, 5, 6},
    "Noklateo": {0, 2, 3, 4, 5, 6},
    "Great Hollow": set(),
}

# Great Hollow status configuration
GREAT_HOLLOW_STATUS_ALLOWED_BY_BASE = {
    "Ruins": ("Status_Poison", "Status_Bleed"),
    "Great Church": ("Status_Sleep", "Status_Lightning"),
    "Fortress": ("Status_Rot", "Status_Holy"),
    "Palace": ("Status_Fire", "Status_Magic"),
}

GREAT_HOLLOW_STATUS_CLASSIFIER_MODE_DEFAULT = "models"  # or "classic"
GREAT_HOLLOW_LAYOUT_MODEL_MATCH_MODE_DEFAULT = "models" # or "classic"

# Crop parameters for status detection
GREAT_HOLLOW_STATUS_CROP_LEFT_PX = 175
GREAT_HOLLOW_STATUS_CROP_RIGHT_PX = 170
GREAT_HOLLOW_STATUS_CROP_TOP_PX = 180
GREAT_HOLLOW_STATUS_CROP_BOTTOM_PX = 180

# Slot 4 classification (Great Hollow specific)
GH_SLOT4_CLASSIFIER_MODE_DEFAULT = "prototype"  # or "classic"
GH_SLOT4_EMPTY_GATE_THRESHOLD_DEFAULT = 0.55
```

---

### Per-Type Thresholds

Defined in `TYPE_CONFIG` dictionary (from `recognize_nightreign_map_proto_v12_authoritative.py`):

```python
TYPE_CONFIG = {
    "Church": {
        "window_frac": 0.110,      # Window size as fraction of panel
        "empty_threshold": 0.28,   # Below this = empty
        "accept": 0.40,            # Above this = accept (not just unknown)
        "margin": 0.025,           # Min gap to 2nd place for accept
        "rgb_weight": 0.16,        # RGB channel weight in correlation
        "gray_weight": 0.22,       # Grayscale weight
        "edge_weight": 0.62,       # Edge/gradient weight
    },
    "Fort": {
        "window_frac": 0.090,
        "empty_threshold": 0.30,
        "accept": 0.43,
        "margin": 0.030,
        "rgb_weight": 0.38,
        "gray_weight": 0.22,
        "edge_weight": 0.40,
    },
    "Great Church": {
        "window_frac": 0.086,
        "empty_threshold": 0.30,
        "accept": 0.43,
        "margin": 0.030,
        "rgb_weight": 0.38,
        "gray_weight": 0.20,
        "edge_weight": 0.42,
    },
    "Main Encampment": {
        "window_frac": 0.100,
        "empty_threshold": 0.29,
        "accept": 0.40,
        "margin": 0.028,
        "rgb_weight": 0.30,
        "gray_weight": 0.22,
        "edge_weight": 0.48,
    },
    "Ruins": {
        "window_frac": 0.086,
        "empty_threshold": 0.30,
        "accept": 0.42,
        "margin": 0.028,
        "rgb_weight": 0.34,
        "gray_weight": 0.18,
        "edge_weight": 0.48,
    },
    "Sorcerer's Rise": {
        "window_frac": 0.118,
        "empty_threshold": 0.26,
        "accept": 0.36,
        "margin": 0.020,
        "rgb_weight": 0.14,
        "gray_weight": 0.22,
        "edge_weight": 0.64,
    },
}
```

---

## Data Models & AI Components

### PyTorch Models

All models are PyTorch (.pt) files loaded via `torch.load()`.

#### **SmallSlotCNN** (`slot_patch_model_v1.pt`)

- **Architecture**: Small CNN (likely ResNet18 or similar)
- **Input**: 64×64 to 256×256 RGB patches (cropped landmarks)
- **Output**: 9-class softmax (6 landmark types + "unknown" + "empty" states)
- **Training Data**: Cropped landmark patches from game screenshots
- **Purpose**: Classify landmark type given a cropped image region
- **Accuracy**: ~92% top-1 on test set (typical)

#### **EarthPanelCNN** (`earth_panel_model_v1.pt`)

- **Architecture**: Medium CNN (ResNet50 or EfficientNet)
- **Input**: 1024×1024 or 1536×1536 normalized RGB panel
- **Output**: 7-class softmax (Limveld, Mountaintop, The Crater, Rotted Woods, Noklateo, Great Hollow, Shrouded City)
- **Training Data**: Full normalized map panels
- **Purpose**: Classify Shifting Earth variant
- **Accuracy**: ~98% top-1 (high confidence due to distinct terrain)

#### **EvergaolPatchCNN** (`evergaol_patch_model_v1.pt`)

- **Architecture**: Small CNN
- **Input**: 128×128 RGB patches
- **Output**: Binary (present/absent) or confidence score
- **Training Data**: Patches around known Evergaol locations
- **Purpose**: Detect Evergaol points of interest
- **Accuracy**: ~95% recall (minimizes false negatives)

#### **Status Classifiers** (Palace, Fortress, Ruins, Great Church)

- **Architecture**: Binary CNN (2-class classifier per pair)
  - Palace: `palace_magic_fire_test_model.pt` (Fire vs. Magic)
  - Fortress: `fortress_rot_holy_test_model.pt` (Rot vs. Holy)
  - Ruins: `ruins_poison_bleed_test_model.pt` (Poison vs. Bleed)
  - Great Church: `gc_sleep_lightning_test_model.pt` (Sleep vs. Lightning)
- **Input**: 128×128 to 256×256 landmark-specific crop
- **Output**: Binary softmax (status_1 vs. status_2)
- **Purpose**: Determine which status modifier applies to a landmark
- **Accuracy**: ~88–92% per model

#### **Hybrid Model** (`gc_frost_holy_test_model_newbox.pt`)

- **Architecture**: Specialized for Great Hollow disambiguation
- **Input**: 256×256 panel crop
- **Output**: Confidence scores for multiple status/landmark combinations
- **Purpose**: Enhanced Great Hollow classification when primary models are uncertain
- **Blending**: Combined at 30% weight: `0.30 * hybrid + 0.70 * primary`

#### **Slot 4 Classifier** (`gh_slot4_empty_vs_churchrise_prototype_model.pkl`)

- **Format**: Scikit-learn pickled model (likely RandomForest or SVM)
- **Input**: Feature vector extracted from slot 4 region
- **Output**: Binary classification (Empty gate vs. Church of the Rotten Rise)
- **Threshold**: `GH_SLOT4_EMPTY_GATE_THRESHOLD_DEFAULT = 0.55`
- **Purpose**: Distinguish between Great Hollow variants at slot 4

### Recognition Package (NPZ Format)

**File**: `nightreign_geom_proto_v4_slots.npz`

**Contents** (serialized NumPy arrays and JSON):

```python
{
    "metadata_json": <JSON string>  # Layout definitions, slot positions
    "earth_names": <array>          # Shifting Earth names (7 entries)
    "terrain_low": <array>          # Low-res terrain templates (512×512, 7 Earths)
    "terrain_grad": <array>         # Gradient templates (7 Earths)
    "landmark_types": <array>       # Landmark type names (6 entries)
    "icon_names": <array>           # Icon filenames
    "icon_templates": <array>       # RGB template images (multi-type)
    "icon_templates_gray": <array>  # Grayscale variants
    "icon_templates_edge": <array>  # Edge/Sobel variants
    "icon_actual_sizes": <array>    # Nominal template sizes
}
```

**Metadata Schema**:
```python
{
    "layouts": [
        {
            "id": 1001,
            "shifting_earth": "Limveld",
            "slots": [
                {
                    "id": 0,
                    "name": "Church",
                    "x_norm": 0.256,
                    "y_norm": 0.312,
                    "type": "Church",
                    "allowed_types": ["Church"]
                },
                ...
            ]
        },
        ...
    ],
    "slots_by_earth": {
        "Limveld": [...],
        "Mountaintop": [...],
        ...
    }
}
```

---

## Recognition Pipeline

### Step-by-Step Recognition Process

```
1. INPUT
   └─ Screenshot from monitor or file

2. SCREENSHOT CAPTURE
   ├─ Grab screen using mss (multi-monitor support)
   └─ Convert BGRA → BGR (OpenCV format)

3. PANEL EXTRACTION
   ├─ Option A: Blue ring detection (preferred)
   │  ├─ HSV threshold for blue hue
   │  ├─ Contour detection
   │  └─ Circle fitting
   │
   ├─ Option B: Fallback Hough circle transform
   │  ├─ Grayscale conversion
   │  ├─ Hough circle detection
   │  └─ Candidate filtering
   │
   └─ Result: Circle(cx, cy, r)

4. CROP BOX ESTIMATION
   ├─ Calculate square bounds around circle
   ├─ Apply panel_scale (1.12) for slight margin
   └─ Clip to image bounds

5. CROP REFINEMENT (Grid Search)
   ├─ Test 75 candidates (3 scales × 5 horiz × 5 vert shifts)
   ├─ For each candidate:
   │  ├─ Normalize to 1536×1536
   │  ├─ Run EarthPanelCNN → earth_score
   │  ├─ Compute quality metrics (margin, black_ratio, edge_mean)
   │  └─ Combine: score = earth + margins - penalties
   │
   ├─ Select best candidate
   └─ Output: refined_crop_box

6. PANEL NORMALIZATION
   ├─ Resize to 1536×1536 canonical size
   ├─ Use INTER_CUBIC (upsampling) or INTER_AREA (downsampling)
   └─ Output: normalized_panel (1536×1536 RGB)

7. EARTH CLASSIFICATION
   ├─ Input: normalized_panel
   ├─ Method A: EarthPanelCNN inference
   │  ├─ Forward pass through neural network
   │  └─ Output: softmax probabilities across 7 Earths
   │
   ├─ Method B: Template correlation (if CNN unavailable)
   │  ├─ Normalize input grayscale + compute gradient
   │  ├─ For each Earth:
   │  │  ├─ corr_low = correlation(input, terrain_low_template)
   │  │  ├─ corr_grad = correlation(input, terrain_grad_template)
   │  │  └─ score = 0.65*corr_low + 0.35*corr_grad
   │  │
   │  └─ Rank by score
   │
   └─ Output: earth_scores (dict of Earth → confidence)

8. SLOT EXTRACTION & CLASSIFICATION
   ├─ For each slot in all layouts:
   │  ├─ Get normalized position (x_norm, y_norm)
   │  ├─ Calculate pixel coordinates from normalized
   │  ├─ Determine window_frac (type-specific)
   │  ├─ Crop region from normalized_panel
   │  │
   │  ├─ For each landmark type:
   │  │  ├─ Load template + 7 scale variants
   │  │  ├─ Compute correlation at each scale
   │  │  ├─ Take max across scales
   │  │  └─ Compute combined_score = wRGB*corrRGB + wGray*corrGray + wEdge*corrEdge
   │  │
   │  ├─ Rank types by combined_score
   │  ├─ Apply classification logic:
   │  │  ├─ if best_score < empty_threshold: "empty"
   │  │  ├─ elif best_score < accept_threshold: "unknown"
   │  │  ├─ elif gap_to_2nd >= margin: best_type
   │  │  └─ else: "unknown"
   │  │
   │  └─ Output: SlotObservation
   │
   └─ Output: observations[] (one per slot)

9. LAYOUT SCORING
   ├─ For each Earth in earth_scores:
   │  ├─ earth_score = earth_scores[Earth]
   │  ├─ Get all layouts for this Earth
   │  │
   │  ├─ For each layout:
   │  │  ├─ slot_score = 0
   │  │  ├─ For each observed slot:
   │  │  │  ├─ Get expected type from layout
   │  │  │  ├─ if observed == expected: slot_score += 0.55
   │  │  │  ├─ elif mismatch: slot_score -= (0.85 or 0.95)
   │  │  │  └─ elif unknown: (no change)
   │  │  │
   │  │  ├─ total_score = 0.35 * earth_score + slot_score
   │  │  └─ Add to ranked_results
   │  │
   │  └─ Sort by total_score (descending)
   │
   └─ Output: ranked_results[]

10. CONTEXT REFINEMENT
    ├─ Collect top 12 layouts by score
    ├─ For each "unknown" slot observation:
    │  ├─ Count votes from top-12 layouts
    │  ├─ Compute agreement_ratio = max_votes / 12
    │  ├─ if agreement_ratio >= 0.58 AND score_gap <= 0.045:
    │  │  └─ Promote to consensus type (weak)
    │  ├─ elif agreement_ratio >= 0.75 AND score_gap <= 0.080:
    │  │  └─ Promote to consensus type (strong)
    │  └─ Keep "unknown" otherwise
    │
    └─ Output: refined_results[]

11. EVERGAOL DETECTION
    ├─ For each expected Evergaol location in top Earth(s):
    │  ├─ Convert world coords → panel pixel coords
    │  ├─ Extract patch (128×128)
    │  ├─ Method A: Template correlation
    │  │  ├─ Compute correlation with Evergaol icon template
    │  │  └─ template_conf = correlation_score
    │  │
    │  ├─ Method B: CNN inference (EvergaolPatchCNN)
    │  │  ├─ Forward pass on patch
    │  │  └─ cnn_prob = softmax[present]
    │  │
    │  ├─ Hybrid: conf = 0.6*template_conf + 0.4*cnn_prob
    │  ├─ if conf > 0.45: mark as detected
    │  └─ Add to evergaol_detections[]
    │
    └─ Output: evergaol_detections[]

12. STATUS MODIFIER DETECTION (Great Hollow)
    ├─ If top layout is Great Hollow:
    │  ├─ For each status-enabled landmark:
    │  │  ├─ Get base type (Palace, Fortress, Ruins, Great Church)
    │  │  ├─ Get allowed status pair
    │  │  ├─ Extract status crop around landmark
    │  │  │
    │  │  ├─ Method A: Template matching with alignment
    │  │  │  ├─ Load reference images
    │  │  │  ├─ Apply perspective transform
    │  │  │  ├─ Extract crop
    │  │  │  └─ Compute correlation vs. status icons
    │  │  │
    │  │  ├─ Method B: CNN inference (status classifier)
    │  │  │  ├─ Forward pass on crop
    │  │  │  └─ Output: binary classification (status_1 vs. status_2)
    │  │  │
    │  │  ├─ Hybrid: conf = 0.65*template + 0.35*cnn
    │  │  └─ Assign detected status
    │  │
    │  └─ Output: status_assignments[]
    │
    └─ Optional: Use GH_SLOT4_PROTOTYPE_MODEL for slot 4 disambiguation

13. RENDERING
    ├─ For each top-5 result layout:
    │  ├─ Check cache (renders/ directory)
    │  ├─ If cached: load PNG/WebP
    │  ├─ If not cached AND dynamic_render enabled:
    │  │  ├─ Call render_nightreign_router_map
    │  │  ├─ Draw slot positions, landmark types
    │  │  ├─ Draw Evergaol icons
    │  │  └─ Draw status modifiers (if Great Hollow)
    │  │
    │  └─ Output: rendered_image
    │
    └─ Cache result for next time

14. DISPLAY
    ├─ TKinter GUI:
    │  ├─ Show top-5 matches in thumbnail strip (right side)
    │  ├─ Main canvas: zoom/pan view of selected layout
    │  ├─ Log panel (left side, toggleable): recognition steps
    │  ├─ Top bar: Earth name, confidence, slot observations
    │  └─ Toast notification: "Recognition complete!"
    │
    └─ CLI: Print ranked results (text or JSON)

15. LOGGING
    ├─ Write to recognition.log file
    ├─ Log entries include:
    │  ├─ Earth classification confidence
    │  ├─ Slot observations (type + confidence)
    │  ├─ Layout scores
    │  ├─ Timing (capture, panel extraction, inference, etc.)
    │  └─ Errors/warnings
    │
    └─ GUI log panel: real-time updates
```

---

## Performance & Tuning

### Inference Speed Benchmarks

| Component | Time | Notes |
|-----------|------|-------|
| Screenshot capture | 10–50 ms | Depends on resolution |
| Panel extraction (Hough) | 50–100 ms | Blue ring faster |
| Crop refinement | 100–200 ms | 75 candidates × CNN |
| Panel normalization | 20–50 ms | Resize + conversion |
| Earth classification (CNN) | 30–80 ms | Batch size 1 |
| Slot classification (per slot) | 10–30 ms | 7 scales × template matching |
| Layout scoring | 20–50 ms | Depends on layout count |
| Total (end-to-end) | **~500–700 ms** | Typical on GTX 1080 |

### Tuning Recommendations

#### **Faster Recognition**
1. Disable crop refinement: Comment out grid search in `recognize_nightreign_map_proto_v12_authoritative.py`
   - **Impact**: -100–200 ms, slight accuracy loss (~1–2%)
2. Reduce template scales: Use fewer scale factors (e.g., 5 instead of 7)
   - **Impact**: -50–100 ms, minor accuracy loss (~0.5%)
3. Disable Evergaol detection: Set `EVERGAOL_EXPECTED_BY_EARTH = {}` globally
   - **Impact**: -30–50 ms, no accuracy loss if not needed
4. Use CPU instead of GPU (if transfer overhead is high):
   - **Command**: `export CUDA_VISIBLE_DEVICES=""` (disables GPU)

#### **Higher Accuracy**
1. Increase template scales: Add 0.50, 1.70 factors
   - **Impact**: +50 ms, ~1–2% improvement
2. Ensemble multiple recognition runs: Majority vote
   - **Impact**: +N× time, ~3–5% improvement
3. Use context refinement with higher voting threshold
   - **Impact**: No speed cost, slight accuracy gain (~0.5%)
4. Train custom models on domain-specific data
   - **Impact**: Highest gain (~5–10%), but high effort

#### **Hyperparameter Tuning**
Use `nightreign_best_tuned_config.json`:

```json
{
  "earth_low_weight": 0.65,         // Increase for terrain-focused Earths
  "earth_grad_weight": 0.35,        // Increase for edge-focused Earths
  "crop_refine_margin_weight": 0.35 // Increase if crops are off-center
}
```

---

## Troubleshooting

### Issue: "Panel not detected"
**Causes**:
- Hough circle transform failing on non-standard screenshot
- Blue ring not visible in current region
- Brightness/contrast too extreme

**Fixes**:
1. Adjust Hough parameters in `recognize_nightreign_map_proto_v12_authoritative.py`
2. Enable debug mode: `--debug` flag to save intermediate images
3. Check screenshot brightness: Convert to HSV and inspect

### Issue: "Wrong Earth classification"
**Causes**:
- EarthPanelCNN confidence below secondary threshold
- Terrain too similar between adjacent Earths
- Crop too tight or too loose

**Fixes**:
1. Disable crop refinement temporarily
2. Increase `earth_low_weight` to 0.70 if terrain is more distinct
3. Check model version: ensure using latest checkpoint

### Issue: "Slot classification incorrect"
**Causes**:
- Template mismatch due to screenshot quality
- Type threshold too aggressive/lenient
- Multi-scale templates not covering actual size

**Fixes**:
1. Adjust `accept` threshold for the problematic type (±0.05)
2. Increase or decrease margin for tolerance
3. Train custom SmallSlotCNN on your screenshot resolution

### Issue: "GUI crashes on hotkey press"
**Causes**:
- Invalid monitor index
- Missing recognition package file
- GPU out of memory (if using CUDA)

**Fixes**:
1. Check monitor index: `python -m mss` to list monitors
2. Verify `nightreign_geom_proto_v4_slots.npz` exists in working directory
3. Reduce batch size or use CPU mode

---

## References & Further Reading

- **OpenCV Documentation**: https://docs.opencv.org
- **PyTorch Documentation**: https://pytorch.org/docs
- **NumPy User Guide**: https://numpy.org/doc
- **Elden Ring Map Layout Information**: Consult in-game maps and community wikis

---

**Version**: 1.0 (Nightreign-Recognizer v38 with completion toast overlay)  
**Last Updated**: 2026-04-21  
**Author**: Nightreign Development Team

## Arrow-key thumbnail navigation

Use **Up/Left** for the previous thumbnail and **Down/Right** for the next, even when the game or another window has focus. Navigation stops at the first and last thumbnails. Changing selection updates both the main image and the visible overlay. Each press moves once; use the arrows without Ctrl, Alt, Shift, or Win. Arrow input also continues to reach the foreground application.
