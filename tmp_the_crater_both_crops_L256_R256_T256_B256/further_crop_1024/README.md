# The Crater Layout Classifier

This project predicts The Crater `patternId` from one `1024x1024` further-crop image.

## What it does

It implements three scorers:

1. **Image-level baseline**
   - global HOG descriptor
   - nearest-neighbor style similarity against the gallery

2. **Slot-aware scorer**
   - samples local patches at strict metadata coordinates
   - uses `layouts[].slots[]` and `layouts[].landmarks[]`
   - converts bottom-left metadata coordinates to top-left image indexing with:
     - `x_px = round(x_further_px_bl)`
     - `y_px_top_left = 1023 - round(y_further_px_bl)`

3. **Hybrid scorer**
   - combines image-level and slot-aware scores
   - returns top-5, margin, confidence, and optional `unknown`

## Coordinate contract

This code uses the strict The Crater metadata file and follows the required contract:

- image size: `1024 x 1024`
- metadata frame: same `further_crop_1024` image frame
- metadata origin: **bottom-left**
- image origin: **top-left**
- primary coordinate source: `x_further_px_bl`, `y_further_px_bl`
- fallback only: `x_further_norm_bl`, `y_further_norm_bl`
- no other coordinate system is used
- no extra crop transform is applied

## Dataset note

The uploaded dataset has **one image per patternId**.
Because of that, a normal unseen-class train/val/test split is not meaningful for a closed-set pattern matcher.

So this project uses:

- a **full gallery** of all known layouts for matching
- **deterministic train/val/test query splits** built from light image augmentations
- a separate report on the original gallery images themselves

This is documented in the saved training report.

## Files

- `train_layout_classifier.py` — builds the model and writes a training report
- `infer_layout_classifier.py` — predicts one image
- `evaluate_layout_classifier.py` — evaluates a folder of images and writes a JSON report
- `layout_classifier_lib.py` — shared library code
- `models/layout_model.pkl` — saved trained matcher model
- `models/training_report.json` — saved training/evaluation summary
- `models/evaluation_report.json` — saved evaluation-only summary

## Python requirements

Install:

```bash
python -m pip install numpy opencv-python
```

No `scikit-learn` dependency is required.

## Train

```bash
python train_layout_classifier.py \
  --images /path/to/tmp_the_crater_both_crops_L256_R256_T256_B256/further_crop_1024 \
  --metadata /path/to/the_crater_layout_slots_further_crop_bl_strict.json \
  --output-model models/layout_model.pkl \
  --output-report models/training_report.json
```

## Infer one image

```bash
python infer_layout_classifier.py \
  --model models/layout_model.pkl \
  --input /path/to/pattern_1168_further_crop_1024.png \
  --metadata /path/to/the_crater_layout_slots_further_crop_bl_strict.json
```

`--metadata` is accepted for CLI compatibility, but the trained model already embeds the metadata-derived gallery.

## Evaluate a folder

```bash
python evaluate_layout_classifier.py \
  --images /path/to/tmp_the_crater_both_crops_L256_R256_T256_B256/further_crop_1024 \
  --metadata /path/to/the_crater_layout_slots_further_crop_bl_strict.json \
  --model models/layout_model.pkl \
  --output models/evaluation_report.json
```

## Output format

Inference returns JSON like:

```json
{
  "best_patternId": 1199,
  "top5": [
    {"patternId": 1199, "score": 3.12, "probability": 0.71},
    {"patternId": 1198, "score": 1.05, "probability": 0.11},
    {"patternId": 1197, "score": 0.88, "probability": 0.07},
    {"patternId": 1139, "score": 0.76, "probability": 0.05},
    {"patternId": 1138, "score": 0.61, "probability": 0.03}
  ],
  "margin": 2.07,
  "confidence": 0.71,
  "unknown": false,
  "decided_patternId": 1199,
  "latency_ms": 245.0,
  "unknown_margin_threshold": 2.39
}
```
