# Limveld layout classifier

Predicts Limveld `patternId` from one `1024x1024` further-crop map image.

## Files

- `train_layout_classifier.py` - builds the gallery matcher model and training report.
- `infer_layout_classifier.py` - predicts best `patternId`, top-5, confidence, margin, and optional `unknown`.
- `evaluate_layout_classifier.py` - evaluates a folder of images and writes a report JSON.
- `layout_classifier_lib.py` - shared feature extraction, metadata parsing, scoring, calibration, and reporting code.
- `models/layout_model.pkl` - saved trained model artifact.
- `models/training_report.json` - training summary.
- `models/evaluation_report.json` - evaluation report placeholder; run local evaluation command to regenerate complete metrics.
- `models/sample_infer.json` - sample one-image inference output.

## Dependencies

```bash
python -m pip install numpy opencv-python scikit-learn
```

## Train

```bash
python train_layout_classifier.py \
  --images /path/to/tmp_limveld_both_crops_L256_R256_T256_B256/further_crop_1024 \
  --metadata /path/to/limveld_layout_slots_further_crop_bl_strict.json \
  --output models/layout_model.pkl \
  --report models/training_report.json
```

## Infer one image

```bash
python infer_layout_classifier.py \
  --model models/layout_model.pkl \
  --input /path/to/pattern_0000_further_crop_1024.png \
  --metadata /path/to/limveld_layout_slots_further_crop_bl_strict.json
```

## Evaluate a folder

```bash
python evaluate_layout_classifier.py \
  --images /path/to/tmp_limveld_both_crops_L256_R256_T256_B256/further_crop_1024 \
  --metadata /path/to/limveld_layout_slots_further_crop_bl_strict.json \
  --model models/layout_model.pkl \
  --output models/evaluation_report.json
```

## Coordinate contract used

- image size: `1024x1024`
- metadata origin: bottom-left
- image array origin: top-left
- conversion:
  - `x_px = round(x_further_px_bl)`
  - `y_px_top_left = (1024 - 1) - round(y_further_px_bl)`

`*_px_bl` fields are used first. `*_norm_bl` is used only as fallback.
