# Rotted Woods Layout Classifier

This project predicts Rotted Woods `patternId` from one `1024x1024` further-crop image.

## Files
- `train_layout_classifier.py`: builds the gallery model and calibration.
- `infer_layout_classifier.py`: predicts one image.
- `evaluate_layout_classifier.py`: evaluates a folder of images.
- `layout_classifier_lib.py`: shared feature extraction, scoring, and evaluation logic.
- `models/layout_model.pkl`: saved model artifact.
- `models/training_report.json`: training/build summary.
- `models/evaluation_report.json`: evaluation report.

## Coordinate contract
- Image size: `1024x1024`
- Metadata origin: bottom-left
- Image origin: top-left
- Conversion used:
  - `x_px = round(x_further_px_bl)`
  - `y_px_top_left = 1023 - round(y_further_px_bl)`
- Primary fields: `x_further_px_bl`, `y_further_px_bl`
- Fallback fields: `x_further_norm_bl`, `y_further_norm_bl`

## Dependencies
```bash
python -m pip install numpy opencv-python scikit-learn
```

## Train
```bash
python train_layout_classifier.py \
  --images /path/to/tmp_rotted_woods_both_crops_L256_R256_T256_B256/further_crop_1024 \
  --metadata /path/to/rotted_woods_layout_slots_further_crop_bl_strict.json \
  --output models/layout_model.pkl \
  --report models/training_report.json
```

## Infer one image
```bash
python infer_layout_classifier.py \
  --model models/layout_model.pkl \
  --input /path/to/pattern_1172_further_crop_1024.png \
  --metadata /path/to/rotted_woods_layout_slots_further_crop_bl_strict.json
```

## Evaluate a folder
```bash
python evaluate_layout_classifier.py \
  --images /path/to/tmp_rotted_woods_both_crops_L256_R256_T256_B256/further_crop_1024 \
  --metadata /path/to/rotted_woods_layout_slots_further_crop_bl_strict.json \
  --model models/layout_model.pkl \
  --output models/evaluation_report.json
```
