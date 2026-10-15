# -*- mode: python ; coding: utf-8 -*-
import importlib.util
from pathlib import Path

block_cipher = None

# Build flavor switch (single source in build_flavor.py):
# - "DEMO" -> NightreignRecognizer_DEMO
# - "v41"  -> NightreignRecognizer_v41
_spec_dir = Path(globals().get("__file__", "nightreign_v41.spec")).resolve().parent
_build_flavor_path = _spec_dir / "build_flavor.py"
if not _build_flavor_path.exists():
    raise FileNotFoundError(f"build_flavor.py not found at: {_build_flavor_path}")

_bf_spec = importlib.util.spec_from_file_location("nightreign_build_flavor_spec", str(_build_flavor_path))
if _bf_spec is None or _bf_spec.loader is None:
    raise RuntimeError(f"Could not load spec for: {_build_flavor_path}")
_bf_mod = importlib.util.module_from_spec(_bf_spec)
_bf_spec.loader.exec_module(_bf_mod)
SHARED_BUILD_FLAVOR = str(getattr(_bf_mod, "BUILD_FLAVOR", "v41"))
BUILD_FLAVOR = str(SHARED_BUILD_FLAVOR).strip() or "v41"
APP_EXE_NAME = "NightreignRecognizer_DEMO" if str(BUILD_FLAVOR).strip().upper() == "DEMO" else f"NightreignRecognizer_{BUILD_FLAVOR}"

a = Analysis(
    ['nightreign_map_recognizer_gui_ai_v39_completion_toast_overlay.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('POI_icons', 'POI_icons'),
        ('maps', 'maps'),
        ('great_hollow_pattern_sets', 'great_hollow_pattern_sets'),
        ('nightreign_sword_small_transparent_48.ico', '.'),
        ('nightreign_sword_large_transparent_256.ico', '.'),
        ('LotResultSmallBaseAndSpot_grouped_by_patternId_14.json', '.'),
        ('LotResultMapPatternFlag.csv', '.'),
        ('recognize_nightreign_map_proto_v12_debugplus_radiusscale_fix.py', '.'),
        ('nightreign_geom_proto_v4_slots.npz', '.'),
        ('nightreign_best_tuned_config_multi.json', '.'),
        ('nightreign_best_tuned_config.json', '.'),
        ('Train_Earth_Selector_v2/infer_earth_selector_v2.py', 'Train_Earth_Selector_v2'),
        ('Train_Earth_Selector_v2/earth_selector_v2_model_all520_plus_new.pkl', 'Train_Earth_Selector_v2'),
        ('Testing/run_parallel_panel_classifiers.py', 'Testing'),
        ('Testing/slot_2/layout_classifier_lib.py', 'Testing/slot_2'),
        ('Testing/slot_2_fixed_geometry_panel_bases260/slot2_status_cnn_gh_variations_retrain_latest.pt', 'Testing/slot_2_fixed_geometry_panel_bases260'),
        ('Testing/slot_2_fixed_geometry_panel_bases260/slot2_status_cnn_offset50_v1.pt', 'Testing/slot_2_fixed_geometry_panel_bases260'),
        ('Testing/slot_2/slot2_status_cnn.pt', 'Testing/slot_2'),
        ('Testing/slot_4/layout_classifier_lib.py', 'Testing/slot_4'),
        ('Testing/slot_4_fixed_geometry_panel_bases260/separated_occupied_empty/slot4_occupied_empty_from_panel_fixed_v1.pt', 'Testing/slot_4_fixed_geometry_panel_bases260/separated_occupied_empty'),
        ('Testing/slot_4_fixed_geometry_panel_bases260/slot4_church_vs_sorcerer_bases260/slot4_church_vs_sorcerer_v1.pt', 'Testing/slot_4_fixed_geometry_panel_bases260/slot4_church_vs_sorcerer_bases260'),
        ('Testing/slot_4_plus_real_heavy/slot4_status_cnn_plus_real_heavy.pt', 'Testing/slot_4_plus_real_heavy'),
        ('Testing/slot_4/slot4_status_cnn.pt', 'Testing/slot_4'),
        ('Testing/slot_5/layout_classifier_lib.py', 'Testing/slot_5'),
        ('Testing/slot5_authoritative_binary_panel_v1/slot5_occupied_empty_authoritative_v1.pt', 'Testing/slot5_authoritative_binary_panel_v1'),
        ('Testing/slot5_church_vs_sorcerer_panel_v2_userlabels/slot5_church_vs_sorcerer_v2.pt', 'Testing/slot5_church_vs_sorcerer_panel_v2_userlabels'),
        ('Testing/slot_5_plus_real_heavy/slot5_status_cnn_plus_real_heavy.pt', 'Testing/slot_5_plus_real_heavy'),
        ('Testing/slot_5_userlabels_panel6_iso/slot5_status_cnn_userlabels_panel6_iso.pt', 'Testing/slot_5_userlabels_panel6_iso'),
        ('Testing/slot_5_userlabels_panel6_iso/slot5_status_cnn_userlabels_panel6_iso_ghslot5_3000x3_20260422.pt', 'Testing/slot_5_userlabels_panel6_iso'),
        ('Testing/slot_7/layout_classifier_lib.py', 'Testing/slot_7'),
        ('Testing/slot_7_fixed_geometry_panel_bases260/separated_occupied_empty/slot7_occupied_empty_from_panel_fixed_v1.pt', 'Testing/slot_7_fixed_geometry_panel_bases260/separated_occupied_empty'),
        ('Testing/slot_7_fixed_geometry_panel_bases260/slot7_church_vs_sorcerer_bases260/slot7_church_vs_sorcerer_v1.pt', 'Testing/slot_7_fixed_geometry_panel_bases260/slot7_church_vs_sorcerer_bases260'),
        ('Testing/slot_7_plus_real_heavy/slot7_status_cnn_plus_real_heavy.pt', 'Testing/slot_7_plus_real_heavy'),
        ('Testing/slot_6/layout_classifier_lib.py', 'Testing/slot_6'),
        ('Testing/slot_6/slot6_status_cnn.pt', 'Testing/slot_6'),
        ('Testing/palace/layout_classifier_lib.py', 'Testing/palace'),
        ('Testing/palace/palace_status_cnn_gh_variations_retrain_latest.pt', 'Testing/palace'),
        ('Testing/palace/palace_status_cnn_retrain_latest.pt', 'Testing/palace'),
        ('Testing/palace/palace_status_cnn.pt', 'Testing/palace'),
        ('Testing/fortress/layout_classifier_lib.py', 'Testing/fortress'),
        ('Testing/fortress/fortress_status_cnn_gh_variations_retrain_latest.pt', 'Testing/fortress'),
        ('Testing/fortress/fortress_status_cnn_retrain_latest.pt', 'Testing/fortress'),
        ('Testing/fortress/fortress_status_cnn.pt', 'Testing/fortress'),
        # Layout-infer assets used by v41 defaults:
        # - Great Hollow: layout_model.pkl
        # - non-GH: NON_GH_LAYOUT_MODEL_VARIANT="v16" -> layout_model_v16.pkl
        ('tmp_great_hollow_both_crops_asym_L175_R170_T180_B180_Patched_multi_image/further_crop_asymmetric/layout_classifier_lib.py', 'tmp_great_hollow_both_crops_asym_L175_R170_T180_B180_Patched_multi_image/further_crop_asymmetric'),
        ('tmp_great_hollow_both_crops_asym_L175_R170_T180_B180_Patched_multi_image/further_crop_asymmetric/layout_model.pkl', 'tmp_great_hollow_both_crops_asym_L175_R170_T180_B180_Patched_multi_image/further_crop_asymmetric'),
        ('tmp_limveld_both_crops_L256_R256_T256_B256/further_crop_1024/layout_classifier_lib.py', 'tmp_limveld_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_limveld_both_crops_L256_R256_T256_B256/further_crop_1024/layout_model_v16.pkl', 'tmp_limveld_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_mountaintop_both_crops_L256_R256_T256_B256/further_crop_1024/layout_classifier_lib.py', 'tmp_mountaintop_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_mountaintop_both_crops_L256_R256_T256_B256/further_crop_1024/layout_model_v16.pkl', 'tmp_mountaintop_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_noklateo_both_crops_L256_R256_T256_B256/further_crop_1024/layout_classifier_lib.py', 'tmp_noklateo_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_noklateo_both_crops_L256_R256_T256_B256/further_crop_1024/layout_model_v16.pkl', 'tmp_noklateo_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_rotted_woods_both_crops_L256_R256_T256_B256/further_crop_1024/layout_classifier_lib.py', 'tmp_rotted_woods_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_rotted_woods_both_crops_L256_R256_T256_B256/further_crop_1024/layout_model_v16.pkl', 'tmp_rotted_woods_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_the_crater_both_crops_L256_R256_T256_B256/further_crop_1024/layout_classifier_lib.py', 'tmp_the_crater_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_the_crater_both_crops_L256_R256_T256_B256/further_crop_1024/layout_model_v16.pkl', 'tmp_the_crater_both_crops_L256_R256_T256_B256/further_crop_1024'),
        ('tmp_palace_magic_fire_test/palace_magic_fire_test_model.pt', 'tmp_palace_magic_fire_test'),
        ('tmp_fortress_rot_holy_test/fortress_rot_holy_test_model.pt', 'tmp_fortress_rot_holy_test'),
        ('tmp_ruins_poison_bleed_test/ruins_poison_bleed_test_model.pt', 'tmp_ruins_poison_bleed_test'),
        ('tmp_gc_sleep_lightning_test/gc_sleep_lightning_test_model.pt', 'tmp_gc_sleep_lightning_test'),
        ('tmp_gc_frost_holy_test_newbox/gc_frost_holy_test_model_newbox.pt', 'tmp_gc_frost_holy_test_newbox'),
        ('slot_ai_test_output_dlc_no_text_nobalance/slot_patch_model_v1.pt', 'slot_ai_test_output_dlc_no_text_nobalance'),
        ('slot_ai_test_output_dlc_no_text/slot_patch_model_v1.pt', 'slot_ai_test_output_dlc_no_text'),
        ('slot_ai_test_output/slot_patch_model_v1.pt', 'slot_ai_test_output'),
        ('evergaol_ai_test_output_no_text/evergaol_patch_model_v1.pt', 'evergaol_ai_test_output_no_text'),
        ('earth_ai_test_output_no_text_prepped/earth_panel_model_v1.pt', 'earth_ai_test_output_no_text_prepped'),
        ('earth_ai_test_output_no_text/earth_panel_model_v1.pt', 'earth_ai_test_output_no_text'),
        ('LotResultSmallBaseAndSpot_grouped_by_patternId_14.json', '.'),
    ],
    hiddenimports=[
        'sklearn',
        'torch',
        'torchvision',
        'torchaudio',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'cv2',
        'numpy',
        # Required while unpickling the scikit-learn earth/layout models.
        # These models are loaded from data files, so PyInstaller cannot infer
        # this compatibility module through normal import analysis.
        'scipy._lib.array_api_compat.numpy.fft',
        'mss',
        'ctypes',
        'ctypes.wintypes',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hook_torch_dlls.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name=APP_EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='nightreign_sword_small_transparent_48.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_EXE_NAME,
)
