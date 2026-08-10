"""Local functional test of the submission notebook's logic (cells 2 onward -- skips the
Kaggle-only offline pip install cell, since nnunetv2==2.8.1 is already installed here).
Uses local paths instead of /kaggle/* mounts, otherwise identical logic to the notebook."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tifffile
from tqdm.auto import tqdm

# ---- local stand-ins for Kaggle paths (env vars, see docs/data.md / docs/checkpoints.md) ----
INPUT_DIR = Path(os.environ["VESUVIUS_DATA_ROOT"])
CHECKPOINT_DIR = Path(os.environ["VESUVIUS_TEST_CHECKPOINT_DIR"])
WORKING_DIR = Path("/tmp/vesuvius_submission_notebook_test")
OUTPUT_DIR = Path("/tmp/vesuvius_submission_notebook_test_output")

NNUNET_RAW = WORKING_DIR / "nnUNet_data" / "nnUNet_raw"
NNUNET_PREPROCESSED = WORKING_DIR / "nnUNet_data" / "nnUNet_preprocessed"
NNUNET_RESULTS = WORKING_DIR / "nnUNet_results"

DATASET_ID = 100
DATASET_NAME = "Dataset100_VesuviusSurface"
CONFIGURATION = "3d_lowres"
PLANS_NAME = "nnUNetResEncUNetMPlans"
TRAINER = "nnUNetTrainer_5epochs"
FOLD = "all"
MODEL_DIR_NAME = f"{TRAINER}__{PLANS_NAME}__{CONFIGURATION}"

TEST_INPUT_DIR = WORKING_DIR / "test_input"
PREDICTIONS_DIR = WORKING_DIR / "predictions"
PREDICTIONS_TIFF_DIR = OUTPUT_DIR / "predictions_tiff"
SUBMISSION_ZIP = OUTPUT_DIR / "submission.zip"


def setup_environment():
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    os.environ["nnUNet_compile"] = "true"
    print(f"nnUNet_raw: {NNUNET_RAW}")
    print(f"nnUNet_results: {NNUNET_RESULTS}")


print("=== [1] setup_environment ===")
setup_environment()

print("\n=== [2] stage checkpoint ===")
dst_model_dir = NNUNET_RESULTS / DATASET_NAME / MODEL_DIR_NAME
src_model_dir = CHECKPOINT_DIR / MODEL_DIR_NAME
shutil.copytree(src_model_dir, dst_model_dir, dirs_exist_ok=True)
checkpoint_path = dst_model_dir / "fold_all" / "checkpoint_final.pth"
assert checkpoint_path.exists(), f"Missing checkpoint at {checkpoint_path}"
print("Staged checkpoint:", checkpoint_path, f"({checkpoint_path.stat().st_size / 1e6:.1f} MB)")


def create_spacing_json(output_path: Path, shape: tuple, spacing: tuple = (1.0, 1.0, 1.0)):
    with open(output_path, "w") as f:
        json.dump({"spacing": list(spacing)}, f)


def prepare_single_case(src_path: Path, dest_path: Path, json_path: Path, use_symlinks: bool = True) -> bool:
    try:
        with tifffile.TiffFile(src_path) as tif:
            shape = tif.pages[0].shape if len(tif.pages) == 1 else (len(tif.pages), *tif.pages[0].shape)
        if use_symlinks:
            if not dest_path.exists():
                dest_path.symlink_to(src_path.resolve())
        else:
            shutil.copy2(src_path, dest_path)
        create_spacing_json(json_path, shape)
        return True
    except Exception as e:
        print(f"Error processing {src_path.name}: {e}")
        return False


def prepare_test_data(input_dir: Path, output_dir: Path, use_symlinks: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir = input_dir / "test_images"
    test_files = sorted(test_images_dir.glob("*.tif"))
    print(f"Found {len(test_files)} test cases")
    for img_path in tqdm(test_files, desc="Preparing test data"):
        case_id = img_path.stem
        prepare_single_case(img_path, output_dir / f"{case_id}_0000.tif", output_dir / f"{case_id}_0000.json", use_symlinks)
    return output_dir


print("\n=== [3] prepare_test_data ===")
prepare_test_data(INPUT_DIR, TEST_INPUT_DIR)


def _run_command(cmd: str, name: str = "Command", tail_lines: int = 30, timeout: Optional[int] = None) -> bool:
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"{name} TIMEOUT after {timeout}s!")
        return False
    if result.returncode != 0:
        print(f"{name} FAILED!")
        print(f"STDERR:\n{result.stderr[-3000:]}")
        return False
    print(f"{name} complete!")
    if result.stdout.strip():
        lines = result.stdout.strip().split("\n")
        print("\n".join(lines[-tail_lines:]))
    return True


def run_inference(input_dir, output_dir, dataset_id, config, fold, plans, trainer,
                   save_probabilities=True, num_processes_preprocessing=2, num_processes_segmentation=2):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"nnUNetv2_predict -d {dataset_id:03d} -c {config} -f {fold}"
    cmd += f" -i {input_dir} -o {output_dir} -p {plans} -tr {trainer}"
    cmd += f" -npp {num_processes_preprocessing} -nps {num_processes_segmentation}"
    cmd += " --verbose"
    if save_probabilities:
        cmd += " --save_probabilities"
    return _run_command(cmd, "Inference")


print("\n=== [4] run_inference ===")
ok = run_inference(
    input_dir=TEST_INPUT_DIR, output_dir=PREDICTIONS_DIR,
    dataset_id=DATASET_ID, config=CONFIGURATION, fold=FOLD,
    plans=PLANS_NAME, trainer=TRAINER,
)
assert ok, "Inference failed"


def load_probabilities(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    return data["probabilities"]


def predictions_to_tiff(pred_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files = list(pred_dir.glob("*.npz"))
    tif_files = list(pred_dir.glob("*.tif"))
    if npz_files:
        print(f"Converting {len(npz_files)} NPZ probability files to TIFF...")
        for npz_path in tqdm(npz_files, desc="Converting to TIFF"):
            case_id = npz_path.stem
            probs = load_probabilities(npz_path)
            pred = np.argmax(probs, axis=0).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    elif tif_files:
        for tif_path in tqdm(tif_files, desc="Copying TIFF"):
            case_id = tif_path.stem
            pred = tifffile.imread(str(tif_path)).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    else:
        print(f"WARNING: No prediction files found in {pred_dir}")


print("\n=== [5] predictions_to_tiff ===")
predictions_to_tiff(PREDICTIONS_DIR, PREDICTIONS_TIFF_DIR)

print("\n=== [6] dimension/dtype sanity check ===")
train_labels_dir = INPUT_DIR / "train_labels"
sample_train_label = next(train_labels_dir.glob("*.tif"))
expected_dtype = tifffile.imread(str(sample_train_label)).dtype
print(f"Expected dtype (from a train label): {expected_dtype}")

all_ok = True
for pred_path in sorted(PREDICTIONS_TIFF_DIR.glob("*.tif")):
    case_id = pred_path.stem
    src_path = INPUT_DIR / "test_images" / f"{case_id}.tif"
    pred_arr = tifffile.imread(str(pred_path))
    src_arr = tifffile.imread(str(src_path))
    shape_ok = pred_arr.shape == src_arr.shape
    dtype_ok = pred_arr.dtype == expected_dtype
    all_ok &= shape_ok and dtype_ok
    print(f"{case_id}: pred shape={pred_arr.shape} dtype={pred_arr.dtype} | src shape={src_arr.shape} | shape_ok={shape_ok} dtype_ok={dtype_ok}")
assert all_ok, "Dimension/dtype mismatch"
print("All predictions match source dimensions and expected dtype.")


def generate_submission(predictions_tiff_dir: Path, output_zip: Path, delete_after_zip: bool = True):
    tiff_files = sorted(predictions_tiff_dir.glob("*.tif"))
    if not tiff_files:
        print(f"No TIFF files found in {predictions_tiff_dir}")
        return None
    print(f"Creating submission ZIP with {len(tiff_files)} files...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for tiff_path in tqdm(tiff_files, desc="Zipping predictions"):
            zipf.write(tiff_path, tiff_path.name)
            if delete_after_zip:
                tiff_path.unlink()
    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Submission saved: {output_zip} ({zip_size_mb:.1f} MB)")
    return output_zip


print("\n=== [7] generate_submission ===")
submission_path = generate_submission(PREDICTIONS_TIFF_DIR, SUBMISSION_ZIP)
assert submission_path is not None and submission_path.exists()
print("\nDONE:", submission_path)

# Verify zip contents
with zipfile.ZipFile(submission_path) as zf:
    names = zf.namelist()
    print(f"\nZip contains {len(names)} file(s): {names}")
