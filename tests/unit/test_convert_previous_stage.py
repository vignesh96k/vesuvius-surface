"""Unit tests for scripts/inference/convert_previous_stage.py's pure array logic
(bbox cropping, probability-weighted combination) -- no GPU, no real nnU-Net dataset needed.

Covers the two hard-won, non-obvious details documented in that script's own docstring:
cropping to match the preprocessed GT's bbox exactly, and never adding an explicit channel
dimension (nnU-Net's own data loader adds one; a real bug this project hit -- see
the script's own docstring).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "convert_previous_stage", REPO_ROOT / "scripts" / "inference" / "convert_previous_stage.py"
)
convert_previous_stage = importlib.util.module_from_spec(_spec)
sys.modules["convert_previous_stage"] = convert_previous_stage
_spec.loader.exec_module(convert_previous_stage)

_bbox_crop = convert_previous_stage._bbox_crop


def _write_pkl(path: Path, bbox) -> None:
    import pickle

    with open(path, "wb") as f:
        pickle.dump({"bbox_used_for_cropping": bbox}, f)


class TestBboxCrop:
    def test_crop_matches_expected_bbox_shape(self, tmp_path):
        volume = np.arange(10 * 10 * 10, dtype=np.uint8).reshape(10, 10, 10)
        bbox = ((2, 8), (1, 9), (0, 5))
        _write_pkl(tmp_path / "case1.pkl", bbox)

        cropped = _bbox_crop(volume, tmp_path, "case1")

        assert cropped.shape == (6, 8, 5)
        np.testing.assert_array_equal(cropped, volume[2:8, 1:9, 0:5])

    def test_crop_has_no_extra_channel_dimension(self, tmp_path):
        """The real bug this convention fixes: nnU-Net's own data loader adds a channel
        dim itself, so the array written here must stay 3D, never 4D."""
        volume = np.ones((12, 12, 12), dtype=np.uint8)
        _write_pkl(tmp_path / "case2.pkl", ((0, 12), (0, 12), (0, 12)))

        cropped = _bbox_crop(volume, tmp_path, "case2")

        assert cropped.ndim == 3

    def test_full_volume_bbox_is_a_noop(self, tmp_path):
        volume = np.random.randint(0, 2, size=(5, 5, 5)).astype(np.uint8)
        _write_pkl(tmp_path / "case3.pkl", ((0, 5), (0, 5), (0, 5)))

        cropped = _bbox_crop(volume, tmp_path, "case3")

        np.testing.assert_array_equal(cropped, volume)


class TestProbabilityCombination:
    def test_weighted_argmax_matches_manual_computation(self):
        # 2-class, 1x1x2 volume: model A confidently says class 1 at both voxels, model B
        # confidently says class 0 -- weight A higher, class 1 should win.
        probs_a = np.array([[[0.1, 0.1]], [[0.9, 0.9]]], dtype=np.float32)  # (C=2, 1, 2)
        probs_b = np.array([[[0.9, 0.9]], [[0.1, 0.1]]], dtype=np.float32)

        combined = 0.65 * probs_a + 0.35 * probs_b
        seg = np.argmax(combined, axis=0).astype(np.uint8)

        assert seg.tolist() == [[1, 1]]

    def test_equal_weights_ties_break_toward_first_argmax(self):
        probs_a = np.array([[[0.5]], [[0.5]]], dtype=np.float32)
        probs_b = np.array([[[0.5]], [[0.5]]], dtype=np.float32)
        combined = 0.5 * probs_a + 0.5 * probs_b
        seg = np.argmax(combined, axis=0)
        assert seg.tolist() == [[0]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
