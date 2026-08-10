"""Unit test for nnUNetTrainerSeeded's seeding utility, in isolation -- does not instantiate
the full nnU-Net trainer (needs plans/dataset_json/GPU context we don't have in CI), just the
_seed_everything() function itself.

This is the only RNG-seeding fix anywhere in the project: stock nnU-Net sets no seed anywhere
(verified from nnUNetTrainer.__init__ and the nnUNetv2_train CLI arg list directly, not
assumed -- see the trainer's own module docstring). Every from-scratch baseline result in
experiment_summary.md traces back to this function actually working.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from vesuvius_surface.training.trainers.nnUNetTrainerSeeded import _seed_everything


class TestSeedEverything:
    def test_numpy_reproducible_after_seeding(self):
        _seed_everything(42)
        a = np.random.rand(10)
        _seed_everything(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_python_random_reproducible_after_seeding(self):
        _seed_everything(42)
        a = [random.random() for _ in range(10)]
        _seed_everything(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_torch_reproducible_after_seeding(self):
        _seed_everything(42)
        a = torch.rand(10)
        _seed_everything(42)
        b = torch.rand(10)
        torch.testing.assert_close(a, b)

    def test_different_seeds_produce_different_sequences(self):
        _seed_everything(1)
        a = np.random.rand(10)
        _seed_everything(2)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
