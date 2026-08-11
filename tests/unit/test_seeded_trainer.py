"""Unit test for the project's RNG seeding utility, in isolation -- deliberately imported
from vesuvius_surface.training.seeding directly (not via nnUNetTrainerSeeded or the
trainers/ package), which has zero nnunetv2/batchgeneratorsv2 dependency, so this test runs
in CI without needing the full nnU-Net training stack installed. See that module's own
docstring for why the split exists.

This is the only RNG-seeding fix anywhere in the project: stock nnU-Net sets no seed anywhere
(verified from nnUNetTrainer.__init__ and the nnUNetv2_train CLI arg list directly, not
assumed). Every from-scratch baseline result in this project traces back to this
function actually working.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from vesuvius_surface.training.seeding import seed_everything


class TestSeedEverything:
    def test_numpy_reproducible_after_seeding(self):
        seed_everything(42)
        a = np.random.rand(10)
        seed_everything(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_python_random_reproducible_after_seeding(self):
        seed_everything(42)
        a = [random.random() for _ in range(10)]
        seed_everything(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_torch_reproducible_after_seeding(self):
        seed_everything(42)
        a = torch.rand(10)
        seed_everything(42)
        b = torch.rand(10)
        torch.testing.assert_close(a, b)

    def test_different_seeds_produce_different_sequences(self):
        seed_everything(1)
        a = np.random.rand(10)
        seed_everything(2)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
