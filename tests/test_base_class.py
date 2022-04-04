import types
import unittest

import pytest
import torch

from torch_audiomentations import PolarityInversion


class TestBaseClass(unittest.TestCase):
    def test_parameters(self):
        # Test that we can access the parameters function of nn.Module
        augment = PolarityInversion(p=1.0, output_type="dict")
        params = augment.parameters()
        assert isinstance(params, types.GeneratorType)

    def test_ndim_check(self):
        augment = PolarityInversion(p=1.0, output_type="dict")
        # 1D tensor not allowed
        with pytest.raises(RuntimeError):
            augment(torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32))
        # 2D tensor not allowed
        with pytest.raises(RuntimeError):
            augment(torch.tensor([[1.0, 0.5, 0.25, 0.125]], dtype=torch.float32))
        # 4D tensor not allowed
        with pytest.raises(RuntimeError):
            augment(torch.tensor([[[[1.0, 0.5, 0.25, 0.125]]]], dtype=torch.float32))
