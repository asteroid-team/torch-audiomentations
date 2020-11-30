import types
import unittest

from torch_audiomentations import PolarityInversion


class TestBaseClass(unittest.TestCase):
    def test_parameters(self):
        # Test that we can access the parameters function of nn.Module
        augment = PolarityInversion(p=1.0)
        params = augment.parameters()
        assert isinstance(params, types.GeneratorType)
