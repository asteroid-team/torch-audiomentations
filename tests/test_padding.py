import unittest
import numpy as np
import torch
from numpy.testing import assert_almost_equal

from torch_audiomentations.augmentations.padding import Padding


class TestPadding(unittest.TestCase):

    def test_PaddingEnd(self):

        audio_samples = torch.rand(size=(2,2,32000), dtype=torch.float32)
        augment = Padding(min_fraction=0.2, max_fraction=0.5, pad_section="end",p=1.0,output_type='dict')
        padded_samples = augment(audio_samples).samples


        self.assertEqual(audio_samples.shape,padded_samples.shape)
        assert_almost_equal(padded_samples[...,-6400:].numpy(),np.zeros((2,2,6400)))

    
    def test_PaddingStart(self):

        audio_samples = torch.rand(size=(2,2,32000), dtype=torch.float32)
        augment = Padding(min_fraction=0.2, max_fraction=0.5, pad_section="start",p=1.0, output_type='dict')
        padded_samples = augment(audio_samples).samples


        self.assertEqual(audio_samples.shape,padded_samples.shape)
        assert_almost_equal(padded_samples[...,:6400].numpy(),np.zeros((2,2,6400)))



