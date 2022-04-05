import unittest
import torch

from torch_audiomentations import TimeInversion


class TestTimeInversion(unittest.TestCase):
    def setUp(self):
        self.augment = TimeInversion(p=1.0, output_type="dict")
        self.samples = torch.arange(1, 100, 1).type(torch.FloatTensor)
        self.expected_samples = torch.arange(99, 0, -1).type(torch.FloatTensor)

    def test_single_channel(self):
        samples = self.samples.unsqueeze(0).unsqueeze(0)  # (B, C, T): (1, 1, 100)
        processed_samples = self.augment(samples=samples, sample_rate=16000).samples

        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertTrue(
            torch.equal(
                processed_samples, self.expected_samples.unsqueeze(0).unsqueeze(0)
            )
        )

    def test_multi_channel(self):
        samples = torch.stack([self.samples, self.samples], dim=0).unsqueeze(
            0
        )  # (B, C, T): (1, 2, 100)
        processed_samples = self.augment(samples=samples, sample_rate=16000).samples

        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertTrue(
            torch.equal(processed_samples[:, 0], self.expected_samples.unsqueeze(0))
        )
        self.assertTrue(
            torch.equal(processed_samples[:, 1], self.expected_samples.unsqueeze(0))
        )
