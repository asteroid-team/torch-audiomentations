from copy import deepcopy

import pytest
import torch
from torch.optim import SGD

from tests.utils import TEST_FIXTURES_DIR
from torch_audiomentations import (
    ApplyBackgroundNoise,
    # ApplyImpulseResponse,
    Gain,
    PeakNormalization,
    PolarityInversion,
    Compose,
    Shift,
)

BG_NOISE_PATH = TEST_FIXTURES_DIR / "bg"
IR_PATH = TEST_FIXTURES_DIR / "ir"


@pytest.mark.parametrize(
    "augment",
    [
        # Differentiable transforms:
        ApplyBackgroundNoise(BG_NOISE_PATH, 20, p=1.0),
        # ApplyImpulseResponse(IR_PATH, p=1.0),
        Compose(
            transforms=[
                Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=1.0),
                PolarityInversion(p=1.0),
            ]
        ),
        Gain(min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0),
        PolarityInversion(p=1.0),
        Shift(p=1.0),
        # Non-differentiable transforms:
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
        # [torch.DoubleTensor [1, 5]], which is output 0 of IndexBackward, is at version 1; expected version 0 instead.
        # Hint: enable anomaly detection to find the operation that failed to compute its gradient,
        # with torch.autograd.set_detect_anomaly(True).
        pytest.param(
            PeakNormalization(p=1.0), marks=pytest.mark.skip("Not differentiable")
        ),
    ],
)
def test_transform_is_differentiable(augment):
    sample_rate = 16000
    # Note: using float64 dtype to be compatible with ApplyBackgroundNoise fixtures
    samples = torch.tensor(
        [[1.0, 0.5, -0.25, -0.125, 0.0]], dtype=torch.float64
    ).unsqueeze(1)
    samples_cpy = deepcopy(samples)

    # We are going to convert the input tensor to a nn.Parameter so that we can
    # track the gradients with respect to it. We'll "optimize" the input signal
    # to be closer to that after the augmentation to test differentiability
    # of the transform. If the signal got changed in any way, and the test
    # didn't crash, it means it works.
    samples = torch.nn.Parameter(samples)
    optim = SGD([samples], lr=1.0)
    for i in range(10):
        optim.zero_grad()
        transformed = augment(samples=samples, sample_rate=sample_rate)
        # Compute mean absolute error
        loss = torch.mean(torch.abs(samples - transformed))
        loss.backward()
        optim.step()

    assert (samples != samples_cpy).any()
