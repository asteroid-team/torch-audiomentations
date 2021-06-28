import os

import pytest
import torch

from torch_audiomentations import ApplyImpulseResponse
from torch_audiomentations.utils.file import load_audio
from .utils import TEST_FIXTURES_DIR


@pytest.fixture
def sample_rate():
    yield 16000


@pytest.fixture
def input_audio(sample_rate):
    return (
        torch.from_numpy(
            load_audio(
                os.path.join(TEST_FIXTURES_DIR, "acoustic_guitar_0.wav"),
                sample_rate=sample_rate,
            )
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )


@pytest.fixture
def input_audios(input_audio):
    batch_size = 32
    return torch.cat([input_audio] * batch_size, dim=0)


@pytest.fixture
def ir_path():
    yield os.path.join(TEST_FIXTURES_DIR, "ir")


@pytest.fixture()
def ir_transform(ir_path, sample_rate):
    return ApplyImpulseResponse(ir_path, p=1.0, sample_rate=sample_rate)


@pytest.fixture()
def ir_transform_no_guarantee(ir_path, sample_rate):
    return ApplyImpulseResponse(ir_path, p=0.0, sample_rate=sample_rate)


def test_impulse_response_guaranteed_with_single_tensor_input(ir_transform, input_audio):
    mixed_input = ir_transform(input_audio)
    assert mixed_input.shape == input_audio.shape
    assert not torch.equal(mixed_input, input_audio)


@pytest.mark.parametrize(
    "compensate_for_propagation_delay",
    [False, True],
)
def test_impulse_response_guaranteed_with_batched_tensor_input(
    ir_path, sample_rate, input_audios, compensate_for_propagation_delay
):
    mixed_inputs = ApplyImpulseResponse(
        ir_path,
        compensate_for_propagation_delay=compensate_for_propagation_delay,
        p=1.0,
        sample_rate=sample_rate,
    )(input_audios)
    assert mixed_inputs.shape == input_audios.shape
    assert not torch.equal(mixed_inputs, input_audios)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_impulse_response_guaranteed_with_batched_cuda_tensor_input(
    input_audios, ir_transform
):
    input_audio_cuda = input_audios.cuda()
    mixed_inputs = ir_transform(input_audio_cuda)
    assert not torch.equal(mixed_inputs, input_audio_cuda)
    assert mixed_inputs.shape == input_audio_cuda.shape
    assert mixed_inputs.dtype == input_audio_cuda.dtype
    assert mixed_inputs.device == input_audio_cuda.device


def test_impulse_response_no_guarantee_with_single_tensor_input(
    input_audio, ir_transform_no_guarantee
):
    mixed_input = ir_transform_no_guarantee(input_audio)
    assert mixed_input.shape == input_audio.shape


def test_impulse_response_no_guarantee_with_batched_tensor_input(
    input_audios, ir_transform_no_guarantee
):
    mixed_inputs = ir_transform_no_guarantee(input_audios)
    assert mixed_inputs.shape == input_audios.shape


def test_impulse_response_guaranteed_with_zero_length_samples(ir_transform):
    empty_audio = torch.empty(0)
    with pytest.warns(UserWarning, match="An empty samples tensor was passed"):
        mixed_inputs = ir_transform(empty_audio)

    assert torch.equal(mixed_inputs, empty_audio)
