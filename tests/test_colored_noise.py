import random
import pytest
import torch

from torch_audiomentations import AddColoredNoise
from torch_audiomentations.utils.io import Audio
from .utils import TEST_FIXTURES_DIR


@pytest.fixture
def setup_audio():
    sample_rate = 16000
    audio = Audio(sample_rate=sample_rate)
    batch_size = 16
    empty_input_audio = torch.empty(0, 1, 16000)

    input_audio = audio(TEST_FIXTURES_DIR / "acoustic_guitar_0.wav").unsqueeze(0)
    input_audios = torch.cat([input_audio] * batch_size, dim=0)

    cl_noise_transform_guaranteed = AddColoredNoise(20, p=1.0, output_type="dict")
    cl_noise_transform_no_guarantee = AddColoredNoise(20, p=0.0, output_type="dict")

    return {
        "sample_rate": sample_rate,
        "empty_input_audio": empty_input_audio,
        "input_audio": input_audio,
        "input_audios": input_audios,
        "cl_noise_transform_guaranteed": cl_noise_transform_guaranteed,
        "cl_noise_transform_no_guarantee": cl_noise_transform_no_guarantee,
    }


def test_colored_noise_no_guarantee_with_single_tensor(setup_audio):
    input_audio = setup_audio["input_audio"]
    transform = setup_audio["cl_noise_transform_no_guarantee"]
    sample_rate = setup_audio["sample_rate"]

    mixed_input = transform(input_audio, sample_rate).samples
    assert torch.equal(mixed_input, input_audio)
    assert mixed_input.size(0) == input_audio.size(0)


def test_background_noise_no_guarantee_with_empty_tensor(setup_audio):
    empty_input_audio = setup_audio["empty_input_audio"]
    transform = setup_audio["cl_noise_transform_no_guarantee"]
    sample_rate = setup_audio["sample_rate"]

    with pytest.warns(UserWarning, match="An empty samples tensor was passed"):
        mixed_input = transform(empty_input_audio, sample_rate).samples

    assert torch.equal(mixed_input, empty_input_audio)
    assert mixed_input.size(0) == empty_input_audio.size(0)


def test_colored_noise_guaranteed_with_zero_length_samples(setup_audio):
    empty_input_audio = setup_audio["empty_input_audio"]
    transform = setup_audio["cl_noise_transform_guaranteed"]
    sample_rate = setup_audio["sample_rate"]

    with pytest.warns(UserWarning, match="An empty samples tensor was passed"):
        mixed_input = transform(empty_input_audio, sample_rate).samples

    assert torch.equal(mixed_input, empty_input_audio)
    assert mixed_input.size(0) == empty_input_audio.size(0)


def test_colored_noise_guaranteed_with_single_tensor(setup_audio):
    input_audio = setup_audio["input_audio"]
    transform = setup_audio["cl_noise_transform_guaranteed"]
    sample_rate = setup_audio["sample_rate"]

    mixed_input = transform(input_audio, sample_rate).samples
    assert not torch.equal(mixed_input, input_audio)
    assert mixed_input.size(0) == input_audio.size(0)
    assert mixed_input.size(1) == input_audio.size(1)


def test_colored_noise_guaranteed_with_batched_tensor(setup_audio):
    random.seed(42)
    input_audios = setup_audio["input_audios"]
    transform = setup_audio["cl_noise_transform_guaranteed"]
    sample_rate = setup_audio["sample_rate"]

    mixed_inputs = transform(input_audios, sample_rate).samples
    assert not torch.equal(mixed_inputs, input_audios)
    assert mixed_inputs.size(0) == input_audios.size(0)
    assert mixed_inputs.size(1) == input_audios.size(1)


def test_same_min_max_f_decay(setup_audio):
    random.seed(42)
    input_audios = setup_audio["input_audios"]
    sample_rate = setup_audio["sample_rate"]

    transform = AddColoredNoise(
        20, min_f_decay=1.0, max_f_decay=1.0, p=1.0, output_type="dict"
    )
    outputs = transform(input_audios, sample_rate).samples
    assert outputs.size(0) == input_audios.size(0)
    assert outputs.size(1) == input_audios.size(1)


def test_invalid_params():
    with pytest.raises(ValueError):
        AddColoredNoise(min_snr_in_db=30, max_snr_in_db=3, p=1.0, output_type="dict")
    with pytest.raises(ValueError):
        AddColoredNoise(min_f_decay=2, max_f_decay=1, p=1.0, output_type="dict")


def test_various_lengths_and_sample_rates():
    random.seed(42)
    transform = AddColoredNoise(
        min_snr_in_db=10, max_snr_in_db=12, p=1.0, output_type="dict"
    )

    for _ in range(100):
        length = random.randint(1000, 100_000)
        sample_rate = random.randint(1000, 100_000)
        input_audio = torch.randn(1, 1, length, dtype=torch.float32)
        output_audio = transform(input_audio, sample_rate=sample_rate).samples

        assert output_audio.shape == input_audio.shape
        assert output_audio.dtype == input_audio.dtype

    input_audio = torch.randn(1, 1, 16001, dtype=torch.float32)
    output_audio = transform(input_audio, sample_rate=16001).samples
    assert output_audio.shape == input_audio.shape
    assert not torch.equal(output_audio, input_audio)
