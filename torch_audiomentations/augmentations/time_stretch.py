import typing

import torch
import torchaudio
from packaging import version

from ..core.transforms_interface import BaseWaveformTransform

# Note: torch.istft was very slow in pytorch==1.6.0 and did not exist in pytorch<1.6.0
if version.parse(torch.__version__) >= version.parse("1.7.0"):
    istft = torch.istft
else:
    from torchaudio.functional import istft


class TimeStretch(BaseWaveformTransform):
    """
    Time stretch the given audio without changing the pitch.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        min_rate=0.8,
        max_rate=1.25,
        n_fft=400,
        leave_length_unchanged=True,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: typing.Optional[str] = None,
        sample_rate: typing.Optional[int] = None,
    ):
        super().__init__(mode, p, p_mode, sample_rate)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.n_fft = n_fft
        assert leave_length_unchanged
        self.leave_length_unchanged = True
        self.time_stretch = torchaudio.transforms.TimeStretch()

    def randomize_parameters(
        self, selected_samples, sample_rate: typing.Optional[int] = None
    ):
        distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_rate, dtype=torch.float32, device=selected_samples.device
            ),
            high=torch.tensor(
                self.max_rate, dtype=torch.float32, device=selected_samples.device
            ),
            validate_args=True,
        )
        selected_batch_size = selected_samples.size(0)
        self.transform_parameters["rates"] = distribution.sample(
            sample_shape=(selected_batch_size,)
        )

    def apply_transform(self, selected_samples, sample_rate: typing.Optional[int] = None):
        selected_batch_size = selected_samples.size(0)

        time_stretched_examples = []
        max_time_stretched_sound_length = 0
        for i in range(selected_batch_size):
            rate = self.transform_parameters["rates"][i]
            spec = torch.stft(selected_samples[i], self.n_fft)
            time_stretched_spec = self.time_stretch(spec, overriding_rate=rate)
            time_stretched_example = istft(time_stretched_spec, self.n_fft)
            time_stretched_examples.append(time_stretched_example)

            num_samples = time_stretched_example.shape[-1]
            if num_samples > max_time_stretched_sound_length:
                max_time_stretched_sound_length = num_samples

        time_stretched_batch = torch.zeros(
            size=(selected_samples.shape), dtype=torch.float32
        )
        for i in range(len(time_stretched_examples)):
            time_stretched_batch[
                i, :, 0 : time_stretched_examples[i].shape[-1]
            ] = time_stretched_examples[i]

        return time_stretched_batch
