import random
import torchaudio.transforms as T

from typing import Optional

from ..core.transforms_interface import BaseWaveformTransform


class TimeStretch(BaseWaveformTransform):
    """
    Time stretch the signal without changing the pitch

    Based on https://github.com/KentoNishi/torch-time-stretch
    """
    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
    ):
        super().__init__(mode, p, p_mode, sample_rate)

        assert min_rate > 0.1
        assert max_rate < 10

        if min_rate > max_rate:
            raise ValueError("min_rate must be smaller than max_rate")

        if not sample_rate:
            raise ValueError("sample_rate is invalid")

        self._sample_rate = sample_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.n_fft = n_fft if n_fft is not None else sample_rate // 64
        self.hop_length = hop_length if hop_length is not None else self.n_fft // 32

    def randomize_parameters(self, selected_samples: torch.Tensor, sample_rate: int = None):
        """
        :param selected_samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        self.transform_parameters['rate'] = random.uniform(self.min_rate, self.max_rate)

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        """
        :param selected_samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = selected_samples.shape

        transformed_samples = selected_samples.reshape(batch_size * num_channels, num_samples)
        transformed_samples = torch.stft(transformed_samples, n_fft=self.n_fft, hop_length=self.hop_length)[None, ...]

        stretcher = T.TimeStretch(fixed_rate=self.transform_parameters['rate'], n_freq=transformed_samples.shape[2], hop_length=self.hop_length)
        transformed_samples = stretcher(transformed_samples)
        del stretcher

        transformed_samples = torch.istft(transformed_samples[0], self.n_fft, self.hop_length)
        transformed_samples = transformed_samples.reshape(batch_size, num_channels, transformed_samples.shape[1])

        return transformed_samples
