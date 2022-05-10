import torch
import torchaudio.transforms as T
from torch import Tensor
from typing import Optional

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict


class VocalTractLengthPerturbation(BaseWaveformTransform):
    """
    Apply Vocal Tract Length Perturbation as defined in 
    http://www.cs.toronto.edu/~hinton/absps/perturb.pdf
    """
    supported_modes = {"per_example"}

    supports_multichannel = False
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_warp_factor: float = 0.9,
        max_warp_factor: float = 1.1,
        n_fft: int = 1024,
        hop_length: int = 256,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param min_warp_factor: The minimum warp factor to use.
        :param max_warp_factor: The maximum warp factor to use.
        :param n_fft: The number of FFT bins to use for stft.
        :param hop_length: The hop length to use for stft.
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:
        :param sample_rate:
        :param target_rate:
        :param output_type:
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        if min_warp_factor >= max_warp_factor:
            raise ValueError("max_warp_factor must be > min_warp_factor")
        if not sample_rate:
            raise ValueError("sample_rate is invalid.")

        self.min_warp_factor = min_warp_factor
        self.max_warp_factor = max_warp_factor
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    @classmethod
    def get_scale_factors(
        cls,
        n_freqs: int,
        warp_factor: Tensor,
        sample_rate: int,
        fhi: int = 4800
    ) -> Tensor:

        factors = []
        freqs = torch.linspace(0, 1, n_freqs)

        f_boundary = fhi * min(warp_factor, 1) / warp_factor
        nyquist = sample_rate / 2
        scale = fhi * min(warp_factor, 1)

        for f in freqs:
            f *= sample_rate
            if f <= f_boundary:
                factors.append(f * warp_factor)
            else:
                warp_freq = nyquist - (nyquist - scale) / (nyquist - scale / warp_factor) * (nyquist - f)
                factors.append(warp_freq)

        factors = torch.FloatTensor(factors)
        factors *= (n_freqs - 1) / torch.max(factors)  # normalize

        return factors

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        dist = torch.distributions.Uniform(
            low=torch.tensor(self.min_warp_factor, dtype=torch.float32, device=samples.device),
            high=torch.tensor(self.max_warp_factor, dtype=torch.float32, device=samples.device),
            validate_args=True,
        )
        self.transform_parameters['warp_factor'] = dist.sample()

    def apply_transform(
        self, 
        samples: Tensor, 
        sample_rate: int,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None
    ) -> ObjectDict:

        batch_size, num_channels, num_samples = samples.shape
        assert num_channels == 1, "Only single channel audio is supported"

        n_to_pad = self.hop_length - (num_samples % self.hop_length)  # enforce integer hoplengths for the FFT
        padded_samples = torch.nn.functional.pad(samples, (0, n_to_pad), 'constant', 0.)

        original_spect = torch.stft(
            padded_samples.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )
        n_freqs = original_spect.size(-2)
        transformed_spect = torch.zeros_like(original_spect)

        warp_factors = self.get_scale_factors(
            n_freqs,
            self.transform_parameters['warp_factor'],
            sample_rate,
        ).to(samples.device)

        # apply warp factor to spectrogram
        for i in range(n_freqs):
            if i == 0 or i + 1 >= n_freqs:
                transformed_spect[:, i, :] = original_spect[:, i, :]
            else:
                warp_up = warp_factors[i] - torch.floor(warp_factors[i])
                warp_down = 1. - warp_up
                pos = int(torch.floor(warp_factors[i]))

                transformed_spect[:, pos, :] += warp_down * original_spect[:, i, :]
                transformed_spect[:, pos + 1, :] += warp_up * original_spect[:, i, :]

        transformed_samples = torch.istft(
            transformed_spect,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )[:, :-n_to_pad]
        transformed_samples = transformed_samples.unsqueeze(1)

        return ObjectDict(
            samples=transformed_samples,
            sample_rate=self.sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
