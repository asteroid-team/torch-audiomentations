import torch
from math import ceil

from torch_audiomentations.utils.fft import rfft, irfft
from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import calculate_rms
from ..utils.io import Audio


def _gen_noise(f_decay, num_samples, sample_rate, device):
    """
    Generate colored noise with f_decay decay using torch.fft
    """
    noise = torch.normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        (sample_rate,),
        device=device,
    )
    spec = rfft(noise)
    mask = 1 / (
        torch.linspace(1, (sample_rate / 2) ** 0.5, spec.shape[0], device=device)
        ** f_decay
    )
    spec *= mask
    noise = Audio.rms_normalize(irfft(spec).unsqueeze(0)).squeeze()
    noise = torch.cat([noise] * int(ceil(num_samples / sample_rate)))
    return noise[:num_samples]


class AddColoredNoise(BaseWaveformTransform):
    """
    Add colored noises to the input audio.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        min_f_decay: float = -2.0,
        max_f_decay: float = 2.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
    ):
        """
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximium SNR in dB.
        :param min_f_decay:
            defines the minimum frequency power decay (1/f**f_decay).
            Typical values are "white noise" (f_decay=0), "pink noise" (f_decay=1),
            "brown noise" (f_decay=2), "blue noise (f_decay=-1)" and "violet noise"
            (f_decay=-2)
        :param max_f_decay:
            defines the maximum power decay (1/f**f_decay) for non-white noises.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """

        super().__init__(mode, p, p_mode, sample_rate)

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay
        if self.min_f_decay > self.max_f_decay:
            raise ValueError("min_f_decay must not be greater than max_f_decay")

    def randomize_parameters(
        self, selected_samples: torch.Tensor, sample_rate: int = None
    ):
        """
        :params selected_samples: (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = selected_samples.shape

        # (batch_size, ) SNRs
        for param, mini, maxi in [
            ("snr_in_db", self.min_snr_in_db, self.max_snr_in_db),
            ("f_decay", self.min_f_decay, self.max_f_decay),
        ]:
            dist = torch.distributions.Uniform(
                low=torch.tensor(
                    mini, dtype=torch.float32, device=selected_samples.device
                ),
                high=torch.tensor(
                    maxi, dtype=torch.float32, device=selected_samples.device
                ),
                validate_args=True,
            )
            self.transform_parameters[param] = dist.sample(sample_shape=(batch_size,))

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        batch_size, num_channels, num_samples = selected_samples.shape

        if sample_rate is None:
            sample_rate = self.sample_rate

        # (batch_size, num_samples)
        noise = torch.stack(
            [
                _gen_noise(
                    self.transform_parameters["f_decay"][i],
                    num_samples,
                    sample_rate,
                    selected_samples.device,
                )
                for i in range(batch_size)
            ]
        )

        # (batch_size, num_channels)
        noise_rms = calculate_rms(selected_samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        return selected_samples + noise_rms.unsqueeze(-1) * noise.view(
            batch_size, 1, num_samples
        ).expand(-1, num_channels, -1)
