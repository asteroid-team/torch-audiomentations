from typing import Optional
import torch

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import calculate_rms
from ..utils.io import Audio


class Mix(BaseWaveformTransform):
    """
    Create a new sample by mixing it with another random sample from the same batch

    Signal-to-noise ratio (where "noise" is the second random sample) is selected
    randomly between `min_snr_in_db` and `max_snr_in_db`.

    `mix_target` controls how resulting targets are generated. It can be one of 
    "original" (targets are those of the original sample) or "union" (targets are the 
    union of original and overlapping targets)

    """

    supports_multichannel = True
    supported_modes = {"per_example", "per_channel"}
    requires_sample_rate = False
    requires_targets = False
    requires_target_rate = False

    def __init__(
        self,
        min_snr_in_db: float = 0.0,
        max_snr_in_db: float = 5.0,
        mix_target: str = "union",
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
        )
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.mix_target = mix_target
        if mix_target == "original":
            self._mix_target = lambda target, background_target, snr: target

        elif mix_target == "union":
            self._mix_target = lambda target, background_target, snr: torch.maximum(
                target, background_target
            )

        else:
            raise ValueError("mix_target must be one of 'original' or 'union'.")

    def randomize_parameters(
        self,
        selected_samples,
        sample_rate: Optional[int] = None,
        targets=None,
        target_rate: Optional[int] = None,
    ):

        batch_size, num_channels, num_samples = selected_samples.shape
        snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db, dtype=torch.float32, device=selected_samples.device,
            ),
            high=torch.tensor(
                self.max_snr_in_db, dtype=torch.float32, device=selected_samples.device,
            ),
            validate_args=True,
        )

        # randomize SNRs
        self.transform_parameters["snr_in_db"] = snr_distribution.sample(
            sample_shape=(batch_size,)
        )

        # randomize index of second sample
        self.transform_parameters["sample_idx"] = torch.randint(
            0, batch_size, (batch_size,), device=selected_samples.device,
        )

    def apply_transform(
        self,
        selected_samples: torch.Tensor,
        sample_rate: int = None,
        targets: torch.Tensor = None,
        target_rate: int = None,
    ):

        snr = self.transform_parameters["snr_in_db"]
        idx = self.transform_parameters["sample_idx"]

        background_samples = Audio.rms_normalize(selected_samples[idx])
        background_rms = calculate_rms(selected_samples) / (
            10 ** (snr.unsqueeze(dim=-1) / 20)
        )

        perturbed_samples = (
            selected_samples + background_rms.unsqueeze(-1) * background_samples
        )
        if targets is None:
            return perturbed_samples

        background_targets = targets[idx]
        perturbed_targets = self._mix_target(targets, background_targets, snr)
        return perturbed_samples, perturbed_targets
