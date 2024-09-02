import logging
import torch
from typing import Optional
from torch import Tensor
from torch.nn.functional import pad

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import convert_decibels_to_amplitude_ratio
from ..utils.object_dict import ObjectDict


class SpliceOut(BaseWaveformTransform):

    """
    spliceout augmentation proposed in https://arxiv.org/pdf/2110.00046.pdf
    silence padding is added at the end to retain the audio length.
    """

    supported_modes = {"per_batch", "per_example"}
    requires_sample_rate = True

    def __init__(
        self,
        num_time_intervals=8,
        max_width=400,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        """
        param num_time_intervals: number of time intervals to spliceout
        param max_width: maximum width of each spliceout in milliseconds
        param n_fft: size of FFT
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.num_time_intervals = num_time_intervals
        self.max_width = max_width

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        self.transform_parameters["splice_lengths"] = torch.randint(
            low=0,
            high=int(sample_rate * self.max_width * 1e-3),
            size=(samples.shape[0], self.num_time_intervals),
        )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        spliceout_samples = []

        for i in range(samples.shape[0]):
            random_lengths = self.transform_parameters["splice_lengths"][i]
            sample = samples[i][:, :]
            for j in range(self.num_time_intervals):
                start = torch.randint(
                    0,
                    sample.shape[-1] - random_lengths[j],
                    size=(1,),
                )

                if random_lengths[j] % 2 != 0:
                    random_lengths[j] += 1

                hann_window_len = random_lengths[j]
                hann_window = torch.hann_window(hann_window_len, device=samples.device)
                hann_window_left, hann_window_right = (
                    hann_window[: hann_window_len // 2],
                    hann_window[hann_window_len // 2 :],
                )

                fading_out, fading_in = (
                    sample[:, start : start + random_lengths[j] // 2],
                    sample[:, start + random_lengths[j] // 2 : start + random_lengths[j]],
                )
                crossfade = hann_window_right * fading_out + hann_window_left * fading_in
                sample = torch.cat(
                    (
                        sample[:, :start],
                        crossfade[:, :],
                        sample[:, start + random_lengths[j] :],
                    ),
                    dim=-1,
                )

            padding = torch.zeros(
                (samples[i].shape[0], samples[i].shape[-1] - sample.shape[-1]),
                dtype=torch.float32,
                device=sample.device,
            )
            sample = torch.cat((sample, padding), dim=-1)
            spliceout_samples.append(sample.unsqueeze(0))

        return ObjectDict(
            samples=torch.cat(spliceout_samples, dim=0),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
