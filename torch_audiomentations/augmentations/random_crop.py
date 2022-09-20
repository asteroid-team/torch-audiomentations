import warnings
from typing import Optional, Union

import torch

from torch_audiomentations.utils.multichannel import is_multichannel
from ..core.transforms_interface import MultichannelAudioNotSupportedException


class RandomCrop(torch.nn.Module):
    """Crop the audio to a predefined length."""

    supports_multichannel = True

    def __init__(
        self,
        output_length: Union[float, int],
        output_length_unit: str = "samples",
        sample_rate: Optional[int] = None,
    ):
        """
        :param output_length: Desired length of output, in samples or in seconds,
            depending on the value of `output_length_unit`
        :param output_length_unit: The unit of output_length
        :param sample_rate:
        """
        super(RandomCrop, self).__init__()
        assert self.output_length_unit in (
            "samples",
            "seconds",
        ), 'output_length_unit must be "samples" or "seconds"'
        self.output_length = output_length
        self.output_length_unit = output_length_unit
        self.sample_rate = sample_rate

    def randomize_parameters(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        if self.output_length_unit == "seconds":
            num_samples = int(self.sample_rate * self.output_length)

            sample_rate = sample_rate or self.sample_rate
            if sample_rate is None:
                raise RuntimeError("sample_rate is required")
        elif self.output_length_unit == "samples":
            num_samples = int(self.output_length)
        else:
            raise ValueError('output_length_unit must be "samples" or "seconds"')

    def forward(self, samples, sample_rate: Optional[int] = None):
        if len(samples) == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(self.__class__.__name__)
            )
            return samples

        if len(samples.shape) != 3:
            raise RuntimeError(
                "torch-audiomentations expects input tensors to be three-dimensional, with"
                " dimension ordering like [batch_size, num_channels, num_samples]. If your"
                " audio is mono, you can use a shape like [batch_size, 1, num_samples]."
            )

        if is_multichannel(samples):
            if samples.shape[1] > samples.shape[2]:
                warnings.warn(
                    "Multichannel audio must have channels first, not channels last. In"
                    " other words, the shape must be (batch size, channels, samples), not"
                    " (batch_size, samples, channels)"
                )
            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException(
                    "{} only supports mono audio, not multichannel audio".format(
                        self.__class__.__name__
                    )
                )

        if samples.shape[2] < self.num_samples:
            warnings.warn("audio length less than cropping length")
            return samples

        start_indices = torch.randint(
            0, samples.shape[2] - self.num_samples, (samples.shape[2],)
        )
        samples_cropped = torch.empty(
            (samples.shape[0], samples.shape[1], self.num_samples), device=samples.device
        )
        for i, sample in enumerate(samples):

            samples_cropped[i] = sample.unsqueeze(0)[
                :, :, start_indices[i] : start_indices[i] + self.num_samples
            ]

        return samples_cropped
