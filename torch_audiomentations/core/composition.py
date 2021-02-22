import random
from typing import List

import torch
import typing

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform


class Compose(torch.nn.Module):
    """This class can apply a sequence of transforms to waveforms."""

    def __init__(
        self,
        transforms: List[torch.nn.Module],
        shuffle: bool = False,
        p: float = 1.0,
        p_mode="per_batch",
    ):
        """

        :param transforms: List of waveform transform instances
        :param shuffle: Should the order of transforms be shuffled?
        :param p: The probability of applying the Compose to the given batch.
        :param p_mode: Only "per_batch" is supported at the moment.
        """
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
        self.p = p
        if p_mode != "per_batch":
            # TODO: Support per_example as well? And per_channel?
            raise ValueError(f'p_mode = "{p_mode}" is not supported')
        self.p_mode = p_mode
        self.shuffle = shuffle

    def forward(self, samples, sample_rate: typing.Optional[int] = None):
        if random.random() < self.p:
            transform_indexes = list(range(len(self.transforms)))
            if self.shuffle:
                random.shuffle(transform_indexes)
            for i in transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, BaseWaveformTransform):
                    samples = self.transforms[i](samples, sample_rate)
                else:
                    samples = self.transforms[i](samples)
        return samples

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        for transform in self.transforms:
            transform.freeze_parameters()

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        for transform in self.transforms:
            transform.unfreeze_parameters()

    @property
    def supported_modes(self) -> set:
        """Return the intersection of supported modes of the transforms in the composition."""
        currently_supported_modes = {"per_batch", "per_example", "per_channel"}
        for transform in self.transforms:
            currently_supported_modes = currently_supported_modes.intersection(
                transform.supported_modes
            )
        return currently_supported_modes
