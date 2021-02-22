import torch
import typing

from ..core.transforms_interface import BaseWaveformTransform


class PeakNormalization(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in each audio snippet
    in the batch becomes 0 dBFS, i.e. the loudest level allowed if all samples must be between
    -1 and 1.

    This transform has an alternative mode (apply_to="only_too_loud_sounds") where it only
    applies to audio snippets that have extreme values outside the [-1, 1] range. This is useful
    for avoiding digital clipping in audio that is too loud, while leaving other audio
    untouched.
    """

    requires_sample_rate = False

    def __init__(
        self,
        apply_to="all",
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: typing.Optional[str] = None,
        sample_rate: typing.Optional[int] = None,
    ):
        super().__init__(mode, p, p_mode, sample_rate)
        assert apply_to in ("all", "only_too_loud_sounds")
        self.apply_to = apply_to

    def randomize_parameters(
        self, selected_samples, sample_rate: typing.Optional[int] = None
    ):
        # Compute the most extreme value of each multichannel audio snippet in the batch
        most_extreme_values, _ = torch.max(torch.abs(selected_samples), dim=-1)
        most_extreme_values, _ = torch.max(most_extreme_values, dim=-1)

        if self.apply_to == "all":
            # Avoid division by zero
            self.transform_parameters["selector"] = (
                most_extreme_values > 0.0
            )  # type: torch.BoolTensor
        elif self.apply_to == "only_too_loud_sounds":
            # Apply peak normalization only to audio examples with
            # values outside the [-1, 1] range
            self.transform_parameters["selector"] = (
                most_extreme_values > 1.0
            )  # type: torch.BoolTensor
        else:
            raise Exception("Unknown value of apply_to in PeakNormalization instance!")
        if self.transform_parameters["selector"].any():
            self.transform_parameters["divisors"] = torch.reshape(
                most_extreme_values[self.transform_parameters["selector"]], (-1, 1, 1)
            )

    def apply_transform(self, selected_samples, sample_rate: typing.Optional[int] = None):
        if "divisors" in self.transform_parameters:
            selected_samples[
                self.transform_parameters["selector"]
            ] /= self.transform_parameters["divisors"]
        return selected_samples
