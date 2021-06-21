import torch
import typing

from ..core.transforms_interface import BaseWaveformTransform


def shift_gpu(tensor: torch.Tensor, r: torch.Tensor, rollover: bool = False):
    """Shift or roll a batch of tensors"""
    b, c, t = tensor.shape

    # Arange indexes
    x = torch.arange(t, device=tensor.device)

    # Apply Roll
    r = r[:, None, None]
    idxs = (x - r).expand([b, c, t])
    ret = torch.gather(tensor, 2, idxs % t)
    if rollover:
        return ret

    # Cut where we've rolled over
    cut_points = (idxs + 1).clamp(0)
    cut_points[cut_points > t] = 0
    ret[cut_points == 0] = 0
    return ret


def shift_cpu(
    selected_samples: torch.Tensor, shift_samples: torch.Tensor, rollover: bool = False
):
    """Shift or roll a batch of tensors with the help of a for loop and torch.roll()"""
    selected_batch_size = selected_samples.size(0)

    for i in range(selected_batch_size):
        num_samples_to_shift = shift_samples[i].item()
        selected_samples[i] = torch.roll(
            selected_samples[i], shifts=num_samples_to_shift, dims=-1
        )

        if not rollover:
            if num_samples_to_shift > 0:
                selected_samples[i, ..., :num_samples_to_shift] = 0.0
            elif num_samples_to_shift < 0:
                selected_samples[i, ..., num_samples_to_shift:] = 0.0

    return selected_samples


class Shift(BaseWaveformTransform):
    """
    Shift the audio forwards or backwards, with or without rollover
    """

    def __init__(
        self,
        min_shift: float = -0.5,
        max_shift: float = 0.5,
        shift_unit: str = "fraction",
        rollover: bool = True,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: typing.Optional[str] = None,
        sample_rate: typing.Optional[int] = None,
    ):
        """

        :param min_shift: minimum amount of shifting in time. See also shift_unit.
        :param max_shift: maximum amount of shifting in time. See also shift_unit.
        :param shift_unit: Defines the unit of the value of min_shift and max_shift.
            "fraction": Fraction of the total sound length
            "samples": Number of audio samples
            "seconds": Number of seconds
        :param rollover: When set to True, samples that roll beyond the first or last position
            are re-introduced at the last or first. When set to False, samples that roll beyond
            the first or last position are discarded. In other words, rollover=False results in
            an empty space (with zeroes).
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(mode, p, p_mode, sample_rate)
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.shift_unit = shift_unit
        self.rollover = rollover
        if self.min_shift > self.max_shift:
            raise ValueError("min_shift must not be greater than max_shift")
        if self.shift_unit not in ("fraction", "samples", "seconds"):
            raise ValueError('shift_unit must be "samples", "fraction" or "seconds"')

    def randomize_parameters(
        self, selected_samples, sample_rate: typing.Optional[int] = None
    ):
        if self.shift_unit == "samples":
            min_shift_in_samples = self.min_shift
            max_shift_in_samples = self.max_shift
        elif self.shift_unit == "fraction":
            min_shift_in_samples = int(round(self.min_shift * selected_samples.shape[-1]))
            max_shift_in_samples = int(round(self.max_shift * selected_samples.shape[-1]))
        elif self.shift_unit == "seconds":
            min_shift_in_samples = int(round(self.min_shift * sample_rate))
            max_shift_in_samples = int(round(self.max_shift * sample_rate))
        else:
            raise ValueError("Invalid shift_unit")

        assert (
            torch.iinfo(torch.int32).min
            <= min_shift_in_samples
            <= torch.iinfo(torch.int32).max
        )
        assert (
            torch.iinfo(torch.int32).min
            <= max_shift_in_samples
            <= torch.iinfo(torch.int32).max
        )
        selected_batch_size = selected_samples.size(0)
        if min_shift_in_samples == max_shift_in_samples:
            self.transform_parameters["num_samples_to_shift"] = torch.full(
                size=(selected_batch_size,),
                fill_value=min_shift_in_samples,
                dtype=torch.int32,
                device=selected_samples.device,
            )
        else:
            self.transform_parameters["num_samples_to_shift"] = torch.randint(
                low=min_shift_in_samples,
                high=max_shift_in_samples + 1,
                size=(selected_batch_size,),
                dtype=torch.int32,
                device=selected_samples.device,
            )

    def apply_transform(self, selected_samples, sample_rate: typing.Optional[int] = None):
        r = self.transform_parameters["num_samples_to_shift"]
        # Select fastest implementation based on device
        shift = shift_gpu if selected_samples.device.type == "cuda" else shift_cpu
        return shift(selected_samples, r, self.rollover)

    def is_sample_rate_required(self) -> bool:
        # Sample rate is required only if shift_unit is "seconds"
        return self.shift_unit == "seconds"
