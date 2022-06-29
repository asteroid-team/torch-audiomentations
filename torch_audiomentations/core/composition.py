import random
from typing import List, Union, Optional, Tuple

from torch import Tensor
import torch.nn
import warnings

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict


class BaseCompose(torch.nn.Module):
    """This class can apply a sequence of transforms to waveforms."""

    def __init__(
        self,
        transforms: List[
            torch.nn.Module
        ],  # FIXME: do we really want to support regular nn.Module?
        shuffle: bool = False,
        p: float = 1.0,
        p_mode="per_batch",
        output_type: Optional[str] = None,
    ):
        """
        :param transforms: List of waveform transform instances
        :param shuffle: Should the order of transforms be shuffled?
        :param p: The probability of applying the Compose to the given batch.
        :param p_mode: Only "per_batch" is supported at the moment.
        :param output_type: This optional argument can be set to "tensor" or "dict".
        """
        super().__init__()
        self.p = p
        if p_mode != "per_batch":
            # TODO: Support per_example as well? And per_channel?
            raise ValueError(f'p_mode = "{p_mode}" is not supported')
        self.p_mode = p_mode
        self.shuffle = shuffle
        self.are_parameters_frozen = False

        if output_type is None:
            warnings.warn(
                f"Transforms now expect an `output_type` argument that currently defaults to 'tensor', "
                f"will default to 'dict' in v0.12, and will be removed in v0.13. Make sure to update "
                f"your code to something like:\n"
                f"  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n"
                f"  >>> augmented_samples = augment(samples).samples",
                FutureWarning,
            )
            output_type = "tensor"

        elif output_type == "tensor":
            warnings.warn(
                f"`output_type` argument will default to 'dict' in v0.12, and will be removed in v0.13. "
                f"Make sure to update your code to something like:\n"
                f"your code to something like:\n"
                f"  >>> augment = {self.__class__.__name__}(..., output_type='dict')\n"
                f"  >>> augmented_samples = augment(samples).samples",
                DeprecationWarning,
            )

        self.output_type = output_type

        self.transforms = torch.nn.ModuleList(transforms)
        for tfm in self.transforms:
            tfm.output_type = "dict"

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        self.are_parameters_frozen = True
        for transform in self.transforms:
            transform.freeze_parameters()

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False
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


class Compose(BaseCompose):
    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        inputs = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )

        if random.random() < self.p:
            transform_indexes = list(range(len(self.transforms)))
            if self.shuffle:
                random.shuffle(transform_indexes)
            for i in transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                    inputs = self.transforms[i](**inputs)

                else:
                    assert isinstance(tfm, torch.nn.Module)
                    # FIXME: do we really want to support regular nn.Module?
                    inputs.samples = self.transforms[i](inputs.samples)

        return inputs.samples if self.output_type == "tensor" else inputs


class SomeOf(BaseCompose):
    """
    SomeOf randomly picks several of the given transforms and applies them.
    The number of transforms to be applied can be chosen as follows:

      - Pick exactly n transforms
        Example: pick exactly 2 of the transforms
                 `SomeOf(2, [transform1, transform2, transform3])`

      - Pick between a minimum and maximum number of transforms
        Example: pick 1 to 3 of the transforms
                 `SomeOf((1, 3), [transform1, transform2, transform3])`

        Example: Pick 2 to all of the transforms
                 `SomeOf((2, None), [transform1, transform2, transform3])`
    """

    def __init__(
        self,
        num_transforms: Union[int, Tuple[int, int]],
        transforms: List[torch.nn.Module],
        p: float = 1.0,
        p_mode="per_batch",
        output_type: Optional[str] = None,
    ):
        super().__init__(
            transforms=transforms, p=p, p_mode=p_mode, output_type=output_type
        )

        self.transform_indexes = []
        self.num_transforms = num_transforms
        self.all_transforms_indexes = list(range(len(self.transforms)))

        if isinstance(num_transforms, tuple):
            self.min_num_transforms = num_transforms[0]
            self.max_num_transforms = (
                num_transforms[1] if num_transforms[1] else len(transforms)
            )
        else:
            self.min_num_transforms = self.max_num_transforms = num_transforms

        assert self.min_num_transforms >= 1, "min_num_transforms must be >= 1"
        assert self.min_num_transforms <= len(
            transforms
        ), "num_transforms must be <= len(transforms)"
        assert self.max_num_transforms <= len(
            transforms
        ), "max_num_transforms must be <= len(transforms)"

    def randomize_parameters(self):
        num_transforms_to_apply = random.randint(
            self.min_num_transforms, self.max_num_transforms
        )
        self.transform_indexes = sorted(
            random.sample(self.all_transforms_indexes, num_transforms_to_apply)
        )

    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        inputs = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )

        if random.random() < self.p:

            if not self.are_parameters_frozen:
                self.randomize_parameters()

            for i in self.transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                    inputs = self.transforms[i](**inputs)

                else:
                    assert isinstance(tfm, torch.nn.Module)
                    # FIXME: do we really want to support regular nn.Module?
                    inputs.samples = self.transforms[i](inputs.samples)

        return inputs.samples if self.output_type == "tensor" else inputs


class OneOf(SomeOf):
    """
    OneOf randomly picks one of the given transforms and applies it.
    """

    def __init__(
        self,
        transforms: List[torch.nn.Module],
        p: float = 1.0,
        p_mode="per_batch",
        output_type: Optional[str] = None,
    ):
        super().__init__(
            num_transforms=1,
            transforms=transforms,
            p=p,
            p_mode=p_mode,
            output_type=output_type,
        )
