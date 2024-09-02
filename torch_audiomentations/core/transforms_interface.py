import warnings

import torch
from torch import Tensor
from typing import Optional
from torch.distributions import Bernoulli

from torch_audiomentations.utils.multichannel import is_multichannel
from torch_audiomentations.utils.object_dict import ObjectDict


class MultichannelAudioNotSupportedException(Exception):
    pass


class EmptyPathException(Exception):
    pass


class ModeNotSupportedException(Exception):
    pass


class BaseWaveformTransform(torch.nn.Module):
    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        """

        :param mode:
            mode="per_channel" means each channel gets processed independently.
            mode="per_example" means each (multichannel) audio snippet gets processed
                independently, i.e. all channels in each audio snippet get processed with the
                same parameters.
            mode="per_batch" means all (multichannel) audio snippets in the batch get processed
                with the same parameters.
        :param p: The probability of the transform being applied to a batch/example/channel
            (see mode and p_mode). This number must be in the range [0.0, 1.0].
        :param p_mode: This optional argument can be set to "per_example" or "per_channel" if
            mode is set to "per_batch", or it can be set to "per_channel" if mode is set to
            "per_example". In the latter case, the transform is applied to the randomly selected
            examples, but the channels in those examples will be processed independently, i.e.
            with different parameters. Default value: Same as mode.
        :param sample_rate: sample_rate can be set either here or when
            calling the transform.
        :param target_rate: target_rate can be set either here or when
            calling the transform.
        :param output_type: This optional argument can be set to "tensor" or "dict".

        """
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.mode = mode
        self._p = p
        self.p_mode = p_mode
        if self.p_mode is None:
            self.p_mode = self.mode
        self.sample_rate = sample_rate
        self.target_rate = target_rate

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

        # Check validity of mode/p_mode combination
        if self.mode not in self.supported_modes:
            raise ModeNotSupportedException(
                "{} does not support mode {}".format(self.__class__.__name__, self.mode)
            )
        if self.p_mode == "per_batch":
            assert self.mode in ("per_batch", "per_example", "per_channel")
        elif self.p_mode == "per_example":
            assert self.mode in ("per_example", "per_channel")
        elif self.p_mode == "per_channel":
            assert self.mode == "per_channel"
        else:
            raise Exception("Unknown p_mode {}".format(self.p_mode))

        self.transform_parameters = {}
        self.are_parameters_frozen = False
        self.bernoulli_distribution = Bernoulli(self._p)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p
        # Update the Bernoulli distribution accordingly
        self.bernoulli_distribution = Bernoulli(self._p)

    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
        # TODO: add support for additional **kwargs (batch_size, ...)-shaped tensors
        # TODO: but do that only when we actually have a use case for it...
    ) -> ObjectDict:
        if not self.training:
            output = ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )
            return output.samples if self.output_type == "tensor" else output

        if not isinstance(samples, Tensor) or len(samples.shape) != 3:
            raise RuntimeError(
                "torch-audiomentations expects three-dimensional input tensors, with"
                " dimension ordering like [batch_size, num_channels, num_samples]. If your"
                " audio is mono, you can use a shape like [batch_size, 1, num_samples]."
            )

        batch_size, num_channels, num_samples = samples.shape

        if batch_size * num_channels * num_samples == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(self.__class__.__name__)
            )
            output = ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )
            return output.samples if self.output_type == "tensor" else output

        if is_multichannel(samples):
            if num_channels > num_samples:
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

        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None and self.is_sample_rate_required():
            raise RuntimeError("sample_rate is required")

        if targets is None and self.is_target_required():
            raise RuntimeError("targets is required")

        has_targets = targets is not None

        if has_targets and not self.supports_target:
            warnings.warn(f"Targets are not (yet) supported by {self.__class__.__name__}")

        if has_targets:
            if not isinstance(targets, Tensor) or len(targets.shape) != 4:
                raise RuntimeError(
                    "torch-audiomentations expects four-dimensional target tensors, with"
                    " dimension ordering like [batch_size, num_channels, num_frames, num_classes]."
                    " If your target is binary, you can use a shape like [batch_size, num_channels, num_frames, 1]."
                    " If your target is for the whole channel, you can use a shape like [batch_size, num_channels, 1, num_classes]."
                )

            (
                target_batch_size,
                target_num_channels,
                num_frames,
                num_classes,
            ) = targets.shape

            if target_batch_size != batch_size:
                raise RuntimeError(
                    f"samples ({batch_size}) and target ({target_batch_size}) batch sizes must be equal."
                )
            if num_channels != target_num_channels:
                raise RuntimeError(
                    f"samples ({num_channels}) and target ({target_num_channels}) number of channels must be equal."
                )

            target_rate = target_rate or self.target_rate
            if target_rate is None:
                if num_frames > 1:
                    target_rate = round(sample_rate * num_frames / num_samples)
                    warnings.warn(
                        f"target_rate is required by {self.__class__.__name__}. "
                        f"It has been automatically inferred from targets shape to {target_rate}. "
                        f"If this is incorrect, you can pass it directly."
                    )
                else:
                    # corner case where num_frames == 1, meaning that the target is for the whole sample,
                    # not frame-based. we arbitrarily set target_rate to 0.
                    target_rate = 0

        if not self.are_parameters_frozen:
            if self.p_mode == "per_example":
                p_sample_size = batch_size

            elif self.p_mode == "per_channel":
                p_sample_size = batch_size * num_channels

            elif self.p_mode == "per_batch":
                p_sample_size = 1

            else:
                raise Exception("Invalid mode")

            self.transform_parameters = {
                "should_apply": self.bernoulli_distribution.sample(
                    sample_shape=(p_sample_size,)
                ).to(dtype=torch.bool, device=samples.device)
            }

        if self.transform_parameters["should_apply"].any():
            cloned_samples = samples.clone()

            if has_targets:
                cloned_targets = targets.clone()
            else:
                cloned_targets = None
                selected_targets = None

            if self.p_mode == "per_channel":
                cloned_samples = cloned_samples.reshape(
                    batch_size * num_channels, 1, num_samples
                )
                selected_samples = cloned_samples[
                    self.transform_parameters["should_apply"]
                ]

                if has_targets:
                    cloned_targets = cloned_targets.reshape(
                        batch_size * num_channels, 1, num_frames, num_classes
                    )
                    selected_targets = cloned_targets[
                        self.transform_parameters["should_apply"]
                    ]

                if not self.are_parameters_frozen:
                    self.randomize_parameters(
                        samples=selected_samples,
                        sample_rate=sample_rate,
                        targets=selected_targets,
                        target_rate=target_rate,
                    )

                perturbed: ObjectDict = self.apply_transform(
                    samples=selected_samples,
                    sample_rate=sample_rate,
                    targets=selected_targets,
                    target_rate=target_rate,
                )

                cloned_samples[
                    self.transform_parameters["should_apply"]
                ] = perturbed.samples
                cloned_samples = cloned_samples.reshape(
                    batch_size, num_channels, num_samples
                )

                if has_targets:
                    cloned_targets[
                        self.transform_parameters["should_apply"]
                    ] = perturbed.targets
                    cloned_targets = cloned_targets.reshape(
                        batch_size, num_channels, num_frames, num_classes
                    )

                output = ObjectDict(
                    samples=cloned_samples,
                    sample_rate=perturbed.sample_rate,
                    targets=cloned_targets,
                    target_rate=perturbed.target_rate,
                )
                return output.samples if self.output_type == "tensor" else output

            elif self.p_mode == "per_example":
                selected_samples = cloned_samples[
                    self.transform_parameters["should_apply"]
                ]

                if has_targets:
                    selected_targets = cloned_targets[
                        self.transform_parameters["should_apply"]
                    ]

                if self.mode == "per_example":
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=selected_samples,
                            sample_rate=sample_rate,
                            targets=selected_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        samples=selected_samples,
                        sample_rate=sample_rate,
                        targets=selected_targets,
                        target_rate=target_rate,
                    )

                    cloned_samples[
                        self.transform_parameters["should_apply"]
                    ] = perturbed.samples

                    if has_targets:
                        cloned_targets[
                            self.transform_parameters["should_apply"]
                        ] = perturbed.targets

                    output = ObjectDict(
                        samples=cloned_samples,
                        sample_rate=perturbed.sample_rate,
                        targets=cloned_targets,
                        target_rate=perturbed.target_rate,
                    )
                    return output.samples if self.output_type == "tensor" else output

                elif self.mode == "per_channel":
                    (
                        selected_batch_size,
                        selected_num_channels,
                        selected_num_samples,
                    ) = selected_samples.shape

                    assert selected_num_samples == num_samples

                    selected_samples = selected_samples.reshape(
                        selected_batch_size * selected_num_channels,
                        1,
                        selected_num_samples,
                    )

                    if has_targets:
                        selected_targets = selected_targets.reshape(
                            selected_batch_size * selected_num_channels,
                            1,
                            num_frames,
                            num_classes,
                        )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=selected_samples,
                            sample_rate=sample_rate,
                            targets=selected_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        selected_samples,
                        sample_rate=sample_rate,
                        targets=selected_targets,
                        target_rate=target_rate,
                    )

                    perturbed.samples = perturbed.samples.reshape(
                        selected_batch_size, selected_num_channels, selected_num_samples
                    )
                    cloned_samples[
                        self.transform_parameters["should_apply"]
                    ] = perturbed.samples

                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(
                            selected_batch_size,
                            selected_num_channels,
                            num_frames,
                            num_classes,
                        )
                        cloned_targets[
                            self.transform_parameters["should_apply"]
                        ] = perturbed.targets

                    output = ObjectDict(
                        samples=cloned_samples,
                        sample_rate=perturbed.sample_rate,
                        targets=cloned_targets,
                        target_rate=perturbed.target_rate,
                    )
                    return output.samples if self.output_type == "tensor" else output

                else:
                    raise Exception("Invalid mode/p_mode combination")

            elif self.p_mode == "per_batch":
                if self.mode == "per_batch":
                    cloned_samples = cloned_samples.reshape(
                        1, batch_size * num_channels, num_samples
                    )

                    if has_targets:
                        cloned_targets = cloned_targets.reshape(
                            1, batch_size * num_channels, num_frames, num_classes
                        )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=cloned_samples,
                            sample_rate=sample_rate,
                            targets=cloned_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        samples=cloned_samples,
                        sample_rate=sample_rate,
                        targets=cloned_targets,
                        target_rate=target_rate,
                    )
                    perturbed.samples = perturbed.samples.reshape(
                        batch_size, num_channels, num_samples
                    )

                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(
                            batch_size, num_channels, num_frames, num_classes
                        )

                    return (
                        perturbed.samples if self.output_type == "tensor" else perturbed
                    )

                elif self.mode == "per_example":
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=cloned_samples,
                            sample_rate=sample_rate,
                            targets=cloned_targets,
                            target_rate=target_rate,
                        )

                    perturbed = self.apply_transform(
                        samples=cloned_samples,
                        sample_rate=sample_rate,
                        targets=cloned_targets,
                        target_rate=target_rate,
                    )

                    return (
                        perturbed.samples if self.output_type == "tensor" else perturbed
                    )

                elif self.mode == "per_channel":
                    cloned_samples = cloned_samples.reshape(
                        batch_size * num_channels, 1, num_samples
                    )

                    if has_targets:
                        cloned_targets = cloned_targets.reshape(
                            batch_size * num_channels, 1, num_frames, num_classes
                        )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(
                            samples=cloned_samples,
                            sample_rate=sample_rate,
                            targets=cloned_targets,
                            target_rate=target_rate,
                        )

                    perturbed: ObjectDict = self.apply_transform(
                        cloned_samples,
                        sample_rate,
                        targets=cloned_targets,
                        target_rate=target_rate,
                    )

                    perturbed.samples = perturbed.samples.reshape(
                        batch_size, num_channels, num_samples
                    )

                    if has_targets:
                        perturbed.targets = perturbed.targets.reshape(
                            batch_size, num_channels, num_frames, num_classes
                        )

                    return (
                        perturbed.samples if self.output_type == "tensor" else perturbed
                    )

                else:
                    raise Exception("Invalid mode")

            else:
                raise Exception("Invalid p_mode {}".format(self.p_mode))

        output = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
        return output.samples if self.output_type == "tensor" else output

    def _forward_unimplemented(self, *inputs) -> None:
        # Avoid IDE error message like "Class ... must implement all abstract methods"
        # See also https://github.com/python/mypy/issues/8795#issuecomment-691658758
        pass

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        pass

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        raise NotImplementedError()

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        raise NotImplementedError()
        # TODO: Clone the params and convert any tensors into json-serializable lists
        # return self.transform_parameters

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False

    def is_sample_rate_required(self) -> bool:
        return self.requires_sample_rate

    def is_target_required(self) -> bool:
        return self.requires_target
