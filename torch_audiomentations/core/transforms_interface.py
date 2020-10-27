import warnings

import torch
import typing
from torch.distributions import Bernoulli

from torch_audiomentations.utils.multichannel import is_multichannel


class MultichannelAudioNotSupportedException(Exception):
    pass


class EmptyPathException(Exception):
    pass


class BaseWaveformTransform(torch.nn.Module):
    supports_multichannel = False

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: typing.Optional[str] = None,
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
        """
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.mode = mode
        self._p = p
        self.p_mode = p_mode
        if self.p_mode is None:
            self.p_mode = self.mode

        # Check validity of mode/p_mode combination
        if self.p_mode == "per_batch":
            assert self.mode in ("per_batch", "per_example", "per_channel")
        elif self.p_mode == "per_example":
            assert self.mode in ("per_example", "per_channel")
        elif self.p_mode == "per_channel":
            assert self.mode == "per_channel"
        else:
            raise Exception("Unknown mode/p_mode {}".format(self.p_mode))

        self.parameters = {}
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

    def forward(self, samples, sample_rate: int):
        if not self.training:
            return samples

        if len(samples) == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(self.__class__.__name__)
            )
            return samples

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

        if not self.are_parameters_frozen:
            if self.p_mode == "per_example":
                p_sample_size = samples.shape[0]
            elif self.p_mode == "per_channel":
                p_sample_size = samples.shape[0] * samples.shape[1]
            elif self.p_mode == "per_batch":
                p_sample_size = 1
            else:
                raise Exception("Invalid mode")
            self.parameters = {
                "should_apply": self.bernoulli_distribution.sample(
                    sample_shape=(p_sample_size,)
                ).to(torch.bool)
            }

        if self.parameters["should_apply"].any():
            cloned_samples = samples.clone()

            if self.p_mode == "per_channel":
                batch_size = cloned_samples.shape[0]
                num_channels = cloned_samples.shape[1]
                cloned_samples = cloned_samples.reshape(
                    batch_size * num_channels, 1, cloned_samples.shape[2]
                )
                selected_samples = cloned_samples[self.parameters["should_apply"]]

                if not self.are_parameters_frozen:
                    self.randomize_parameters(selected_samples, sample_rate)

                cloned_samples[self.parameters["should_apply"]] = self.apply_transform(
                    selected_samples, sample_rate
                )

                cloned_samples = cloned_samples.reshape(
                    batch_size, num_channels, cloned_samples.shape[2]
                )

                return cloned_samples

            elif self.p_mode == "per_example":
                selected_samples = cloned_samples[self.parameters["should_apply"]]

                if self.mode == "per_example":
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(selected_samples, sample_rate)

                    cloned_samples[
                        self.parameters["should_apply"]
                    ] = self.apply_transform(selected_samples, sample_rate)
                    return cloned_samples
                elif self.mode == "per_channel":
                    batch_size = selected_samples.shape[0]
                    num_channels = selected_samples.shape[1]
                    selected_samples = selected_samples.reshape(
                        batch_size * num_channels, 1, selected_samples.shape[2]
                    )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(selected_samples, sample_rate)

                    perturbed_samples = self.apply_transform(
                        selected_samples, sample_rate
                    )
                    perturbed_samples = perturbed_samples.reshape(
                        batch_size, num_channels, selected_samples.shape[2]
                    )

                    cloned_samples[self.parameters["should_apply"]] = perturbed_samples
                    return cloned_samples
                else:
                    raise Exception("Invalid mode/p_mode combination")
            elif self.p_mode == "per_batch":
                if self.mode == "per_batch":
                    batch_size = cloned_samples.shape[0]
                    num_channels = cloned_samples.shape[1]
                    cloned_samples = cloned_samples.reshape(
                        1, batch_size * num_channels, cloned_samples.shape[2]
                    )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(cloned_samples, sample_rate)

                    perturbed_samples = self.apply_transform(cloned_samples, sample_rate)
                    perturbed_samples = perturbed_samples.reshape(
                        batch_size, num_channels, cloned_samples.shape[2]
                    )
                    return perturbed_samples
                elif self.mode == "per_example":
                    if not self.are_parameters_frozen:
                        self.randomize_parameters(cloned_samples, sample_rate)
                    return self.apply_transform(cloned_samples, sample_rate)
                elif self.mode == "per_channel":
                    batch_size = cloned_samples.shape[0]
                    num_channels = cloned_samples.shape[1]
                    cloned_samples = cloned_samples.reshape(
                        batch_size * num_channels, 1, cloned_samples.shape[2]
                    )

                    if not self.are_parameters_frozen:
                        self.randomize_parameters(cloned_samples, sample_rate)

                    perturbed_samples = self.apply_transform(cloned_samples, sample_rate)

                    perturbed_samples = perturbed_samples.reshape(
                        batch_size, num_channels, cloned_samples.shape[2]
                    )
                    return perturbed_samples
                else:
                    raise Exception("Invalid mode")
            else:
                raise Exception("Invalid p_mode {}".format(self.p_mode))

        return samples

    def _forward_unimplemented(self, *inputs) -> None:
        # Avoid IDE error message like "Class ... must implement all abstract methods"
        # See also https://github.com/python/mypy/issues/8795#issuecomment-691658758
        pass

    def randomize_parameters(self, selected_samples, sample_rate: int):
        pass

    def apply_transform(self, selected_samples, sample_rate: int):
        raise NotImplementedError()

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        raise NotImplementedError()
        # TODO: Clone the params and convert any tensors into json-serializable lists
        # return self.parameters

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
