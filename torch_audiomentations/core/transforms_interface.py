import warnings

import torch
from torch.distributions import Bernoulli

from torch_audiomentations.utils.multichannel import is_multichannel


class MultichannelAudioNotSupportedException(Exception):
    pass


class EmptyPathException(Exception):
    pass


class BaseWaveformTransform(torch.nn.Module):
    supports_multichannel = False

    def __init__(self, p: float = 0.5):
        super(BaseWaveformTransform, self).__init__()
        assert 0.0 <= p <= 1.0
        self._p = p
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
        if len(samples) == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(
                    self.__class__.__name__
                )
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
            batch_size = samples.shape[0]
            self.parameters = {
                "should_apply": self.bernoulli_distribution.sample(
                    sample_shape=(batch_size,)
                ).to(torch.bool)
            }
            if self.parameters["should_apply"].any():
                self.randomize_parameters(
                    samples[self.parameters["should_apply"]], sample_rate
                )

        if self.parameters["should_apply"].any():
            cloned_samples = samples.detach().clone()
            cloned_samples[self.parameters["should_apply"]] = self.apply_transform(
                samples[self.parameters["should_apply"]], sample_rate
            )
            return cloned_samples

        return samples

    def _forward_unimplemented(self, *inputs) -> None:
        # Avoid IDE error message like "Class ... must implement all abstract methods"
        # See also https://github.com/python/mypy/issues/8795#issuecomment-691658758
        pass

    def randomize_parameters(self, selected_samples, sample_rate: int):
        pass

    def apply_transform(self, selected_samples, sample_rate: int):
        raise NotImplementedError

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        raise NotImplementedError
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
