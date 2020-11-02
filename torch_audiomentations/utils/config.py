import warnings
from typing import Any, Dict, Mapping, Text, Union
from pathlib import Path
import torch_audiomentations
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

# TODO: remove this try/except/else once Compose is available
# https://github.com/asteroid-team/torch-audiomentations/issues/23
try:
    from torch_audiomentations import Compose
except ImportError:
    COMPOSE_NOT_IMPLEMENTED = True
else:
    COMPOSE_NOT_IMPLEMENTED = False

# TODO: define this elsewhere?
# TODO: update when a new type of transform is added (e.g. BaseSpectrogramTransform?)
# TODO: remove this if/else once Compose is available
# https://github.com/asteroid-team/torch-audiomentations/issues/23
if COMPOSE_NOT_IMPLEMENTED:
    Transform = Union[BaseWaveformTransform]
else:
    Transform = Union[BaseWaveformTransform, Compose]


def from_dict(config: Dict[Text, Dict[Text, Any]]) -> Transform:
    """Instantiate a transform from a configuration dictionary.

    `from_dict` can be used to instantiate a transform from its class name.
    For instance, these two pieces of code are equivalent:

    >>> from torch_audiomentations import TransformClassName
    >>> transform = TransformClassName(param_name=param_value, ...)

    >>> transform = from_dict({"TransformClassName": {"param_name": param_value, ...}}) 

    Transforms composition is also supported:

    >>> compose = from_dict({"FirstTransform": {"param": value},
    ...                      "SecondTransform": {"param": value}})

    :param config: configuration - a configuration dictionary
    :returns: A transform.
    :rtype Transform:
    """

    if len(config) > 1:

        # TODO: remove this once Compose is available
        # https://github.com/asteroid-team/torch-audiomentations/issues/23
        if COMPOSE_NOT_IMPLEMENTED:
            raise ValueError(
                "torch_audiomentations does not implement Compose transforms"
            )

        # dictionary order is guaranteed to be insertion order since Python 3.7,
        # and it was already the case in Python 3.6 but not officially.
        # therefore, when `config` refers to more than one transform, we create
        # a Compose transform using the dictionary order

        transforms = [
            from_dict({TransformClassName: transform_params})
            for TransformClassName, transform_params in config.items()
        ]
        return Compose(transforms)

    TransformClassName, transform_params = config.popitem()

    try:
        TransformClass = getattr(torch_audiomentations, TransformClassName)
    except AttributeError:
        raise ValueError(
            f"torch_audiomentations does not implement {TransformClassName} transform."
        )

    if not isinstance(transform_params, dict):
        raise ValueError(
            "Transform parameters must be provided as {'param_name': param_value'} dictionary."
        )

    return TransformClass(**transform_params)


def from_yaml(file_yml: Union[Path, Text]) -> Transform:
    """Instantiate a transform from a YAML configuration file.

    `from_yaml` can be used to instantiate a transform from a YAML file.
    For instance, these two pieces of code are equivalent:

    >>> from torch_audiomentations import TransformClassName
    >>> transform = TransformClassName(param_name=param_value, ...)

    >>> transform = from_yaml("config.yml")
    
    where the content of `config.yml` is something like:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # config.yml
    TransformClassName: 
        param_name: param_value
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Transforms composition is also supported:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # config.yml
    FirstTransform:
        param: value
    SecondTransform:
        param: value
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :param file_yml: configuration file - a path to a YAML file with the following structure:
    :returns: A transform.
    :rtype Transform:
    """

    try:
        import yaml
    except ImportError as e:
        raise ImportError("")

    with open(file_yml, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return from_dict(config)
