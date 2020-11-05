from pathlib import Path
from typing import Any, Dict, Text, Union

import torch_audiomentations
from torch_audiomentations import Compose
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

# TODO: define this elsewhere?
# TODO: update when a new type of transform is added (e.g. BaseSpectrogramTransform? OneOf? SomeOf?)
# https://github.com/asteroid-team/torch-audiomentations/issues/26
Transform = Union[BaseWaveformTransform, Compose]


def from_dict(config: Dict[Text, Union[Text, Dict[Text, Any]]]) -> Transform:
    """Instantiate a transform from a configuration dictionary.

    `from_dict` can be used to instantiate a transform from its class name.
    For instance, these two pieces of code are equivalent:

    >>> from torch_audiomentations import Gain
    >>> transform = Gain(min_gain_in_db=-12.0)

    >>> transform = from_dict({'transform': 'Gain',
    ...                        'params': {'min_gain_in_db': -12.0}})

    Transforms composition is also supported:

    >>> compose = from_dict(
    ...    {'transform': 'Compose',
    ...     'params': {'transforms': [{'transform': 'Gain',
    ...                                'params': {'min_gain_in_db': -12.0,
    ...                                           'mode': 'per_channel'}},
    ...                               {'transform': 'PolarityInversion'}],
    ...                'shuffle': True}})

    :param config: configuration - a configuration dictionary
    :returns: A transform.
    :rtype Transform:
    """

    try:
        TransformClassName: Text = config["transform"]
    except KeyError:
        raise ValueError(
            "A (currently missing) 'transform' key should be used to define the transform type."
        )

    try:
        TransformClass = getattr(torch_audiomentations, TransformClassName)
    except AttributeError:
        raise ValueError(
            f"torch_audiomentations does not implement {TransformClassName} transform."
        )

    transform_params: Dict = config.get("params", dict())
    if not isinstance(transform_params, dict):
        raise ValueError(
            "Transform parameters must be provided as {'param_name': param_value} dictionary."
        )

    if TransformClassName in ["Compose", "OneOf", "SomeOf"]:
        transform_params["transforms"] = [
            from_dict(sub_transform_config)
            for sub_transform_config in transform_params["transforms"]
        ]

    return TransformClass(**transform_params)


def from_yaml(file_yml: Union[Path, Text]) -> Transform:
    """Instantiate a transform from a YAML configuration file.

    `from_yaml` can be used to instantiate a transform from a YAML file.
    For instance, these two pieces of code are equivalent:

    >>> from torch_audiomentations import Gain
    >>> transform = Gain(min_gain_in_db=-12.0, mode="per_channel")

    >>> transform = from_yaml("config.yml")

    where the content of `config.yml` is something like:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # config.yml
    transform: Gain
    params:
      min_gain_in_db: -12.0
      mode: per_channel
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Transforms composition is also supported:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # config.yml
    transform: Compose
    params:
      shuffle: True
      transforms:
        - transform: Gain
          params:
            min_gain_in_db: -12.0
            mode: per_channel
        - transform: PolarityInversion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :param file_yml: configuration file - a path to a YAML file with the following structure:
    :returns: A transform.
    :rtype Transform:
    """

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML package is needed by `from_yaml`: please install it first."
        )

    with open(file_yml, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return from_dict(config)
