from pathlib import Path
from typing import Any, Dict, Text, Optional, Union
from importlib import import_module

import torch_audiomentations
from torch_audiomentations import Compose
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

# TODO: define this elsewhere?
# TODO: update when a new type of transform is added (e.g. BaseSpectrogramTransform? OneOf? SomeOf?)
# https://github.com/asteroid-team/torch-audiomentations/issues/26
Transform = Union[BaseWaveformTransform, Compose]


def get_class_by_name(
    class_name: str, default_module_name: str = "torch_audiomentations"
) -> type:
    """Load class by its name

    Parameters
    ----------
    class_name : `str`
    default_module_name : `str`, optional
        When provided and `class_name` does not contain the absolute path.
        Defaults to "torch_audiomentations".

    Returns
    -------
    Klass : `type`
        Class.

    Example
    -------
    >>> YourAugmentation = get_class_by_name('your_package.your_module.YourAugmentation')
    >>> YourAugmentation = get_class_by_name('YourAugmentation', default_module_name='your_package.your_module')

    >>> from torch_audiomentations import Gain
    >>> assert Gain == get_class_by_name('Gain')
    """
    tokens = class_name.split(".")

    if len(tokens) == 1:
        if default_module_name is None:
            msg = (
                f'Could not infer module name from class name "{class_name}".'
                f"Please provide default module name."
            )
            raise ValueError(msg)
        module_name = default_module_name
    else:
        module_name = ".".join(tokens[:-1])
        class_name = tokens[-1]

    return getattr(import_module(module_name), class_name)


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
        TransformClass = get_class_by_name(TransformClassName)
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
