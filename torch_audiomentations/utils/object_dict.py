# Inspired by tornado
# https://www.tornadoweb.org/en/stable/_modules/tornado/util.html#ObjectDict

try:
    import typing
    from typing import cast

    _ObjectDictBase = typing.Dict[str, typing.Any]
except ImportError:
    _ObjectDictBase = dict


class ObjectDict(_ObjectDictBase):
    """
    Make a dictionary behave like an object, with attribute-style access.

    Here are some examples of how it can be used:
    
    o = ObjectDict(my_dict)
    # or like this:
    o = ObjectDict(samples=samples, sample_rate=sample_rate)

    # Attribute-style access
    samples = o.samples

    # Dict-style access
    samples = o["samples"]
    """

    def __getattr__(self, name):
        # type: (str) -> Any
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        # type: (str, Any) -> None
        self[name] = value
