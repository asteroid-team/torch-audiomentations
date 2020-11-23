import unittest

from tests.utils import TEST_FIXTURES_DIR
from torch_audiomentations import from_dict, from_yaml
from torch_audiomentations import Gain, Compose


class TestFromConfig(unittest.TestCase):
    def test_from_dict(self):
        config = {
            "transform": "Gain",
            "params": {"min_gain_in_db": -12.0, "mode": "per_channel"},
        }
        transform = from_dict(config)

        assert isinstance(transform, Gain)
        assert transform.min_gain_in_db == -12.0
        assert transform.max_gain_in_db == 6.0
        assert transform.mode == "per_channel"

    def test_from_yaml(self):
        file_yml = TEST_FIXTURES_DIR / "config.yml"
        transform = from_yaml(file_yml)

        assert isinstance(transform, Gain)
        assert transform.min_gain_in_db == -12.0
        assert transform.max_gain_in_db == 6.0
        assert transform.mode == "per_channel"

    def test_from_dict_compose(self):
        config = {
            "transform": "Compose",
            "params": {
                "shuffle": True,
                "transforms": [
                    {
                        "transform": "Gain",
                        "params": {"min_gain_in_db": -12.0, "mode": "per_channel"},
                    },
                    {"transform": "PolarityInversion"},
                ],
            },
        }
        transform = from_dict(config)
        assert isinstance(transform, Compose)

    def test_from_yaml_compose(self):
        file_yml = TEST_FIXTURES_DIR / "config_compose.yml"
        transform = from_yaml(file_yml)
        assert isinstance(transform, Compose)
