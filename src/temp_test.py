import logging
from pathlib import Path

from config import RunConfig


def test_loading_config(json_file: Path) -> None:
    """Tests if a config JSON file can be loaded correctly."""
    _ = RunConfig.from_json(json_file)
    logging.debug("Configuration loaded!")


if __name__ == "__main__":
    configuration_file = Path("configurations/example_configuration.json")
    test_loading_config(configuration_file)
