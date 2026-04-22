import logging
from pathlib import Path

from config import RunConfig


def setup_logging() -> None:
    """Configures the root logger to output to the console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def test_loading_config(json_file: Path) -> None:
    """Tests if a config JSON file can be loaded correctly."""
    _ = RunConfig.from_json(json_file)
    logging.debug("Configuration loaded!")


if __name__ == "__main__":
    setup_logging()
    conf_file_example = Path("configurations/example_configuration.json")
    conf_file_frenchr = Path("configurations/generate_french_royalty_cots.json")
    test_loading_config(conf_file_example)
    test_loading_config(conf_file_frenchr)
