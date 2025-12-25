from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def load_config_file(config_file_path: Path) -> CommentedMap:
    """Load a YAML file as a commented map"""
    yaml = YAML()
    with open(config_file_path, "r") as fp:
        config = yaml.load(fp)

    return config


def save_config_file(config_data: CommentedMap, config_file_path: Path) -> None:
    """Save a CommentedMap to a YAML file"""
    yaml = YAML()
    with open(config_file_path, "w") as fp:
        config = yaml.dump(config_data, fp)

    return config
