"""
Logging module for implementing global logging in deepxromm
"""

import logging.config
import yaml
from pathlib import Path
from platformdirs import user_log_dir

config_file = Path(__file__) / "logging_config.yaml"
with config_file.open("r") as fp:
    config = yaml.safe_load(fp)

# Set filename path for file handler
log_dir = Path(user_log_dir("deepxromm"))
log_dir.mkdir(exist_ok=True, parents=True)
config["handlers"]["file"]["filename"] = str(log_dir / "deepxromm.log")
logging.config.dictConfig(config)

logger = logging.getLogger("deepxromm")
