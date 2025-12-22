"""
Logging module for implementing global logging in deepxromm
"""

import logging.config
import yaml
from pathlib import Path
from platformdirs import user_log_dir, user_config_dir

# Load default config
config_file = Path(__file__) / "default_logging_config.yaml"
with config_file.open("r") as fp:
    config = yaml.safe_load(fp)

# Load user config (if it exists)
config_dir = Path(user_config_dir("deepxromm"))
user_config_file = config_dir / "logging_config.yaml"
if user_config_file.exists():
    with user_config_file.open("r") as fp:
        user_config = yaml.safe_load(fp)

# Merge 2 configs to provide defaults, without overwriting user preferences
config = config | user_config

# Set filename path for file handler
log_dir = Path(user_log_dir("deepxromm"))
log_dir.mkdir(exist_ok=True, parents=True)
config["handlers"]["file"]["filename"] = str(log_dir / "deepxromm.log")

# Configure logging package using logging module
logging.config.dictConfig(config)

# Get non-root logger
logger = logging.getLogger("deepxromm")
