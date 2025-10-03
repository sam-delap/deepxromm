# deepxromm/logging_utils.py
import logging
import os

LOG_DIR = os.path.join(os.path.expanduser("~"), ".deepxromm", "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("deepxromm")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
