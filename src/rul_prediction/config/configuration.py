"""
Configuration loader for the RUL prediction project.

WHY: Hardcoding paths, hyperparameters, and settings directly in scripts
means every change requires editing code. A YAML-driven config lets
non-engineers (or future-you in a hurry) change behavior without touching
source code, and makes experiments reproducible (you can version-control
different config files per experiment).

HOW: Loads configs/config.yaml into a plain Python dict (kept simple for
Phase 1 -- later phases may upgrade this to typed dataclasses or Pydantic
models for stricter validation).

WHERE: Used like this:

    from rul_prediction.config.configuration import load_config
    config = load_config()
    raw_data_dir = config["paths"]["raw_data_dir"]
"""

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and return the YAML configuration file as a dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        RULException: If the file is missing or cannot be parsed.
    """
    try:
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RULException(e, sys) from e
