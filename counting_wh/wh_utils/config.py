"""
Load and parse the config file for other modules

usage:

    from wh_utils.config import cfg
"""

import os
import yaml

config_path = os.path.join(os.getcwd(), "config.yml")

with open(config_path, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    cfg["tif_dir"] = cfg.get(
        "tif_dir", os.path.join(cfg["proj_root"], "images", "RawImages")
    )  # This is generated so not included in the config file


def resolve_device(value="auto"):
    """
    Turn a config 'device' value into something YOLOv5 accepts on this machine.

    "auto" picks the best available backend so one config works on every
    machine: an NVIDIA GPU (cuda:0) on Windows/Linux, Apple Silicon (mps) on
    macOS, otherwise cpu. Any other value is passed through untouched, so you
    can still pin a specific device such as "cuda:1" or "cpu".
    """
    if value is None:
        value = "auto"
    value = str(value)
    if value != "auto":
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
