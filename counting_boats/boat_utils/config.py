"""
Load and parse the config file for other modules

usage:

    from boat_utils.config import cfg
"""

import os
import warnings

import yaml

config_path = os.path.join(os.getcwd(), "config.yml")

# config.yml drives the automated Planet acquisition pipeline only. The
# training / testing / deployment workflows each pass their own config file
# explicitly, so this is optional: if it is absent (for example it has been
# archived), fall back to an empty cfg rather than making the whole package
# impossible to import.
if os.path.exists(config_path):
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader) or {}
    if cfg.get("output_dir"):
        os.makedirs(cfg["output_dir"], exist_ok=True)
    cfg["tif_dir"] = cfg.get(
        "tif_dir", os.path.join(cfg.get("proj_root", "."), "images", "RawImages")
    )  # This is generated so not included in the config file
else:
    cfg = {}
    warnings.warn(
        f"No config.yml found at {config_path}. That file is only needed for the "
        "automated Planet acquisition pipeline; training, testing and deployment "
        "pass their own config explicitly and are unaffected.",
        stacklevel=2,
    )


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
