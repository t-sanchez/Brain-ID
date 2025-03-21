"""Wrapper to train/test models."""

import os
import pytz
from datetime import datetime

from utils.config import Config


def build_out_dir(cfg, exp_name="", job_name=""):
    """
    Construct the output directory based on the config
    """
    tz_CH = pytz.timezone("Europe/Zurich")

    if cfg.eval_only:
        out_dir = os.path.join(
            cfg.out_dir,
            "Test",
            cfg.exp_name,
            cfg.job_name,
            datetime.now(tz_CH).strftime("%m%d-%H%M"),
        )
    else:
        out_dir = os.path.join(
            cfg.out_dir,
            cfg.exp_name,
            cfg.job_name,
            datetime.now(tz_CH).strftime("%m%d-%H%M"),
        )
    return out_dir


def merge_and_update_from_dict(cfg, dct):
    """
    (Compatible for submitit's Dict as attribute trick)
    Merge dict as dict() to config as CfgNode().
    Args:
        cfg: dict
        dct: dict
    """
    if dct is not None:
        for key, value in dct.items():
            if isinstance(value, dict):
                if key in cfg.keys():
                    sub_cfgnode = cfg[key]
                else:
                    sub_cfgnode = dict()
                    cfg.__setattr__(key, sub_cfgnode)
                sub_cfgnode = merge_and_update_from_dict(sub_cfgnode, value)
            else:
                cfg[key] = value
    return cfg


# def load_config(default_cfg_file, add_cfg_files=[], cfg_dir=""):
#     cfg = Config(default_cfg_file)

#     for cfg_file in add_cfg_files:
#         if os.path.isabs(cfg_file):
#             add_cfg = Config(cfg_file)
#         else:
#             # assert os.path.isabs(cfg_dir)
#             if not cfg_file.endswith(".yaml"):
#                 cfg_file += ".yaml"
#             add_cfg = Config(os.path.join(cfg_dir, cfg_file))
#         cfg = merge_and_update_from_dict(cfg, add_cfg)
#     return build_out_dir(
#         cfg, exp_name=cfg["exp_name"], job_name=cfg["job_name"]
#     )
