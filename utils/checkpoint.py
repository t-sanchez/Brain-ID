#!/usr/bin/env python3

"""Functions that handle saving and loading of checkpoints."""

import os

import torch
import torch.nn as nn

import utils.distributed as du
import utils.logging as logging
from utils.env import checkpoint_pathmgr as pathmgr

from tabulate import tabulate

logger = logging.get_logger(__name__)


import copy
import logging
import re
from typing import Dict, List
import torch
from tabulate import tabulate


def _group_keys_by_module(keys: List[str], original_names: Dict[str, str]):
    """
    Params in the same submodule are grouped together.

    Args:
        keys: names of all parameters
        original_names: mapping from parameter name to their name in the checkpoint

    Returns:
        dict[name -> all other names in the same group]
    """

    def _submodule_name(key):
        pos = key.rfind(".")
        if pos < 0:
            return None
        prefix = key[: pos + 1]
        return prefix

    all_submodules = [_submodule_name(k) for k in keys]
    all_submodules = [x for x in all_submodules if x]
    all_submodules = sorted(all_submodules, key=len)

    ret = {}
    for prefix in all_submodules:
        group = [k for k in keys if k.startswith(prefix)]
        if len(group) <= 1:
            continue
        original_name_lcp = _longest_common_prefix_str(
            [original_names[k] for k in group]
        )
        if len(original_name_lcp) == 0:
            # don't group weights if original names don't share prefix
            continue

        for k in group:
            if k in ret:
                continue
            ret[k] = group
    return ret


def _longest_common_prefix(names):
    """
    ["abc.zfg", "abc.zef"] -> "abc."
    """
    names = [n.split(".") for n in names]
    m1, m2 = min(names), max(names)
    ret = [a for a, b in zip(m1, m2) if a == b]
    ret = ".".join(ret) + "." if len(ret) else ""
    return ret


def _longest_common_prefix_str(names):
    m1, m2 = min(names), max(names)
    lcp = []
    for a, b in zip(m1, m2):
        if a == b:
            lcp.append(a)
        else:
            break
    lcp = "".join(lcp)
    return lcp


def _group_str(names):
    """
    Turn "common1", "common2", "common3" into "common{1,2,3}"
    """
    lcp = _longest_common_prefix_str(names)
    rest = [x[len(lcp) :] for x in names]
    rest = "{" + ",".join(rest) + "}"
    ret = lcp + rest

    # add some simplification for BN specifically
    ret = ret.replace("bn_{beta,running_mean,running_var,gamma}", "bn_*")
    ret = ret.replace(
        "bn_beta,bn_running_mean,bn_running_var,bn_gamma", "bn_*"
    )
    return ret


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if du.is_master_proc() and not pathmgr.exists(checkpoint_dir):
        try:
            pathmgr.mkdirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    name = "checkpoint_latest.pyth"
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = pathmgr.ls(d) if pathmgr.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cfg, cur_iter):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cfg (CfgNode): configs to save.
        cur_epoch (int): current number of epoch of the model.
    """
    if cur_iter + 1 == cfg.SOLVER.MAX_EPOCH:
        return True

    return (cur_iter + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(path_to_job, model, optimizer, iter, cfg, scaler=None):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
        scaler (GradScaler): the mixed precision scale.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        return
    # Ensure that the checkpoint dir exists.
    pathmgr.mkdirs(get_checkpoint_dir(path_to_job))
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()

    # Record the state.
    checkpoint = {
        "epoch": iter,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    # Write the current epoch checkpoint & update the latest epoch checkpoint
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, iter + 1)
    with pathmgr.open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    path_to_latest_checkpoint = get_last_checkpoint(path_to_job)
    with pathmgr.open(path_to_latest_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    return path_to_checkpoint


def load_checkpoint(
    path_to_checkpoint,
    models,
    optimizer=None,
    model_keys=["model"],
    exclude_key=None,
    to_print=True,
):
    """
    Load the checkpoint from the given file.
    """
    assert pathmgr.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    if to_print:
        logger.info(
            "Loading network weights from {}.".format(path_to_checkpoint)
        )

    # Load the checkpoint on CPU to avoid GPU mem spike.

    def find_model_key(keys, model_key):
        for k in keys:
            if model_key in k:
                return k
        for k in keys:
            if "model" in k:
                if to_print:
                    logger.info(
                        'Have not found model state_dict according to the given key, but using the "model" as key instead!'
                    )
                    return k

    with pathmgr.open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    for i, model in enumerate(models):
        ms = model
        # ms = model.module if data_parallel else model # Account for the DDP wrapper in the multi-gpu setting.
        model_dict = ms.state_dict()

        k = find_model_key(checkpoint.keys(), model_keys[i])
        pre_train_dict = checkpoint[k]

        ms.load_state_dict(
            align_and_update_state_dicts(
                model_dict,
                pre_train_dict,
                exclude_key=exclude_key,
                to_print=to_print,
            ),
            strict=False,
        )

    if optimizer and "optimizaer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    best_val_stats = (
        checkpoint["best_val_stats"]
        if "best_val_stats" in checkpoint
        else None
    )
    return checkpoint["epoch"], best_val_stats


def load_test_checkpoint(cfg, model):
    """
    Loading checkpoint logic for testing.
    """
    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            squeeze_temporal=cfg.TEST.CHECKPOINT_SQUEEZE_TEMPORAL,
        )
    elif has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
        )
    else:
        logger.info(
            "Unknown way of loading checkpoint. Using random initialization, only for debugging."
        )


def load_train_checkpoint(cfg, model, optimizer, scaler=None):
    """
    Loading checkpoint logic for training.
    """
    if cfg.TRAIN.AUTO_RESUME and has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        logger.info("Load from last checkpoint, {}.".format(last_checkpoint))
        checkpoint_epoch = load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer, scaler=scaler
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "" and cfg.TRAIN.FINETUNE:
        logger.info("Finetune from given checkpoint file.")
        checkpoint_epoch = load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler=scaler,
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            freeze_pretrain=cfg.TRAIN.FREEZE_PRETRAIN,
        )
        start_epoch = (
            checkpoint_epoch + 1
            if cfg.TRAIN.FINETUNE_START_EPOCH == 0
            else cfg.TRAIN.FINETUNE_START_EPOCH
        )
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler=scaler,
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    return start_epoch


# Note the current matching is not symmetric.
# it assumes model_state_dict will have longer names.
def align_and_update_state_dicts(
    model_state_dict, ckpt_state_dict, exclude_key=None, to_print=True
):
    """
    Match names between the two state-dict, and returns a new chkpt_state_dict with names
    converted to match model_state_dict with heuristics. The returned dict can be later
    loaded with fvcore checkpointer.
    """
    if exclude_key is not None:
        model_keys = sorted(
            [k for k in model_state_dict.keys() if exclude_key not in k]
        )
    else:
        model_keys = sorted(model_state_dict.keys())
    original_keys = {x: x for x in ckpt_state_dict.keys()}
    ckpt_keys = sorted(ckpt_state_dict.keys())

    def match(a, b):
        if (a == b or a.endswith("." + b)) and to_print:
            print("matched")
            print(a, "--", b)
        return a == b or a.endswith("." + b)

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # ckpt_key string, if it matches
    match_matrix = [
        len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(model_keys), len(ckpt_keys)
    )
    # use the matched one with longest size in case of multiple matches
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # logger = logging.getLogger(__name__)
    # matched_pairs (matched checkpoint key --> matched model key)
    matched_keys = {}
    result_state_dict = {}
    for idx_model, idx_ckpt in enumerate(idxs.tolist()):
        if idx_ckpt == -1:
            continue
        key_model = model_keys[idx_model]
        key_ckpt = ckpt_keys[idx_ckpt]
        value_ckpt = ckpt_state_dict[key_ckpt]
        shape_in_model = model_state_dict[key_model].shape

        if shape_in_model != value_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_ckpt, value_ckpt.shape, key_model, shape_in_model
                )
            )
            if shape_in_model[0] != value_ckpt.shape[0] and len(
                shape_in_model
            ) == len(
                value_ckpt.shape
            ):  # different embed_dim setup
                logger.warning(
                    "{} will not be loaded. Please double check and see if this is desired.".format(
                        key_ckpt
                    )
                )
                logger.warning("--- shape_in_model: {}".format(shape_in_model))
                logger.warning("--- ckpt shape: {}".format(value_ckpt.shape))
            else:
                logger.warning(
                    "{} will be loaded for the center frame with the weights from the 2D conv layers in pre-trained models and\
                        initialize other weights as zero. Please double check and see if this is desired.".format(
                        key_ckpt
                    )
                )
                assert key_model not in result_state_dict
                logger.warning("--- shape_in_model: {}".format(shape_in_model))
                logger.warning("--- ckpt shape: {}".format(value_ckpt.shape))
                # load pre-trained 2D weights on the parameters' center termporal frame while others as 0. (B, C, (T,) H, W)
                nn.init.constant_(model_state_dict[key_model], 0.0)
                model_state_dict[key_model][
                    :, :, int(shape_in_model[2] / 2)
                ] = value_ckpt
                result_state_dict[key_model] = model_state_dict[key_model]
                logger.warning(
                    "--- loaded to T: {}".format(int(shape_in_model[2] / 2))
                )
                logger.warning(
                    "--- reshaped ckpt: {}".format(
                        result_state_dict[key_model].shape
                    )
                )
                matched_keys[key_ckpt] = key_model
        else:
            assert key_model not in result_state_dict
            result_state_dict[key_model] = value_ckpt
            if key_ckpt in matched_keys:  # already added to matched_keys
                logger.error(
                    "Ambiguity found for {} in checkpoint!"
                    "It matches at least two keys in the model ({} and {}).".format(
                        key_ckpt, key_model, matched_keys[key_ckpt]
                    )
                )
                raise ValueError(
                    "Cannot match one checkpoint key to multiple keys in the model."
                )
            if to_print:
                logger.info("Matching {} to {}".format(key_ckpt, key_model))
            matched_keys[key_ckpt] = key_model

    # logging:
    matched_model_keys = sorted(matched_keys.values())

    if len(matched_model_keys) == 0:
        logger.warning("No weights in checkpoint matched with model.")
        return ckpt_state_dict
    common_prefix = _longest_common_prefix(matched_model_keys)
    rev_matched_keys = {v: k for k, v in matched_keys.items()}
    original_keys = {
        k: original_keys[rev_matched_keys[k]] for k in matched_model_keys
    }

    model_key_groups = _group_keys_by_module(matched_model_keys, original_keys)

    table = []
    memo = set()
    for key_model in matched_model_keys:
        if to_print:
            print("  matched:", key_model)
        if key_model in memo:
            continue
        if key_model in model_key_groups:
            group = model_key_groups[key_model]
            memo |= set(group)
            shapes = [tuple(model_state_dict[k].shape) for k in group]
            table.append(
                (
                    _longest_common_prefix(
                        [k[len(common_prefix) :] for k in group]
                    )
                    + "*",
                    _group_str([original_keys[k] for k in group]),
                    " ".join([str(x).replace(" ", "") for x in shapes]),
                )
            )
        else:
            key_checkpoint = original_keys[key_model]
            shape = str(tuple(model_state_dict[key_model].shape))
            table.append(
                (key_model[len(common_prefix) :], key_checkpoint, shape)
            )
    table_str = tabulate(
        table,
        tablefmt="pipe",
        headers=["Names in Model", "Names in Checkpoint", "Shapes"],
    )
    if to_print:
        logger.info(
            "Following weights matched with "
            + (f"submodule {common_prefix[:-1]}" if common_prefix else "model")
            + ":\n"
            + table_str
        )

    unmatched_ckpt_keys = [
        k for k in ckpt_keys if k not in set(matched_keys.keys())
    ]
    unmatched_model_keys = [
        k for k in model_keys if k not in set(matched_keys.values())
    ]
    # for k in unmatched_ckpt_keys:
    # result_state_dict[k] = ckpt_state_dict[k]
    # result_state_dict[k] = model_state_dict[k]
    # logger.info('unmatched:', k)
    for k in unmatched_model_keys:
        # logger.info('unmatched:', k)
        result_state_dict[k] = model_state_dict[k]

    return result_state_dict
