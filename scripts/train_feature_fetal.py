import datetime
import os, sys
import pdb

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import yaml
import json
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils.checkpoint import load_checkpoint
import utils.logging as logging
import utils.misc as utils

from BrainID.visualizer import TaskVisualizer, FeatVisualizer
from BrainID.datasets import build_fetal_dataset
from BrainID.models import build_feat_model, build_optimizer, build_schedulers
from BrainID.engine import train_one_epoch_feature
import pdb

from BrainID.datasets.fetal_id_synth import (
    RandomBlockPatchFetalDataset,
    BlockRandomSampler,
)
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.get_logger(__name__)


def train(args: Namespace) -> None:
    output_dir = utils.make_dir(args.out_dir)
    yaml.dump(
        vars(args), open(output_dir / "config.yaml", "w"), allow_unicode=True
    )

    vis_train_dir = utils.make_dir(os.path.join(output_dir, "vis-train"))
    ckp_output_dir = utils.make_dir(os.path.join(output_dir, "ckp"))
    ckp_epoch_dir = utils.make_dir(os.path.join(ckp_output_dir, "epoch"))

    # ============ setup logging  ... ============
    logging.setup_logging(output_dir)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info(
        "\n".join(
            "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
        )
    )
    if args.device is not None:  # assign to specified device
        device = args.device
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"
    logger.info("device: %s" % device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # ============ preparing data ... ============
    base_module = instantiate(args.fetalsynthgen.__dict__)
    dataset_train = build_fetal_dataset(
        dm=base_module,
        dataset_name=args.dataset_name,
        train=True,
        device=device,
        # config_dir=args.train_fetal_cfg,
        # config_name="dhcp_onlinesynth",
        # dataset_name=args.dataset_name,
    )

    if args.train_patch:
        dataset_train = RandomBlockPatchFetalDataset(
            dataset=dataset_train,
            patch_size=args.patch_size,
            boundary=args.patch_boundary,
            patch_per_subject=args.patch_per_subject,
        )

    if args.train_patch:
        sampler_train = BlockRandomSampler(
            dataset_train, args.patch_per_subject
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        # collate_fn=utils.collate_fn, # apply custom data cooker if needed
        num_workers=args.num_workers,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    visualizers = {"result": TaskVisualizer(args)}
    if args.visualizer.feat_vis:
        visualizers["feature"] = FeatVisualizer(args)

    # ============ building model ... ============

    args, model, processors, criterion, postprocessor = build_feat_model(
        args, device=device
    )  # train: True; test: False

    model_without_ddp = model
    # Use multi-process data parallel model in the multi-gpu setting
    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info("Num of trainable model params: {}".format(n_parameters))

    # ============ preparing optimizer ... ============
    param_dicts = utils.get_params_groups(model_without_ddp)
    optimizer = build_optimizer(args, param_dicts)

    # ============ init schedulers ... ============
    lr_scheduler, wd_scheduler = build_schedulers(
        args, len(data_loader_train), args.lr, args.min_lr
    )
    logger.info(f"Optimizer and schedulers ready.")

    best_val_stats = None
    args.start_epoch = 0
    # Load weights if provided
    if args.resume or args.eval_only:
        if args.ckp_path:
            ckp_path = args.ckp_path
        else:
            ckp_path = sorted(glob.glob(ckp_output_dir + "/*.pth"))

        args.start_epoch, best_val_stats = load_checkpoint(
            ckp_path,
            [model_without_ddp],
            optimizer,
            ["model"],
            exclude_key="supervised_seg",
        )
        logger.info(f"Resume epoch: {args.start_epoch}")
    else:
        logger.info("Starting from scratch")
    if args.reset_epoch:
        args.start_epoch = 0
    logger.info(f"Start epoch: {args.start_epoch}")

    # ============ start training ... ============

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.n_epochs):

        checkpoint_paths = [ckp_output_dir / "checkpoint_latest.pth"]

        if epoch % 25 == 0:
            # ============ save model ... ============
            checkpoint_paths.append(
                ckp_epoch_dir / f"checkpoint_epoch_{epoch}.pth"
            )

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "best_val_stats": best_val_stats,
                    },
                    checkpoint_path,
                )

        # for itr, (subjects, samples) in enumerate(data_loader_train):
        #     if itr > 40:
        #         break
        #     with torch.no_grad():
        #         samples = utils.nested_dict_to_device(samples, device)
        #         subjects = utils.nested_dict_to_device(subjects, device)

        #         outputs, _ = model(samples)
        #         for processor in processors:
        #             outputs = processor(outputs, samples)

        #         epoch_vis_dir = (
        #             utils.make_dir(os.path.join(output_dir, "vis-train"))
        #             if epoch is not None
        #             else output_dir
        #         )
        #         visualizers["result"].visualize_all(
        #             subjects,
        #             samples,
        #             outputs,
        #             epoch_vis_dir,
        #             output_names=args.output_names + args.aux_output_names,
        #             target_names=args.target_names,
        #             suffix=f"_{itr}",
        #         )
        # pdb.set_trace()
        # ============ training one epoch ... ============

        log_stats = train_one_epoch_feature(
            epoch,
            args,
            model_without_ddp,
            processors,
            criterion,
            data_loader_train,
            optimizer,
            lr_scheduler,
            wd_scheduler,
            postprocessor,
            visualizers,
            vis_train_dir,
            device,
        )

        # ============ writing logs ... ============
        if utils.is_main_process():
            with (Path(output_dir) / "log.txt").open("a") as f:
                f.write("epoch %s - " % str(epoch).zfill(5))
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


#####################################################################################
if __name__ == "__main__":

    with initialize(version_base=None, config_path="../cfgs_hydra"):
        cfg = compose(
            config_name=sys.argv[1],
        )
    args = utils.dictconfig_to_namespace(cfg)
    train(args)
