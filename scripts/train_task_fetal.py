import datetime
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import yaml
import json
import random
import time
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.checkpoint import load_checkpoint
import utils.logging as logging
import utils.misc as utils
from BrainID.visualizer import TaskVisualizer, FeatVisualizer
from BrainID.datasets import build_fetal_dataset
from BrainID.models import (
    build_downstream_model,
    build_optimizer,
    build_schedulers,
)
from BrainID.engine import train_one_epoch_downstream, eval_model
import wandb
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

wandb.login()
logger = logging.get_logger(__name__)
from fetalsynthgen.dataloader import FetalDataModule


def eval(args: Namespace, base_module: FetalDataModule) -> None:
    import pdb

    output_dir = utils.make_dir(args.out_dir)
    yaml.dump(
        vars(args), open(output_dir / "config.yaml", "w"), allow_unicode=True
    )

    vis_train_dir = utils.make_dir(os.path.join(output_dir, "vis-train"))
    vis_val_dir = utils.make_dir(os.path.join(output_dir, "vis-val"))
    ckp_output_dir = utils.make_dir(os.path.join(output_dir, "ckp"))
    ckp_epoch_dir = utils.make_dir(os.path.join(ckp_output_dir, "epoch"))

    # ============ setup logging  ... ============
    logging.setup_logging(output_dir)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info(args)

    if args.device is not None:  # assign to specified device
        device = args.device
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
    torch.backends.cudnn.benchmark = True

    # ============ preparing data ... ============
    # dataset_train = build_dataset_single(vars(args.dataset_name)['train'], split = 'train', args = args, device = args.device_generator if args.device_generator is not None else device)
    run = wandb.init(
        project="fetal_brain_id",
        config=vars(args),
        entity="tsanchez",
    )
    # base_module = instantiate(args.fetalsynthgen.__dict__)
    dataset_train = build_fetal_dataset(
        dm=base_module,
        dataset_name=args.dataset_name,
        target_key=args.target_key,
        transform_target=args.transform_target,
        target_threshold=args.target_threshold,
        device=device,
    )
    import pdb

    dataset_test = build_fetal_dataset(
        dm=base_module,
        dataset_name=args.dataset_name,
        target_key=args.target_key,
        train=False,
        transform_target=args.transform_target,
        target_threshold=args.target_threshold,
        device=device,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        # collate_fn=utils.collate_fn, # apply custom data cooker if needed
        num_workers=args.num_workers,
        # multiprocessing_context="spawn" if args.num_workers > 0 else None,
        # persistent_workers=True,
    )

    visualizers = {}  # "result": TaskVisualizer(args)}
    if args.visualizer.feat_vis:
        visualizers["feature"] = FeatVisualizer(args)

    # ============ building model ... ============

    (
        args,
        _,
        _,
        processors,
        criterion,
        postprocessor,
        model_joined,
    ) = build_downstream_model(args, device=device)

    # load feature extractor weights to evaluate
    load_checkpoint(
        args.feat_ext_ckp_path, [model_joined], model_keys=["model"]
    )

    if args.freeze_feat:
        model_joined.backbone.eval()

    logger.info(f"Feature extractor model built.")
    logger.info(
        "Num of feat model params: {}".format(
            sum(
                p.numel()
                for p in model_joined.backbone.parameters()
                if p.requires_grad
            )
        )
    )
    logger.info(
        "Num of trainable {} model params: {}".format(
            args.task,
            sum(
                p.numel()
                for p in model_joined.head.parameters()
                if p.requires_grad
            ),
        )
    )

    # ============ preparing optimizer and schedulers ... ============
    param_dicts = utils.get_params_groups(model_joined.head)
    optimizer = build_optimizer(args, param_dicts)
    lr_scheduler, wd_scheduler = build_schedulers(
        args, len(data_loader_train), args.lr, args.min_lr
    )

    feat_optimizer = feat_lr_scheduler = feat_wd_scheduler = None
    if not args.freeze_feat:
        feat_param_dicts = utils.get_params_groups(model_joined.backbone)
        feat_optimizer = build_optimizer(args, feat_param_dicts)
        feat_lr_scheduler, feat_wd_scheduler = build_schedulers(
            args,
            len(data_loader_train),
            args.feat_opt.lr,
            args.feat_opt.min_lr,
        )

    logger.info("Optimizer and schedulers ready.")

    best_val_stats = None
    args.start_epoch = 0
    # Load weights if provided
    if args.resume or args.eval_only:
        if args.ckp_path:
            ckp_path = args.ckp_path
        else:
            ckp_path = sorted(glob.glob(ckp_output_dir + "/*.pth"))

        args.start_epoch, best_val_stats = load_checkpoint(
            ckp_path, [model_joined.head], optimizer, ["model"]
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

        # ============ save model ... ============
        if epoch % 25 == 0:
            checkpoint_paths.append(
                ckp_epoch_dir / f"checkpoint_epoch_{epoch}.pth"
            )

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_joined.head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "best_val_stats": best_val_stats,
                    },
                    checkpoint_path,
                )

            if not args.freeze_feat:
                checkpoint_paths = [
                    ckp_output_dir / "checkpoint_feat_latest.pth"
                ]
                checkpoint_paths.append(
                    ckp_epoch_dir / f"checkpoint_feat_epoch_{epoch}.pth"
                )

                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": model_joined.backbone.state_dict(),
                            "optimizer": feat_optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            "best_val_stats": best_val_stats,
                        },
                        checkpoint_path,
                    )
        # Check parent dir of ckp_output_dir
        datetime = str(ckp_output_dir.parent.name).replace("-", "_")

        train_df_log = ckp_output_dir / f"train_log_{datetime}.csv"
        val_df_log = ckp_output_dir / f"val_log_{datetime}.csv"
        # ============ training one epoch ... ============

        log_stats = train_one_epoch_downstream(
            epoch,
            args,
            # feat_extractor,
            # model,
            model_joined,
            processors,
            criterion,
            data_loader_train,
            optimizer,
            lr_scheduler,
            wd_scheduler,
            feat_optimizer,
            feat_lr_scheduler,
            feat_wd_scheduler,
            postprocessor,
            visualizers,
            vis_train_dir,
            run,
            device,
            log_csv=train_df_log,
        )

        # ============ evaluating the model ... ============
        with torch.no_grad():
            eval_model(
                args,
                model_joined,
                processors,
                criterion,
                dataloader_test,
                epoch,
                logger=run,
                step="val",
                device=device,
                log_csv=val_df_log,
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
        datamodule = instantiate(cfg.fetalsynthgen)
    args = utils.dictconfig_to_namespace(cfg)
    eval(args, datamodule)
