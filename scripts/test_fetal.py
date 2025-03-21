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
)
from tqdm import tqdm
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from utils.process_cfg import build_out_dir
from utils.misc import PredictionLoggerQC, log_step
from BrainID.engine import eval_model
from hydra import compose, initialize
from hydra.utils import instantiate
import wandb

logger = logging.get_logger(__name__)


def get_params_groups(model):
    all = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        all.append(param)
    return [{"params": all}]


def get_dataloader(args, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        # collate_fn=utils.collate_fn, # apply custom data cooker if needed
        num_workers=args.num_workers,
        # multiprocessing_context="spawn" if args.num_workers > 0 else None,
        # persistent_workers=True,
    )


def eval(args: Namespace) -> None:
    output_dir = utils.make_dir(build_out_dir(args))
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
    logger.info(
        "\n".join(
            "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
        )
    )

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
        project="fetal_brain_id_eval",
        config=vars(args),
        entity="tsanchez",
    )

    base_module = instantiate(args.fetalsynthgen.__dict__)
    dataset_train = build_fetal_dataset(
        dm=base_module,
        dataset_name=args.dataset_name,
        target_key=args.target_key,
        transform_target=args.transform_target,
        target_threshold=args.target_threshold,
        device=device,
    )
    dataset_test = build_fetal_dataset(
        dm=base_module,
        dataset_name=args.dataset_name,
        target_key=args.target_key,
        train=False,
        transform_target=args.transform_target,
        target_threshold=args.target_threshold,
        device=device,
    )

    dataloader_train = get_dataloader(args, dataset_train)
    dataloader_test = get_dataloader(args, dataset_test)

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

    # Evaluation loop: evaluate the model's performance across epochs
    folder_path = Path(args.model_ckp_path).parent / "epoch"
    epochs = sorted(
        [
            int(ckp.split("_")[-1][:-4])
            for ckp in glob.glob(str(folder_path / "checkpoint_epoch_*.pth"))
        ]
    )
    load_checkpoint(
        args.feat_ext_ckp_path, [model_joined], model_keys=["model"]
    )

    datetime = str(ckp_output_dir.parent.name).replace("-", "_")
    val_df_log = ckp_output_dir / f"val_log_{datetime}.csv"
    test_df_log = ckp_output_dir / f"test_log_{datetime}.csv"
    for epoch in args.test_at_epochs:

        ckp_path = folder_path / f"checkpoint_epoch_{epoch}.pth"
        load_checkpoint(
            ckp_path, [model_joined.head], model_keys=["model"], to_print=False
        )
        model_joined.eval()
        with torch.no_grad():
            print(f"Epoch {epoch} -- Evaluating the training set")
            eval_model(
                args,
                model_joined,
                processors,
                criterion,
                dataloader_train,
                epoch=epoch,
                logger=run,
                step="val",
                device=args.device,
                log_csv=val_df_log,
            )
            print(f"Epoch {epoch} -- Evaluating the testing set")
            eval_model(
                args,
                model_joined,
                processors,
                criterion,
                dataloader_test,
                epoch,
                logger=run,
                step="test",
                device=args.device,
                log_csv=test_df_log,
            )


#####################################################################################

if __name__ == "__main__":
    with initialize(version_base=None, config_path="../cfgs_hydra"):
        cfg = compose(
            config_name=sys.argv[1],
        )
    args = utils.dictconfig_to_namespace(cfg)
    eval(args)

    # utils.launch_job(args, eval)
