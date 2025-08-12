import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import os
import faulthandler
from optuna.integration import PyTorchLightningPruningCallback
from hydra.core.hydra_config import HydraConfig

from BrainID.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
)

# Enable faulthandler for debugging segfaults
faulthandler.enable()

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
torch.set_float32_matmul_precision("high")

log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> float:
    """Train the model and return val/auroc_epoch for Optuna optimization."""

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Pass Hydra config to WandB logger if present
    for lg in logger:
        if "WandbLogger" in lg.__class__.__name__:
            wandb_logger = lg
            wandb_logger.experiment.config.update(
                OmegaConf.to_container(cfg, resolve=True),
                allow_val_change=True
            )
            break

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Handle resume/backbone loading
    if cfg.load_backbone and cfg.resume_training:
        raise ValueError(
            "Both load_backbone and resume_training are set. Please set only one."
        )
    if cfg.load_backbone:
        model.load_feature_weights(cfg.get("feat_ckpt"))
        resume_ckpt = None
    elif cfg.resume_training:
        model = hydra.utils.instantiate(cfg.model, _recursive_=True)
        resume_ckpt = cfg.get("resume_ckpt")
    else:
        resume_ckpt = None

    if cfg.get("freeze_feat", False):
        log.info("Freezing feature extractor")
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        ckpt_path=resume_ckpt,
    )

    # Retrieve AUROC after training
    metric_name = "val/auroc_epoch"
    if metric_name not in trainer.callback_metrics:
        raise ValueError(
            f"Metric '{metric_name}' not found in callback_metrics. "
            "Make sure QCMetrics logs it with on_epoch=True."
        )

    auroc_val = trainer.callback_metrics[metric_name]
    score = float(auroc_val.cpu().item())
    log.info(f"Final {metric_name}: {score:.4f}")

    return score


@hydra.main(
    version_base="1.3",
    config_path="../cfgs_hydra",
    config_name="train_qc_fetal",
)
def main(cfg: DictConfig) -> float:
    """Hydra main entry point — returns metric for Optuna sweeper."""
    return train(cfg)


if __name__ == "__main__":
    main()