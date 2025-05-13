from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import os

from BrainID.utils import RankedLogger, instantiate_callbacks, instantiate_loggers
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

torch.set_float32_matmul_precision('high')
log = RankedLogger(__name__, rank_zero_only=True)
import faulthandler

faulthandler.enable()

def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # import pdb

    # pdb.set_trace()
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)


    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")


    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        #fast_dev_run=4,
    )

    # Assert either load_backbone or resume_training is set
    if cfg.load_backbone and cfg.resume_training:
        raise ValueError(
            "Both load_backbone and resume_training are set. Please set only one of them."
        )
    if cfg.load_backbone:
        model.load_feature_weights(cfg.get("feat_ckpt"))
        resume_ckpt = None
    elif cfg.resume_training:
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        resume_ckpt = cfg.get("resume_ckpt")
    else:
        resume_ckpt = None
        
    
    
    trainer.fit(
    model=model,
    train_dataloaders=datamodule.train_dataloader(),
    val_dataloaders=datamodule.val_dataloader(),
    ckpt_path = resume_ckpt,
    )


@hydra.main(
    version_base="1.3", config_path="../cfgs_hydra", config_name="train_qc_fetal"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
