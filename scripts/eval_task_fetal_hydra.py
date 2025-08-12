import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
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

def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        #fast_dev_run=4,
        #limit_train_batches=0.05,
        #limit_val_batches=0.1,
    )

    model = hydra.utils.instantiate(cfg.model, _recursive_=True)
    
    resume_ckpt = cfg.get("resume_ckpt")
    weights = torch.load(resume_ckpt, map_location="cpu", weights_only=False)

    trainer.test(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=resume_ckpt)



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
    test(cfg)


if __name__ == "__main__":
    main()
