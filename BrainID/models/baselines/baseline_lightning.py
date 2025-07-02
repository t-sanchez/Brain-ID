from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, Dice
import monai
from BrainID.models.joiner import get_joiner
from BrainID.utils.misc import nested_dict_to_device
from BrainID.utils.misc import nested_dict_copy
import pdb

import os




class BaselineModel(LightningModule):
    def __init__(
        self,
        task: str,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        n_batches_vis: int = 3,
    ) -> None:
        """Initialize a UnetModule.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.task = task
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # for averaging loss across batches
        self.n_batches_vis = n_batches_vis

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.input_key = "input"  # if self.task != "seg" else "image"
        self.visualization_data = []

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param samples: An input dictionary
        :return: A dictionary of outputs
        """
        outputs, _ = self.model(samples)

        return outputs

    def return_dict_copy(self, dictionary, device):
        dictionary = nested_dict_to_device(dictionary, device)
        dict_copy = nested_dict_copy(dictionary)
        del dictionary
        return dict_copy

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        batch = nested_dict_to_device(batch, self.device)

        outputs = self.forward(batch[self.input_key])

        subjects = {k: batch[k] for k in batch.keys() if k != self.input_key}
        samples = [{self.input_key: x} for x in batch[self.input_key]]
        losses = self.criterion(outputs, subjects, samples)

        return (
            losses,
            self.return_dict_copy(outputs, "cpu"),
            self.return_dict_copy(subjects, "cpu"),
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, preds, targets = self.model_step(batch)

        loss = losses["loss"]
        # update and log metrics
        self.train_loss(loss)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return {"loss": loss, "preds": preds}

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        losses, preds, targets = self.model_step(batch)

        self.val_loss(losses["loss"])
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        if batch_idx < self.n_batches_vis:
            preds_save = {
                "pred_" + k: [p[k] for p in preds] for k in preds[0].keys()
            }
            self.visualization_data.append(
                {
                    "inputs": [
                        x.detach().cpu() for x in batch[self.input_key]
                    ],
                    "batch_idx": batch_idx,
                    **targets,
                    **preds_save,
                }
            )

        return {"loss": losses["loss"], "preds": preds}

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def on_validation_end(self):
        # Explicitly delete tensors in visualization_data to help free memory
        for item in self.visualization_data:
            for key, value in item.items():
                if isinstance(value, list):
                    for v in value:
                        if isinstance(v, torch.Tensor):
                            del v
                elif isinstance(value, torch.Tensor):
                    del value
        del self.visualization_data
        self.visualization_data = []
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self):
        self.visualization_data = []

    def load_feature_weights(self, ckpt_path):
        """Load the feature weights from a checkpoint file.

        :param ckpt_path: Path to the checkpoint file.
        """
        if ckpt_path is not None:
            # Load the checkpoint for the feature extractor -- The ckpt will be the
            checkpoint = torch.load(ckpt_path, weights_only=False)[
                "state_dict"
            ]
            # Extract and match the keys that start with model.backbone
            # and remove the prefix "model.backbone."
            # This is because the checkpoint was saved with a different prefix
            # than the one used in the model.
            checkpoint = {
                k.replace("model.backbone.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("model.backbone.")
            }

            self.model.backbone.load_state_dict(checkpoint)

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler for training.

        :return: A tuple containing the optimizer and the learning rate scheduler.
        """
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, "monitor": "train/loss"}]
