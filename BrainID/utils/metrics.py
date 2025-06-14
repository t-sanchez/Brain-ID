from lightning.pytorch.callbacks import Callback
from torchmetrics import Accuracy, AUROC, Precision, Recall
from torchmetrics import Dice
from BrainID.utils.misc import nested_dict_to_device
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class QCMetrics(Callback):
    def __init__(
        self,
        on_train=True,
        on_val=True,
        on_test=True,
        run_fetmrqc=False,
        fetmrqc_df=None,
        fetmrqc_target=None,
    ):
        super().__init__()
        self.on_train = on_train
        self.on_val = on_val
        self.on_test = on_test
        self.run_fetmrqc = run_fetmrqc
        self.fetmrqc_df = fetmrqc_df
        self.target = fetmrqc_target
        if self.run_fetmrqc:
            assert (
                self.fetmrqc_df is not None
            ), "If run_fetmrqc is True, fetmrqc_df must be provided."
            self.fetmrqc_df = pd.read_csv(self.fetmrqc_df)
            self._train_and_pred_fetmrqc()

        self._metrics = {}
        if self.on_train:
            self._metrics["train"] = self._get_metrics_dict()
        if self.on_val:
            self._metrics["val"] = self._get_metrics_dict()
        if self.on_test:
            self._metrics["test"] = self._get_metrics_dict()

    def _train_and_pred_fetmrqc(self):
        rf = RandomForestClassifier()
        # All columns after centroid
        centroid_col = self.fetmrqc_df.columns.get_loc("centroid")
        x = self.fetmrqc_df.iloc[:, centroid_col + 1 :]
        y = self.fetmrqc_df[self.target]
        train_x = x[x["split"] == "train"]
        train_y = y[x["split"] == "train"]
        val_x = x[x["split"] == "val"]
        val_y = y[x["split"] == "val"]
        rf.fit(train_x, train_y)
        val_pred = rf.predict(val_x)

        self.fetmrqc_metrics = self._get_metrics_dict()
        for name, metric in self.fetmrqc_metrics.items():
            metric.update(val_pred, val_y)

    def _get_metrics_dict(self):
        return {
            "ba": Accuracy(task="binary", num_classes=2, average="macro"),
            "auroc": AUROC(task="binary", num_classes=2),
            "precision": Precision(task="binary", num_classes=2),
            "recall": Recall(task="binary", num_classes=2),
        }

    def _compute_metrics(self, trainer, pl_module, outputs, batch, split):
        batch = nested_dict_to_device(batch, "cpu")
        y_true = batch["label"]

        y_pred = outputs["preds"][0]["pred"]

        for name, metric in self._metrics[split].items():
            metric.update(y_pred, y_true)

        pl_module.log_dict(
            {
                f"{split}/{name}": metric.compute()
                for name, metric in self._metrics[split].items()
            },
            on_step=True,
            on_epoch=True,
            batch_size=len(y_true),
        )

    def on_train_start(self, trainer, pl_module) -> None:
        """Lightning hook that is called when training begins."""

        for k, v in self._metrics.items():
            for metric in v.values():
                metric.reset()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self.on_train:
            self._compute_metrics(trainer, pl_module, outputs, batch, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self.on_val:
            self._compute_metrics(trainer, pl_module, outputs, batch, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.on_test:
            self._compute_metrics(trainer, pl_module, outputs, batch, "test")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.run_fetmrqc:

            pl_module.log_dict(
                {
                    f"fetmrqc/{name}": metric.compute()
                    for name, metric in self.fetmrqc_metrics.items()
                },
                on_step=True,
                on_epoch=True,
                batch_size=len(self.fetmrqc_df),
            )


class SegMetrics(Callback):
    def __init__(self, on_train=True, on_val=True, on_test=True):
        super().__init__()
        self.on_train = on_train
        self.on_val = on_val
        self.on_test = on_test

        self._metrics = {}
        if self.on_train:
            self._metrics["train"] = self._get_metrics_dict()
        if self.on_val:
            self._metrics["val"] = self._get_metrics_dict()
        if self.on_test:
            self._metrics["test"] = self._get_metrics_dict()

    def _get_metrics_dict(self):
        return {
            "dice": Dice(num_classes=10, ignore_index=0),
        }

    def _compute_metrics(self, trainer, pl_module, outputs, batch, split):
        batch = nested_dict_to_device(batch, "cpu")

        y_true = batch["label"]
        y_pred = outputs["preds"][0]["seg"]

        for name, metric in self._metrics[split].items():
            metric.update(y_pred, y_true)

        pl_module.log_dict(
            {
                f"{split}/{name}": metric.compute()
                for name, metric in self._metrics[split].items()
            },
            on_step=True,
            on_epoch=True,
            batch_size=len(y_true),
        )

    def on_train_start(self, trainer, pl_module) -> None:
        """Lightning hook that is called when training begins."""

        for k, v in self._metrics.items():
            for metric in v.values():
                metric.reset()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self.on_train:
            self._compute_metrics(trainer, pl_module, outputs, batch, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self.on_val:
            self._compute_metrics(trainer, pl_module, outputs, batch, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.on_test:
            self._compute_metrics(trainer, pl_module, outputs, batch, "test")
