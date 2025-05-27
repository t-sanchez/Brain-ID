import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from lightning.pytorch.callbacks import Callback
import pdb

import os


def print_open_fds():
    pid = os.getpid()
    num_fds = len(os.listdir(f"/proc/{pid}/fd"))
    print(f"[PID {pid}] Open file descriptors: {num_fds}")


def get_slices(
    volume,
    directions=("axial", "coronal", "sagittal"),
    num_slices=5,
    spacing=5,
):
    """
    Extract num_slices evenly spaced slices from a 3D tensor in the given directions.
    """
    slices = {}

    D, H, W = volume.shape[-3:]

    half = num_slices // 2

    for direction in directions:
        if direction == "axial":
            center = W // 2
            indices = [
                int(torch.clamp(torch.tensor(center + i * spacing), 0, W - 1))
                for i in range(-half, half + 1)
            ]
            images = [volume[:, :, idx] for idx in indices]

        elif direction == "coronal":
            center = H // 2
            indices = [
                int(torch.clamp(torch.tensor(center + i * spacing), 0, H - 1))
                for i in range(-half, half + 1)
            ]
            images = [volume[:, idx, :] for idx in indices]
        elif direction == "sagittal":
            center = D // 2
            indices = [
                int(torch.clamp(torch.tensor(center + i * spacing), 0, D - 1))
                for i in range(-half, half + 1)
            ]
            images = [volume[idx, :, :] for idx in indices]
        slices[direction] = images

    return slices


class ValidationSnapshot(Callback):
    def __init__(self, output_dir, num_slices=5):
        super().__init__()

        self.num_slices = num_slices
        assert num_slices % 2 == 1, "num_slices must be odd"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, "visualization_data"):
            return

        data = pl_module.visualization_data
        if not data:
            return

        directions = ["axial", "coronal", "sagittal"]
        n_batches = len(data)
        n_rows = 3 + 3  # input/pred/target + 3 features
        n_cols = self.num_slices * n_batches

        fig, axes = plt.subplots(
            len(directions) * n_rows,
            self.num_slices * n_batches,
            figsize=(self.num_slices * n_batches, len(directions) * n_rows),
            gridspec_kw=dict(wspace=0.0, hspace=0.0),
        )
        if axes.ndim == 1:
            axes = axes.unsqueeze(0)

        for batch_idx, batch in enumerate(data):
            # Choose a random sample from the batch
            sample_idx = torch.randint(len(batch["inputs"]), (1,)).item()
            input_vol = batch["inputs"][sample_idx].squeeze()
            pred_vol = batch["pred_image"][sample_idx].squeeze()
            target_vol = batch["image"].squeeze()
            feat_vols = batch["pred_feat"][sample_idx][-1].squeeze()

            slices_dict = {
                "Input": get_slices(input_vol, directions, self.num_slices),
                "Prediction": get_slices(
                    pred_vol, directions, self.num_slices
                ),
                "Target": get_slices(target_vol, directions, self.num_slices),
                "Feature1": get_slices(
                    feat_vols[0], directions, self.num_slices
                ),
                "Feature2": get_slices(
                    feat_vols[1], directions, self.num_slices
                ),
                "Feature3": get_slices(
                    feat_vols[2], directions, self.num_slices
                ),
            }

            for d_idx, direction in enumerate(directions):
                for r_idx, key in enumerate(slices_dict.keys()):
                    row = d_idx * n_rows + r_idx
                    for s_idx, img in enumerate(slices_dict[key][direction]):
                        col = batch_idx * self.num_slices + s_idx
                        ax = axes[row, col]
                        ax.imshow(img.T.numpy(), cmap="gray", origin="lower")
                        ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(
            self.output_dir, f"val_epoch_{trainer.current_epoch}.png"
        )

        fig.savefig(save_path, dpi=150)

        plt.close(fig)


class ValidationFeaturesSnapshot(Callback):
    def __init__(self, output_dir, num_slices=5):
        super().__init__()

        self.num_slices = num_slices
        assert num_slices % 2 == 1, "num_slices must be odd"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, "visualization_data"):
            return

        data = pl_module.visualization_data
        if not data:
            return

        directions = ["axial", "coronal", "sagittal"]

        # Do a visualization of the features
        batch = data[0]
        sample_idx = torch.randint(len(batch["inputs"]), (1,)).item()
        input_vol = batch["inputs"][sample_idx].squeeze()
        if "pred_image" in batch:
            pred_vol = batch["pred_image"][sample_idx].squeeze()
        target_vol = batch["image"].squeeze()
        feat_vols = batch["pred_feat"][sample_idx][-1].squeeze()

        slices_dict = {
            "Input": get_slices(input_vol, directions, self.num_slices),
            "Prediction": (
                get_slices(pred_vol, directions, self.num_slices)
                if "pred_image" in batch
                else None
            ),
            "Target": get_slices(target_vol, directions, self.num_slices),
        }

        n_rows = 3 + 48
        if "pred_image" not in batch:
            del slices_dict["Prediction"]
            n_rows = 2 + 48
        for i in range(len(feat_vols)):
            slices_dict[f"Feature{i+1}"] = get_slices(
                feat_vols[i], directions, self.num_slices
            )

        fig, axes = plt.subplots(
            n_rows,
            self.num_slices * len(directions),
            figsize=(
                self.num_slices * len(directions),
                n_rows,
            ),
            gridspec_kw=dict(wspace=0.0, hspace=0.0),
        )

        for r_idx, key in enumerate(slices_dict):
            for d_idx, direction in enumerate(directions):
                row = r_idx
                for s_idx, img in enumerate(slices_dict[key][direction]):
                    col = d_idx * self.num_slices + s_idx
                    ax = axes[row, col]
                    ax.imshow(img.T.numpy(), cmap="gray", origin="lower")
                    ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(
            self.output_dir, f"feature_val_epoch_{trainer.current_epoch}.png"
        )
        fig.savefig(save_path, dpi=150)

        plt.close(fig)


class QCValidationSnapshot(Callback):
    def __init__(self, output_dir, num_slices=5, vis_feat=True):
        super().__init__()

        self.num_slices = num_slices
        assert num_slices % 2 == 1, "num_slices must be odd"
        self.output_dir = output_dir
        self.vis_feat = vis_feat
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, "visualization_data"):
            return

        data = pl_module.visualization_data
        if not data:
            return

        directions = ["axial", "coronal", "sagittal"]
        n_batches = len(data)
        n_rows = 5 if self.vis_feat else 2
        n_cols = self.num_slices * n_batches

        fig, axes = plt.subplots(
            len(directions) * n_rows,
            n_cols,
            figsize=(n_cols, len(directions) * n_rows),
            gridspec_kw=dict(wspace=0.0, hspace=0.0),
        )
        if axes.ndim == 1:
            axes = axes.unsqueeze(0)

        for batch_idx, batch in enumerate(data):
            # Choose a random sample from the batch
            sample_idx = torch.randint(len(batch["inputs"]), (1,)).item()
            input_vol = batch["inputs"][sample_idx].squeeze()
            pred = batch["pred_pred"][sample_idx].squeeze()
            pred = torch.softmax(pred, dim=0)[1]
            target_vol = batch["image"].squeeze()
            target = batch["label"].squeeze()[1]

            slices_dict = {
                "Input": get_slices(input_vol, directions, self.num_slices),
                "Target": get_slices(target_vol, directions, self.num_slices),
            }
            if self.vis_feat:
                feat_vols = batch["pred_feat"][sample_idx][-1].squeeze()
                slices_dict.update(
                    {
                        "Feature1": get_slices(
                            feat_vols[0], directions, self.num_slices
                        ),
                        "Feature2": get_slices(
                            feat_vols[1], directions, self.num_slices
                        ),
                        "Feature3": get_slices(
                            feat_vols[2], directions, self.num_slices
                        ),
                    }
                )

            for d_idx, direction in enumerate(directions):
                for r_idx, key in enumerate(slices_dict.keys()):
                    row = d_idx * n_rows + r_idx
                    mid_slice = (self.num_slices - 1) // 2
                    for s_idx, img in enumerate(slices_dict[key][direction]):
                        col = batch_idx * self.num_slices + s_idx
                        ax = axes[row, col]
                        ax.imshow(img.T.numpy(), cmap="gray", origin="lower")
                        # Remove ticks and ticklabels
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])

                        # Set title for middle slice
                        if row == 0 and s_idx == mid_slice:
                            ax.set_title(
                                f"Quality: {pred:.2f} (GT: {target})",
                                fontsize=8,
                            )

        plt.tight_layout()
        save_path = os.path.join(
            self.output_dir, f"qc_val_epoch_{trainer.current_epoch}.png"
        )

        fig.savefig(save_path, dpi=150)

        plt.close(fig)


class SegValidationSnapshot(Callback):
    def __init__(self, output_dir, num_slices=5):
        super().__init__()

        self.num_slices = num_slices
        assert num_slices % 2 == 1, "num_slices must be odd"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, "visualization_data"):
            return

        data = pl_module.visualization_data
        if not data:
            return

        directions = ["axial", "coronal", "sagittal"]
        n_batches = len(data)
        n_rows = 7
        n_cols = self.num_slices * n_batches

        fig, axes = plt.subplots(
            len(directions) * n_rows,
            n_cols,
            figsize=(n_cols, len(directions) * n_rows),
            gridspec_kw=dict(wspace=0.0, hspace=0.0),
        )
        if axes.ndim == 1:
            axes = axes.unsqueeze(0)

        for batch_idx, batch in enumerate(data):
            # Choose a random sample from the batch
            sample_idx = torch.randint(len(batch["inputs"]), (1,)).item()
            input_vol = batch["inputs"][sample_idx].squeeze()
            pred_seg = batch["pred_seg"][sample_idx].squeeze()

            pred_seg = torch.argmax(pred_seg, dim=0)
            target_vol = batch["image"].squeeze()
            target_seg = batch["label"].squeeze()
            feat_vols = batch["pred_feat"][sample_idx][-1].squeeze()

            slices_dict = {
                "Input": get_slices(input_vol, directions, self.num_slices),
                "Target": get_slices(target_vol, directions, self.num_slices),
                "Seg": get_slices(pred_seg, directions, self.num_slices),
                "GT seg": get_slices(target_seg, directions, self.num_slices),
                "Feature1": get_slices(
                    feat_vols[0], directions, self.num_slices
                ),
                "Feature2": get_slices(
                    feat_vols[1], directions, self.num_slices
                ),
                "Feature3": get_slices(
                    feat_vols[2], directions, self.num_slices
                ),
            }

            for d_idx, direction in enumerate(directions):
                for r_idx, key in enumerate(slices_dict.keys()):
                    row = d_idx * n_rows + r_idx
                    for s_idx, img in enumerate(slices_dict[key][direction]):
                        col = batch_idx * self.num_slices + s_idx
                        ax = axes[row, col]
                        cmap = "gray" if "seg" not in key.lower() else "tab10"
                        ax.imshow(img.T.numpy(), cmap=cmap, origin="lower")
                        ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(
            self.output_dir, f"seg_val_epoch_{trainer.current_epoch}.png"
        )

        fig.savefig(save_path, dpi=150)

        plt.close(fig)
