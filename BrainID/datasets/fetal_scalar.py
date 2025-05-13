import time
import torch
import numpy as np
import pandas as pd
import os
from monai.transforms import (
    Compose,
    ScaleIntensity,
    Orientation,
    CropForeground,
    SpatialPad,
    CenterSpatialCrop,
)
from fetalsyngen.utils.image_reading import SimpleITKReader
from fetalsyngen.generator.model import FetalSynthGen

import os


def print_open_fds():
    pid = os.getpid()
    num_fds = len(os.listdir(f"/proc/{pid}/fd"))
    print(f"[PID {pid}] Open file descriptors: {num_fds}")


class FetalScalarDataset:
    """Abstract class defining a dataset for loading fetal data for a scalar task."""

    def __init__(
        self,
        df: str | pd.DataFrame,
        target_key: str = "qcglobal",
        generator: FetalSynthGen | None = None,
        split: str | None = None,
        load_key: str = "im",
        transform_target: str = "binarize",
        target_threshold: float = 1.0,
    ) -> dict:
        # ToDo: transforms for DA and generator for deformation + also scalar transform
        """
        Args:
            scalar_df: Path to the csv file with the scalar data.
                If a DataFrame is provided, it will be used directly.
                It should have the following columns:
                `sub`, `ses`, `target_key` and `split` (if split is not None).
            split: Split to use. Should be one of `train`, `val` or `test`. If None,
                all data is used.

        """
        super().__init__()

        if isinstance(df, str):
            df = pd.read_csv(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "df should be a path to a csv file or a pandas DataFrame."
            )
        self.df = df

        self.generator = generator
        self.split = split
        self.target_key = target_key
        self.load_key = load_key

        assert (
            load_key in self.df.columns
        ), f"load_key {load_key} not in df columns"
        assert (
            target_key in self.df.columns
        ), f"target_key {target_key} not in df columns"
        if split is not None:
            assert "split" in self.df.columns, "'split' not in df columns"
            self.df = self.df[self.df["split"] == split]
        self._check_images_exist()
        print(f"FetalScalarDataset: {len(self.df)} subjects.")

        self.loader = SimpleITKReader()
        self.scaler = ScaleIntensity(minv=0, maxv=1)
        self.orientation = Orientation(axcodes="RAS")

        self.transform_target = transform_target
        self.target_threshold = target_threshold
        assert self.transform_target in [
            "binarize",
            "soft_binarize",
        ], "transform_target must be either 'binarize' or 'soft_binarize'"

        self.base_transforms = Compose(
            [
                CropForeground(
                    allow_smaller=True,
                    margin=0,
                ),
                SpatialPad(
                    spatial_size=(256, 256, 256),
                    mode="constant",
                ),
                CenterSpatialCrop(
                    roi_size=(256, 256, 256),
                ),
            ]
        )

    def asym_sigmoid(self, x, a=1.0, b1=6, b2=3):
        if x < a:
            return 1 / (1 + torch.exp(-b1 * (x - a)))
        else:
            return 1 / (1 + torch.exp(-b2 * (x - a)))

    def scale_label(self, label):
        if self.transform_target == "binarize":
            label_trf = (label > self.target_threshold).type(torch.float32)
            label_trf = torch.tensor([1 - label_trf, label_trf])
        elif self.transform_target == "soft_binarize":
            # Apply asym sigmoid in a vectorized way
            label_trf = self.asym_sigmoid(label, a=self.target_threshold)
            label_trf = torch.tensor([1 - label_trf, label_trf])
        return label_trf

    def _check_images_exist(self):
        """Check if the images exist and are valid."""
        for idx, row in self.df.iterrows():
            img_path = row[self.load_key]
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f"Image {img_path} does not exist (idx {idx})."
                )

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        """Get the labels for the dataset."""
        labels = torch.tensor(self.df[self.target_key].values)
        # apply scale_label in a vectorized way

        for i in range(len(labels)):
            labels[i] = self.scale_label(labels[i])[1].item()
        return labels

    # def __getitem__(self, idx):
    #     """Get the item at the given index."""
    #     row = self.df.iloc[idx]
    #     img_path = row[self.load_key]
    #     label = torch.tensor(row[self.target_key])

    #     # Load the image
    #     img = self.loader(img_path)

    #     img = img.unsqueeze(0)
    #     img = self.scaler(img)
    #     img = self.orientation(img)
    #     img = self.base_transforms(img).squeeze(0)

    #     img_orig = img.clone()
    #     if self.generator is not None:
    #         img, _, _, synth_params = self.generator.generate(
    #             seeds=None, image=img, segmentation=img > 0
    #         )

    #     # Transform the target
    #     label_trf = self.scale_label(label)

    #     data = {
    #         "input": img.cpu().unsqueeze(0),
    #         "label": label_trf.cpu(),
    #     }
    #     if self.generator is not None:
    #         data["image"] = img_orig.cpu().unsqueeze(0)
    #         data["synth_params"] = synth_params
    #     print_open_fds()
    #     return data
    
    def __getitem__(self, idx):
        """Get the item at the given index."""
        row = self.df.iloc[idx]
        img_path = row[self.load_key]
        label = torch.tensor(row[self.target_key])

        # Load the image
        img = self.loader(img_path)

        img = img.unsqueeze(0)
        img = self.scaler(img)
        img = self.orientation(img)
        img = self.base_transforms(img).squeeze(0)

        # If generator is used (likely on GPU)
        if self.generator is not None:
            img, _, _, synth_params = self.generator.generate(
                seeds=None, image=img, segmentation=img > 0
            )
            img_orig = img.clone().detach().to("cpu", non_blocking=True)
            img = img.detach().to("cpu", non_blocking=True)
        else:
            img_orig = img.clone().detach()
            img = img.detach()

        # Transform the target
        label_trf = self.scale_label(label).detach()

        data = {
            "input": img.unsqueeze(0).contiguous(),
            "label": label_trf.contiguous(),
        }

        if self.generator is not None:
            data["image"] = img_orig.unsqueeze(0).contiguous()
            data["synth_params"] = synth_params  # Should be small dict/float, no memory risk
        print_open_fds()
        return data

class FetalSegmDataset:
    """Abstract class defining a dataset for loading fetal data for a segmentation task."""

    def __init__(
        self,
        df: str | pd.DataFrame,
        generator: FetalSynthGen | None = None,
        split: str | None = None,
        load_key: str = "im",
    ) -> dict:
        # ToDo: transforms for DA and generator for deformation + also scalar transform
        """
        Args:
            scalar_df: Path to the csv file with the scalar data.
                If a DataFrame is provided, it will be used directly.
                It should have the following columns:
                `sub`, `ses`, `target_key` and `split` (if split is not None).
            split: Split to use. Should be one of `train`, `val` or `test`. If None,
                all data is used.

        """
        super().__init__()

        if isinstance(df, str):
            df = pd.read_csv(df)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "df should be a path to a csv file or a pandas DataFrame."
            )
        self.df = df

        self.generator = generator
        self.split = split
        if split is not None:
            assert split in self.df.columns, f"split {split} not in df columns"
            self.df = self.df[self.df[split] == split]
        self._check_images_exist()
        print(f"FetalScalarDataset: {len(self.df)} subjects.")

        self.loader = SimpleITKReader()
        self.scaler = ScaleIntensity(minv=0, maxv=1)
        self.orientation = Orientation(axcodes="RAS")

        self.base_transforms = Compose(
            [
                CropForeground(
                    allow_smaller=True,
                    margin=0,
                ),
                SpatialPad(
                    spatial_size=(256, 256, 256),
                    mode="constant",
                ),
                CenterSpatialCrop(
                    roi_size=(256, 256, 256),
                ),
            ]
        )

    def asym_sigmoid(self, x, a=1.0, b1=6, b2=3):
        if x < a:
            return 1 / (1 + torch.exp(-b1 * (x - a)))
        else:
            return 1 / (1 + torch.exp(-b2 * (x - a)))

    def scale_target(self, target):
        if self.transform_target == "binarize":
            target_trf = (target > self.target_threshold).type(torch.float32)
            target_trf = torch.tensor([1 - target_trf, target_trf])
        elif self.transform_target == "soft_binarize":
            # Apply asym sigmoid in a vectorized way
            target_trf = self.asym_sigmoid(target, a=self.target_threshold)
            target_trf = torch.tensor([1 - target_trf, target_trf])
        return target_trf.unsqueeze(0)

    def _check_images_exist(self):
        """Check if the images exist and are valid."""
        for idx, row in self.df.iterrows():
            img_path = row[self.load_key]
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f"Image {img_path} does not exist (idx {idx})."
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get the item at the given index."""
        row = self.df.iloc[idx]
        img_path = row[self.load_key]
        target = torch.tensor(row[self.target_key])

        # Load the image
        img = self.loader(img_path)

        img = img.unsqueeze(0)
        img = self.scaler(img)
        img = self.orientation(img)
        img = self.base_transforms(img).squeeze(0)

        img_orig = img.clone()
        if self.generator is not None:
            img, _, _, synth_params = self.generator.generate(
                seeds=None, image=img, segmentation=img > 0
            )

        # Transform the target
        target_trf = self.scale_target(target)

        data = {
            "input": img.cpu().unsqueeze(0),
            "target": target_trf.cpu(),
        }
        if self.generator is not None:
            data["input_orig"] = img_orig.cpu().unsqueeze(0)
            data["synth_params"] = synth_params
        return data
