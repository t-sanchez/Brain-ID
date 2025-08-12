import time
import torch
import numpy as np
import pandas as pd
import os
from monai.transforms import (
    Compose,
    ScaleIntensity,
    Orientation,
    CropForegroundd,
    SpatialPad,
    CenterSpatialCrop,
)
from fetalsyngen.utils.image_reading import SimpleITKReader
from fetalsyngen.generator.model import FetalSynthGen

import os


class FetalScalarDataset:
    """Abstract class defining a dataset for loading fetal data for a scalar task."""

    def __init__(
        self,
        df: str | pd.DataFrame,
        target_key: str = "qcglobal",
        generator: FetalSynthGen | None = None,
        split: str | None = None,
        load_key: str = "im",
        augment_deform=True,
        augment_artifact=False,
        transform_target: str = "binarize",
        target_threshold: float = 1.0,
        return_iqms: bool = False,
        b1: float = 6.0,
        b2: float = 3.0,
        use_seg: bool = False,
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
        self.augment_deform = augment_deform
        self.augment_artifact = augment_artifact
        if self.augment_deform or self.augment_artifact:
            assert (
                self.generator is not None
            ), "If augment_deform or augment_artifact is True, generator must be provided."
        self.split = split
        self.target_key = target_key
        self.load_key = load_key
        self.use_seg = use_seg
        self.return_iqms = return_iqms
        assert (
            load_key in self.df.columns
        ), f"load_key {load_key} not in df columns"
        assert (
            target_key in self.df.columns
        ), f"target_key {target_key} not in df columns"

        if self.use_seg:
            assert "seg" in self.df.columns, f"'seg' not in df columns with use_seg={self.use_seg}"
        if split is not None:
            assert "split" in self.df.columns, "'split' not in df columns"
            self.df = self.df[self.df["split"] == split]
        if self.return_iqms:
            self.iqms_min = None
            self.iqms_max = None
            self.compute_iqms()

        self._check_images_exist()
        print(f"FetalScalarDataset: {len(self.df)} subjects.")

        self.loader = SimpleITKReader()
        self.scaler = ScaleIntensity(minv=0, maxv=1)
        self.orientation = Orientation(axcodes="RAS")

        self.transform_target = transform_target
        self.target_threshold = target_threshold
        self.b1 = b1
        self.b2 = b2
        assert self.transform_target in [
            "binarize",
            "soft_binarize",
        ], "transform_target must be either 'binarize' or 'soft_binarize'"

        self.base_transforms = Compose(
            [
                CropForegroundd(
                    keys=["im", "seg"],
                    source_key="im",
                    allow_smaller=True,
                    margin=5,
                    allow_missing_keys=True
                ),
                # SpatialPad(
                #     spatial_size=(256, 256, 256),
                #     mode="constant",
                # ),
                # CenterSpatialCrop(
                #     roi_size=(256, 256, 256),
                # ),
            ]
        )

    def set_min_max_iqms(self, min_iqms, max_iqms):
        self.iqms_min = min_iqms
        self.iqms_max = max_iqms
        self.compute_iqms()

    def compute_iqms(self):
        first_idx = self.df.columns.get_loc("centroid")
        last = self.df.columns.get_loc("seg_topology_mask_ec_nan")
        self.iqms = self.df.iloc[:, first_idx:last + 1]
        cols_train = [c for c in self.iqms.columns if "_nan" not in c]
        self.iqms = self.iqms[cols_train]
        self.iqms_min = self.iqms.min().values if self.iqms_min is None else self.iqms_min
        self.iqms_max = self.iqms.max().values if self.iqms_max is None else self.iqms_max
        self.iqms = (self.iqms - self.iqms.min()) / (self.iqms.max() - self.iqms.min() + 1e-8)

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
            label_trf = self.asym_sigmoid(label, a=self.target_threshold, b1=self.b1, b2=self.b2)
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
        data = {"im": img.squeeze(0)}
        if self.use_seg:
            seg = self.loader(row["seg"])
            seg = seg.unsqueeze(0)
            seg = self.orientation(seg)
            data["seg"] = seg.squeeze(0)

        data = self.base_transforms(data)
        # affine and orig
        
        img = data["im"]
        aff = img.affine
        # If generator is used (likely on GPU)
        img_orig = None
        if not (self.augment_deform or self.augment_artifact):
            img_orig = img.clone()
        
        synth_params = {}
        if self.augment_deform:
            # Only does spatial deformation
            img = img.to(self.generator.device)
            if self.use_seg:
                seg = data["seg"].to(self.generator.device)
            else:
                seg = img > 0
            img, seg, _, params = self.generator.generate(
                seeds=None, image=img, segmentation=seg
            )

            img_orig = img.clone().detach().to("cpu", non_blocking=True)
            img = img.detach().to("cpu", non_blocking=True)
            if self.use_seg:
                seg = seg.detach().to("cpu", non_blocking=True)
            synth_params.update(params)
        if self.augment_artifact:
            img = img.to(self.generator.device)
            # Only does artifact generation
            img_orig = img.clone() if img_orig is None else img_orig
            img, params = self.generator.augment(
                image=img, segmentation=img > 0
            )
            img = img.detach().to("cpu", non_blocking=True)            
            synth_params.update(params)
        if self.augment_artifact or self.augment_deform:
            img_orig = self.scaler(img_orig)
            img = self.scaler(img)
        # Transform the target
        label_trf = self.scale_label(label).detach()

        data = {
            "input": img.unsqueeze(0).contiguous(),
            "label": label_trf.contiguous(),
            "path": img_path,
            "image": img_orig.unsqueeze(0).contiguous(),
            "affine": aff.unsqueeze(0).contiguous(),
        }

        if self.use_seg:
            data["seg"] = seg.detach().cpu().unsqueeze(0).contiguous()
        # Save input
        if self.return_iqms:
            iqms = self.iqms.iloc[idx]
            data["iqms"] = torch.tensor(iqms.values, dtype=torch.float32).unsqueeze(0)
            data["iqms_min"] = torch.tensor(self.iqms_min, dtype=torch.float32).unsqueeze(0)
            data["iqms_max"] = torch.tensor(self.iqms_max, dtype=torch.float32).unsqueeze(0)
        if self.generator is not None:
            data["synth_params"] = (
                synth_params  # Should be small dict/float, no memory risk
            )

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
