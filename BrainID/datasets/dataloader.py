import lightning as L
from torch.utils.data import DataLoader
from BrainID.datasets.fetal_id_synth import (
    BrainIDFetalSynthDataset,
    BlockRandomSampler,
    RandomBlockPatchFetalDataset,
    FetalSynthIDDataset,
)
from BrainID.datasets.fetal_scalar import FetalScalarDataset
from fetalsyngen.data.datasets import FetalTestDataset, FetalSynthDataset
from fetalsyngen.generator.model import FetalSynthGen
from torch.utils.data import RandomSampler

import pandas as pd
import monai
import logging
import torch
from torch.utils.data import WeightedRandomSampler


class FeatureDataModule(L.LightningDataModule):

    def __init__(
        self,
        split_file: str,
        bids_path: str,
        seed_path: str,
        train_type: str,
        train_split: str,
        val_type: str,
        val_split: str,
        test_split: str,
        n_all_samples: int,
        n_mild_samples: int,
        n_severe_samples: int,
        generator_mild: FetalSynthGen | None,
        generator_severe: FetalSynthGen | None,
        transforms: monai.transforms.Compose,
        num_workers: int = 1,
        batch_size: int = 1,
        train_patch: bool = False,
        patch_size: int = 128,
        patch_boundary: int = 20,
        patch_per_subject: int = 3,
        mask_image: bool = False,
    ):
        super().__init__()
        self.split_file = split_file
        self.bids_path = bids_path
        self.seed_path = seed_path
        self.train_type = train_type
        self.train_split = train_split
        assert n_all_samples == n_mild_samples + n_severe_samples
        self.n_mild_samples = n_mild_samples
        self.n_severe_samples = n_severe_samples
        self.val_type = val_type
        self.val_split = val_split
        self.test_split = test_split
        self.generator_mild = generator_mild
        self.generator_severe = generator_severe
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.transform = transforms

        self.train_patch = train_patch
        self.patch_size = patch_size
        self.patch_boundary = patch_boundary
        self.patch_per_subject = patch_per_subject

        self.mask_image = mask_image

        assert self.train_type in ["synth", "real"]
        assert self.val_type in ["synth", "real", "test"]

        self.train_subjects, self.val_subjects, self.test_subjects = (
            self.get_subjects()
        )
        # Initialize datasets
        if self.train_type == "synth":

            assert self.generator_mild is not None
            assert self.generator_severe is not None

            self.train_ds = BrainIDFetalSynthDataset(
                bids_path=self.bids_path,
                seed_path=self.seed_path,
                sub_list=self.train_subjects,
                n_mild_samples=self.n_mild_samples,
                n_severe_samples=self.n_severe_samples,
                load_image=True,
                image_as_intensity=False,
                generator_mild=self.generator_mild,
                generator_severe=self.generator_severe,
                mask_image=self.mask_image,
            )
        elif self.train_type == "real":
            self.train_ds = FetalSynthDataset(
                bids_path=self.bids_path,
                seed_path=None,
                sub_list=self.train_subjects,
                load_image=True,
                image_as_intensity=True,
                generator=self.generator_mild,
            )

        if self.val_type == "synth":
            assert self.generator_mild is not None
            assert self.generator_severe is not None

            self.val_ds = BrainIDFetalSynthDataset(
                bids_path=self.bids_path,
                seed_path=self.seed_path,
                sub_list=self.val_subjects,
                n_mild_samples=self.n_mild_samples,
                n_severe_samples=self.n_severe_samples,
                load_image=True,
                image_as_intensity=False,
                generator_mild=self.generator_mild,
                generator_severe=self.generator_severe,
                mask_image=self.mask_image,
            )
        elif self.val_type == "real":
            self.val_ds = FetalSynthDataset(
                bids_path=self.bids_path,
                seed_path=None,
                sub_list=self.val_subjects,
                load_image=True,
                image_as_intensity=True,
                generator=self.generator_mild,
            )
        elif self.val_type == "test":
            self.val_ds = FetalTestDataset(
                bids_path=self.bids_path,
                sub_list=self.val_subjects,
                transforms=self.transform,
            )

        self.test_ds = None
        if self.test_split is not None:
            self.test_ds = FetalTestDataset(
                bids_path=self.bids_path,
                sub_list=self.test_subjects,
                transforms=self.transform,
            )

        if self.train_patch:
            self.train_ds = RandomBlockPatchFetalDataset(
                dataset=self.train_ds,
                patch_size=self.patch_size,
                boundary=self.patch_boundary,
                patch_per_subject=self.patch_per_subject,
            )
        # log dataset size
        logging.info(f"Train dataset size: {len(self.train_ds)}")
        logging.info(f"Val dataset size: {len(self.val_ds)}")
        logging.info(
            f"Test dataset size: {len(self.test_ds) if self.test_ds else 0}"
        )

    def get_subjects(self):
        split_df = pd.read_csv(self.split_file)
        assert "participant_id" in split_df.columns
        assert "splits" in split_df.columns

        train_subjects = sorted(
            split_df[
                split_df.splits == self.train_split
            ].participant_id.tolist()
        )

        val_subjects = sorted(
            split_df[split_df.splits == self.val_split].participant_id.tolist()
        )

        test_subjects = sorted(
            split_df[
                split_df.splits == self.test_split
            ].participant_id.tolist()
        )

        return train_subjects, val_subjects, test_subjects

    def collate(self, batch):
        if len(batch) > 1:
            raise ValueError(
                "Batch size > 1 is not supported. Please set batch size to 1."
            )
        batch = batch[0]

        batch = {
            k: v.unsqueeze(0) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        batch["input"] = [x.unsqueeze(0) for x in batch["input"]]
        return batch

    def train_dataloader(self):
        sampler = RandomSampler(self.train_ds)
        if self.train_patch:
            sampler = BlockRandomSampler(self.train_ds, self.patch_per_subject)

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            sampler=sampler,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            pin_memory=False,
            timeout=120 if self.num_workers > 0 else 0,
            # persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=False,
            timeout=120 if self.num_workers > 0 else 0,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            raise ValueError("Test dataset is not defined.")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            timeout=120 if self.num_workers > 0 else 0,
        )


class QCDataModule(L.LightningDataModule):

    def __init__(
        self,
        df: str | pd.DataFrame,
        generator: FetalSynthGen,
        train_split: str,
        val_split: str,
        test_split: str,
        transforms: monai.transforms.Compose,
        load_key: str = "im",
        target_key: str = "qcglobal",
        transform_target: str = "binarize",
        target_threshold: float = 1.0,
        b1: float = 6.0,
        b2: float = 3.0,
        num_workers: int = 1,
        batch_size: int = 1,
        reweight_train: bool = False,
        train_patch: bool = False,
        patch_size: int = 128,
        patch_boundary: int = 20,
        patch_per_subject: int = 3,
    ):
        super().__init__()
        self.df = df
        self.generator = generator
        self.target_key = target_key
        self.load_key = load_key
        self.transform_target = transform_target
        self.target_threshold = target_threshold
        self.b1 = b1
        self.b2 = b2
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.transforms = transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.reweight_train = reweight_train
        self.train_patch = train_patch
        self.patch_size = patch_size
        self.patch_boundary = patch_boundary
        self.patch_per_subject = patch_per_subject

        self.train_ds = FetalScalarDataset(
            df=self.df,
            target_key=self.target_key,
            generator=self.generator,
            split=self.train_split,
            load_key=self.load_key,
            transform_target=self.transform_target,
            target_threshold=self.target_threshold,
            b1=self.b1,
            b2=self.b2,
        )

        self.val_ds = FetalScalarDataset(
            df=self.df,
            target_key=self.target_key,
            generator=self.generator,
            split=self.val_split,
            load_key=self.load_key,
            transform_target=self.transform_target,
            target_threshold=self.target_threshold,
            b1=self.b1,
            b2=self.b2,
        )

        self.test_ds = FetalScalarDataset(
            df=self.df,
            target_key=self.target_key,
            generator=self.generator,
            split=self.test_split,
            load_key=self.load_key,
            transform_target=self.transform_target,
            target_threshold=self.target_threshold,
            b1=self.b1,
            b2=self.b2,
        )

        if self.train_patch:
            self.train_ds = RandomBlockPatchFetalDataset(
                dataset=self.train_ds,
                patch_size=self.patch_size,
                boundary=self.patch_boundary,
                patch_per_subject=self.patch_per_subject,
            )

    def collate(self, batch):
        if len(batch) > 1:
            raise ValueError(
                "Batch size > 1 is not supported. Please set batch size to 1."
            )
        batch = batch[0]

        batch = {
            k: v.unsqueeze(0) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        batch["input"] = [x.unsqueeze(0) for x in batch["input"]]
        return batch

    def train_dataloader(self):
        if self.reweight_train:
            labels = self.train_ds.get_labels()
            # If labels are binary: weight each class equally
            if len(torch.unique(labels)) == 2:
                # labels is a torch tensor
                labels = labels.tolist()
                n_1 = labels.count(1)
                n_0 = labels.count(0)
                weights = [
                    1.0 / n_1 if label == 1 else 1.0 / n_0 for label in labels
                ]
            else:
                # Do a histogram and equalize the weights
                hist = torch.histc(torch.tensor(labels), bins=20)
                hist = hist / hist.sum()
                weights = [1.0 / hist[int(label)] for label in labels]
            sampler = WeightedRandomSampler(
                weights, len(self.train_ds), replacement=True
            )
        else:
            sampler = None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            num_workers=self.num_workers,
            sampler=sampler,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            # timeout=20 if self.num_workers > 0 else 0,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            num_workers=self.num_workers,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            pin_memory=False,
            # timeout=20 if self.num_workers > 0 else 0,
            persistent_workers=False,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            raise ValueError("Test dataset is not defined.")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            num_workers=self.num_workers,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            pin_memory=False,
            # timeout=20 if self.num_workers > 0 else 0,
            persistent_workers=False,
        )


class SegDataModule(L.LightningDataModule):

    def __init__(
        self,
        split_file: str,
        bids_path: str,
        seed_path: str,
        train_type: str,
        train_split: str,
        val_type: str,
        val_split: str,
        test_split: str,
        generator: FetalSynthGen | None,
        transforms: monai.transforms.Compose,
        num_workers: int = 1,
        batch_size: int = 1,
        train_patch: bool = False,
        patch_size: int = 128,
        patch_boundary: int = 20,
        patch_per_subject: int = 2,
    ):
        super().__init__()
        self.split_file = split_file
        self.bids_path = bids_path
        self.seed_path = seed_path
        self.train_type = train_type
        self.train_split = train_split
        self.val_type = val_type
        self.val_split = val_split
        self.test_split = test_split
        self.generator = generator
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.transform = transforms
        self.train_patch = train_patch
        self.patch_size = patch_size
        self.patch_boundary = patch_boundary
        self.patch_per_subject = patch_per_subject

        assert self.train_type in ["synth", "real"]
        assert self.val_type in ["synth", "real", "test"]

        self.train_subjects, self.val_subjects, self.test_subjects = (
            self.get_subjects()
        )
        # Initialize datasets
        if self.train_type == "synth":

            assert self.generator is not None

            self.train_ds = FetalSynthIDDataset(
                bids_path=self.bids_path,
                seed_path=self.seed_path,
                sub_list=self.train_subjects,
                load_image=True,
                image_as_intensity=False,
                generator=self.generator,
            )
        elif self.train_type == "real":
            self.train_ds = FetalSynthIDDataset(
                bids_path=self.bids_path,
                seed_path=None,
                sub_list=self.train_subjects,
                load_image=True,
                image_as_intensity=True,
                generator=self.generator,
            )

        if self.val_type == "synth":
            assert self.generator is not None
            self.val_ds = FetalSynthIDDataset(
                bids_path=self.bids_path,
                seed_path=self.seed_path,
                sub_list=self.val_subjects,
                load_image=True,
                image_as_intensity=False,
                generator=self.generator,
            )
        elif self.val_type == "real":
            self.val_ds = FetalSynthIDDataset(
                bids_path=self.bids_path,
                seed_path=None,
                sub_list=self.val_subjects,
                load_image=True,
                image_as_intensity=True,
                generator=self.generator,
            )
        elif self.val_type == "test":
            self.val_ds = FetalTestDataset(
                bids_path=self.bids_path,
                sub_list=self.val_subjects,
                transforms=self.transform,
            )

        self.test_ds = FetalTestDataset(
            bids_path=self.bids_path,
            sub_list=self.test_subjects,
            transforms=self.transform,
        )
        # log dataset size
        logging.info(f"Train dataset size: {len(self.train_ds)}")
        logging.info(f"Val dataset size: {len(self.val_ds)}")
        logging.info(f"Test dataset size: {len(self.test_ds)}")

        if self.train_patch:
            self.train_ds = RandomBlockPatchFetalDataset(
                dataset=self.train_ds,
                patch_size=self.patch_size,
                boundary=self.patch_boundary,
                patch_per_subject=self.patch_per_subject,
            )

    def get_subjects(self):
        split_df = pd.read_csv(self.split_file)
        assert "participant_id" in split_df.columns
        assert "splits" in split_df.columns

        train_subjects = split_df[
            split_df.splits == self.train_split
        ].participant_id.tolist()

        val_subjects = split_df[
            split_df.splits == self.val_split
        ].participant_id.tolist()

        test_subjects = split_df[
            split_df.splits == self.test_split
        ].participant_id.tolist()

        return train_subjects, val_subjects, test_subjects

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
        )
