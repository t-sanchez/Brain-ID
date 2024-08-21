from typing import (
    Iterator,
    List,
    Sized,
)
import torch
from fetalsynthgen.definitions import GeneratorParams
from fetalsynthgen.dataset import SynthDataset
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.sampler import Sampler


class FetalIDSynth(SynthDataset):

    def __init__(
        self,
        bids_path: str,
        subjects: list[str] | None = None,
        transforms=None,
        # keep it for consistency with SynthDataset
        rescale_res: float = 0.5,
        generator_params: GeneratorParams | None = None,
        seed_path: str | None = None,
        segm_path: str | None = None,
        device: str = "cuda",
    ):

        super().__init__(
            bids_path,
            subjects,
            transforms,
            rescale_res,
            generator_params,
            seed_path,
            segm_path,
            device,
        )

    def __getitem__(self, idx):

        # 1. Sample random meta label subclasses number, if needed
        seeds = self.seed_paths[self._sub_ses_idx(idx)]

        with torch.inference_mode():

            segmentation = self.segm_paths[idx]
            image = self.im_paths[idx]
            # define extras (segmentations to be deformed together with seeds)
            extras = {
                "nearest": [segmentation],
                "linear": [image],
            }  # + [seeds] if needed for debug

            affine = segmentation.affine
            shape = segmentation.shape

            synth_image, extras = self.generator.generate(shape, seeds, extras)
            shape = [1, self.nchannels, *synth_image.shape[-3:]]

            data = {
                "image": synth_image.view(shape),
                "label": extras["nearest"][0].view(shape),
                "gt": extras["linear"][0].view(shape),
                "name": self.subjects[idx],
            }
            # fill nans with zeros
            data = self.filler(data)
            data = self.scaler(data)
            data["label"] = data["label"].long()

            samples = []
            for samp in range(self.nchannels):
                sample_dict = {
                    "image_def": extras["linear"][0][None, samp],
                    "input": synth_image[None, samp],
                }
                samples.append(sample_dict)

            device = synth_image.device
            subjects = {
                "name": self.subjects[idx],
                "image": torch.tensor(image.get_fdata(), device=device).view(
                    1, *shape[-3:]
                ),
                "aff": affine,
                "shp": torch.tensor(segmentation.shape).to(device),
            }
            # fill nans with zeros

        return subjects, samples


class RandomBlockPatchFetalDataset(Dataset):
    def __init__(
        self,
        dataset,
        patch_size: int,
        boundary: int = 0,
        patch_per_subject: int = 1,
    ):
        super().__init__()
        self.dataset = dataset
        assert isinstance(
            dataset, FetalIDSynth
        ), "Dataset must be FetalIDSynth"
        self.patch_size = patch_size
        self.boundary = boundary
        self.patch_per_subject = patch_per_subject
        self.idx_dataset = None

    def random_patch(self, shape):
        """
        Get a random slice of length self.patch_size
        from the shape, within the bounds of the shape.
        """
        assert all(
            self.boundary * 2 + self.patch_size < s for s in shape
        ), f"Patch size + 2*boundary ({self.boundary*2+self.patch_size}) is larger than image size ({shape})"
        max_start = [s - self.patch_size - self.boundary for s in shape]

        # Get a random start index for each dimension
        # Return the slice
        slice_idx = []
        for max_ in max_start:
            idx = np.random.randint(self.boundary, max_)
            slice_idx.append((idx, idx + self.patch_size))
        return slice_idx

    def get_sub_samp(self, idx):
        if idx != self.idx_dataset:
            subjects, samples = self.dataset[idx]
            self.idx_dataset = idx
            self.subjects = subjects
            self.samples = samples
        return self.subjects, self.samples

    def __len__(self):
        return len(self.dataset) * self.patch_per_subject

    def __getitem__(self, idx):

        idx_dataset = idx // self.patch_per_subject
        subjects, samples = self.get_sub_samp(idx_dataset)

        im_size = subjects["image"].shape[-3:]
        slice_idx = self.random_patch(im_size)
        slice_ = [slice(*idx) for idx in slice_idx]
        samples_patch = []
        subjects_patch = {
            "name": subjects["name"],
            "aff": subjects["aff"],
            "slice_patch": slice_idx,
        }
        for sample in samples:

            sample_dict = {
                "image_def": sample["image_def"][
                    :, slice_[0], slice_[1], slice_[2]
                ],
                "input": sample["input"][:, slice_[0], slice_[1], slice_[2]],
            }
            samples_patch.append(sample_dict)
        subjects_patch["image"] = subjects["image"][
            :, slice_[0], slice_[1], slice_[2]
        ]
        subjects_patch["shp"] = torch.tensor(subjects_patch["image"].shape)
        subjects_patch["shp_init"] = torch.tensor(subjects["image"].shape)
        return subjects_patch, samples_patch


class BlockRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        block_size (int): size of the block to sample from
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        block_size: int = 1,
    ) -> None:
        self.data_source = data_source
        self.block_size = block_size

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        num_blocks = n // self.block_size
        block_indices = torch.randperm(num_blocks, generator=generator)[
            : self.num_samples // self.block_size
        ]

        for block_idx in block_indices.tolist():
            start_idx = block_idx * self.block_size
            for i in range(start_idx, start_idx + self.block_size):
                if i < n:
                    yield i

    def __len__(self) -> int:
        return self.num_samples
