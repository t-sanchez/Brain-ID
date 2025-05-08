from typing import (
    Iterator,
    List,
    Sized,
)
import torch
# from fetalsynthgen.definitions import GeneratorParams
# from fetalsynthgen.dataset import FetalBIDSDataset, SynthDataset
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.sampler import Sampler
from pathlib import Path
from fetalsyngen.data.datasets import FetalSynthDataset
from fetalsyngen.generator.model import FetalSynthGen
import time
from collections import defaultdict


class BrainIDFetalSynthDataset(FetalSynthDataset):
    """Dataset class for generating/augmenting on-the-fly fetal images" """

    def __init__(
        self,
        bids_path: str,
        generator_mild: FetalSynthGen,
        generator_severe: FetalSynthGen,
        n_mild_samples: int,
        n_severe_samples: int,
        seed_path: str | None,
        sub_list: list[str] | None,
        load_image: bool = False,
        image_as_intensity: bool = False,
        mask_image: bool = False,
        
    ):
        """

        Args:
            bids_path: Path to the bids-formatted folder with the data.
            seed_path: Path to the folder with the seeds to use for
                intensity sampling. See `scripts/seed_generation.py`
                for details on the data formatting. If seed_path is None,
                the intensity  sampling step is skipped and the output image
                intensities will be based on the input image.
            generator: a class object defining a generator to use.
            sub_list: List of the subjects to use. If None, all subjects are used.
            load_image: If **True**, the image is loaded and passed to the generator,
                where it can be used as the intensity prior instead of a random
                intensity sampling or spatially deformed with the same transformation
                field as segmentation and the syntehtic image. Default is **False**.
            image_as_intensity: If **True**, the image is used as the intensity prior,
                instead of sampling the intensities from the seeds. Default is **False**.
        """
        super().__init__(bids_path, generator_mild, seed_path, sub_list, load_image, image_as_intensity)
        self.seed_path = (
            Path(seed_path) if isinstance(seed_path, str) else None
        )
        self.generator_mild = generator_mild
        self.generator_severe = generator_severe
        self.n_mild_samples = n_mild_samples
        self.n_severe_samples = n_severe_samples
        self.mask_image = mask_image

    def sample(self, idx, genparams: dict = {}) -> tuple[dict, dict]:
        """
        Retrieve a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.
            genparams (dict): Dictionary with generation parameters.
                Used for fixed generation. Should follow exactly the same structure
                and be of the same type as the returned generation parameters.
                Can be used to replicate the augmentations (power)
                used for the generation of a specific sample.
        Returns:
            Dictionaries with the generated data and the generation parameters.
                First dictionary contains the `image`, `label` and the `name` keys.
                The second dictionary contains the parameters used for the generation.

        !!! Note
            The `image` is scaled to `[0, 1]` and oriented with the `label` to **RAS**
            and returned on the device  specified in the `generator` initialization.
        """
        # use generation_params to track the parameters used for the generation
        generation_params = {}

        image = self.loader(self.img_paths[idx]) if self.load_image else None
        segm = self.loader(self.segm_paths[idx])

        # orient to RAS for consistency
        image = (
            self.orientation(image.unsqueeze(0)).squeeze(0)
            if self.load_image
            else None
        )
        segm = self.orientation(segm.unsqueeze(0)).squeeze(0)

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        # initialize seeds as dictionary
        # with paths to the seeds volumes
        # or None if image is to be used as intensity prior
        if self.seed_path is not None:
            seeds = self.seed_paths[name]
        if self.image_as_intensity:
            seeds = None

        # log input data
        generation_params["idx"] = idx
        generation_params["img_paths"] = str(self.img_paths[idx])
        generation_params["segm_paths"] = str(self.segm_paths[idx])
        generation_params["seeds"] = str(self.seed_path)
        generation_time_start = time.time()

        # Generate the contrast image

        seeds, selected_seeds = self.generator_mild.intensity_generator.load_seeds(
            seeds=seeds, genparams=genparams.get("selected_seeds", {})
        )
        xx2, yy2, zz2, flip, deform_params = self.generator_mild.spatial_deform.generate_deformation_and_flip(
            image_shape=seeds.shape,
            genparams=genparams,
        )
     
        # ensure that tensors are on the same device

        device = self.generator_mild.device
        im_run = image.to(device).clone() if image is not None else None
        segm_run = segm.to(device).clone()
        samples = []
        synth_params_defaultdict = defaultdict(list)
        for i in range(self.n_mild_samples+self.n_severe_samples):
            if i < self.n_mild_samples:
                generator = self.generator_mild
            else:
                generator = self.generator_severe

            # 1. Generate intensity output.
            output, seed_intensities = (
                generator.intensity_generator.sample_intensities(
                    seeds=seeds,
                    device=device,
                    genparams=genparams.get("seed_intensities", {}),
                )
            )
            output = output.to(device)
            affine = output.affine
            # 2. Spatially deform the data
            im_out, segm_out, output = self.generator.spatial_deform.apply_deformation_and_flip(im_run, segm_run, output, xx2, yy2, zz2, flip)
            synth_params = {
                "selected_seeds": selected_seeds,
                "seed_intensities": seed_intensities,
                "deform_params": deform_params,
            }

            # generate the synthetic data
            gen_output, synth_params_aug = generator.augment(
                image=output, segmentation=segm_out,  genparams=genparams
            )
        
            # scale the images to [0, 1]
            gen_output = self.scaler(gen_output)
            image = self.scaler(image) if image is not None else None

            # ensure image and segmentation are on the cpu
            gen_output = gen_output.cpu()
            
            samples.append(gen_output.unsqueeze(0))
            synth_params = synth_params.update(synth_params_aug)
            for k, v in synth_params_aug.items():
                synth_params_defaultdict[k].append(v)

        if self.mask_image:
            mask = segm_out > 0
            im_out = im_out * mask if im_out is not None else None
        im_out = im_out.cpu() if im_out is not None else None
        segm_out = segm_out.cpu()

        generation_params = {**generation_params, **synth_params_defaultdict}
        generation_params["generation_time"] = (
            time.time() - generation_time_start
        )

        data_out = {
            "input": samples,
            "name": name,
            "image": im_out.unsqueeze(0) if image is not None else None,
            "seg": segm_out.unsqueeze(0).long(),
            "aff": affine.cpu(),
            "shp": torch.tensor(segm_out.shape).cpu(),
        }

        return data_out
    
    def __getitem__(self, idx, genparams: dict = {}):

        return self.sample(idx, genparams)


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
            dataset, BrainIDFetalSynthDataset
        ), "Dataset must be BrainIDFetalSynthDataset"
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
            self.batch = self.dataset[idx]
            self.idx_dataset = idx
        return self.batch

    def __len__(self):
        return len(self.dataset) * self.patch_per_subject

    def __getitem__(self, idx):

        idx_dataset = idx // self.patch_per_subject
        batch = self.get_sub_samp(idx_dataset)

        im_size = batch["image"].shape[-3:]
        slice_idx = self.random_patch(im_size)
        slice_ = [slice(*idx) for idx in slice_idx]
        subjects_patch = {
            "input": [],
            "name": batch["name"],
            "aff": batch["aff"],
            "slice_patch": slice_idx,
        }
        for sample in batch["input"]:
            subjects_patch["input"].append(sample[:, slice_[0], slice_[1], slice_[2]])
        subjects_patch["image"] = batch["image"][
            :, slice_[0], slice_[1], slice_[2]
        ]
        subjects_patch["seg"] = batch["seg"][:, slice_[0], slice_[1], slice_[2]]
        subjects_patch["shp"] = torch.tensor(subjects_patch["image"].shape)
        subjects_patch["shp_init"] = torch.tensor(batch["image"].shape)

        return subjects_patch


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
    

class FetalSynthIDDataset(FetalSynthDataset):
    def __init__(
        self,
        bids_path: str,
        generator: FetalSynthGen,
        seed_path: str | None,
        sub_list: list[str] | None,
        load_image: bool = False,
        image_as_intensity: bool = False,
    ):
        super().__init__(bids_path, generator, seed_path, sub_list, load_image, image_as_intensity)

    def sample(self, idx, genparams: dict = {}) -> tuple[dict, dict]:
        """
        Retrieve a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.
            genparams (dict): Dictionary with generation parameters.
                Used for fixed generation. Should follow exactly the same structure
                and be of the same type as the returned generation parameters.
                Can be used to replicate the augmentations (power)
                used for the generation of a specific sample.
        Returns:
            Dictionaries with the generated data and the generation parameters.
                First dictionary contains the `image`, `label` and the `name` keys.
                The second dictionary contains the parameters used for the generation.

        !!! Note
            The `image` is scaled to `[0, 1]` and oriented with the `label` to **RAS**
            and returned on the device  specified in the `generator` initialization.
        """
        # use generation_params to track the parameters used for the generation
        generation_params = {}

        image = self.loader(self.img_paths[idx]) if self.load_image else None
        segm = self.loader(self.segm_paths[idx])

        # orient to RAS for consistency
        image = (
            self.orientation(image.unsqueeze(0)).squeeze(0)
            if self.load_image
            else None
        )
        segm = self.orientation(segm.unsqueeze(0)).squeeze(0)

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        # initialize seeds as dictionary
        # with paths to the seeds volumes
        # or None if image is to be used as intensity prior
        if self.seed_path is not None:
            seeds = self.seed_paths[name]
        if self.image_as_intensity:
            seeds = None

        # log input data
        generation_params["idx"] = idx
        generation_params["img_paths"] = str(self.img_paths[idx])
        generation_params["segm_paths"] = str(self.img_paths[idx])
        generation_params["seeds"] = str(self.seed_path)
        generation_time_start = time.time()

        # generate the synthetic data
        gen_output, segmentation, image, synth_params = self.generator.sample(
            image=image, segmentation=segm, seeds=seeds, genparams=genparams
        )

        # scale the images to [0, 1]
        gen_output = self.scaler(gen_output)
        image = self.scaler(image) if image is not None else None

        # ensure image and segmentation are on the cpu
        gen_output = gen_output.cpu()
        segmentation = segmentation.cpu()
        image = image.cpu() if image is not None else None

        generation_params = {**generation_params, **synth_params}
        generation_params["generation_time"] = (
            time.time() - generation_time_start
        )
        data_out = {
            "input": gen_output.unsqueeze(0),
            "image": image.unsqueeze(0) if image is not None else None,
            "label": segmentation.unsqueeze(0).long(),
            "name": name,
        }

        return data_out, generation_params