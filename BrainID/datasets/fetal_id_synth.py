from typing import (
    Iterator,
    List,
    Sized,
)
import torch
from fetalsynthgen.definitions import GeneratorParams
from fetalsynthgen.dataset import FetalBIDSDataset, SynthDataset
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.sampler import Sampler
from pathlib import Path


class FetalIDSynth(SynthDataset):

    def __init__(
        self,
        bids_path: str,
        subjects: list[str] | None = None,
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
            shape = [self.nchannels, *synth_image.shape[-3:]]

            # metadata = self.generator.metadata_list
            data = {
                "image": synth_image.view(shape),
                "gt": extras["linear"][0].view(shape),
            }

            data = self.scale_data(data)

            samples = []
            for samp in range(self.nchannels):
                sample_dict = {
                    "input": data["image"][None, samp],
                }
                samples.append(sample_dict)

            device = synth_image.device
            subjects = {
                "name": self.subjects[idx],
                "image": data["gt"][None, 0],
                "seg": extras["nearest"][0].view(shape)[None, 0].long(),
                "aff": affine,
                "shp": torch.tensor(segmentation.shape).to(device),
            }

            # fill nans with zeros
            # import json

            # with open(
            #     f"/home/tsanchez/Documents/mial/repositories/Brain-ID/outs/BrainID/feat-anat/anat-uncertainty/metadata/{self.subjects[idx]}.json",
            #     "w",
            # ) as f:
            #     json.dump(metadata, f, indent=4)

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
                # "image_def": sample["image_def"][
                #    :, slice_[0], slice_[1], slice_[2]
                # ],
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


from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensityd,
    SignalFillEmptyd,
    CropForegroundd,
    SpatialPadd,
    Orientation,
    Spacingd,
    CenterSpatialCropd,
    Resized,
)


class RealDataset(FetalBIDSDataset):
    """
    Dataset class for datasets of real images.
    Responsible for reading real (not synthetic) images, segmentations and metadata.
    """

    def __init__(
        self,
        bids_path: str,
        subjects: list[str] | None = None,
        rescale_res: float = 0.5,
        segm_path: str | None = None,
        use_json: bool = True,
        device: str = "cuda",
        target_key: str | None = None,
        transform_target: str | None = None,
        target_threshold: float | None = None,
        augmentations: Compose | None = None,
        **kwargs,
    ):
        """Args:
        bids_path (str): Path to the bids folder with the data.
        subjects list[str]: subjects to read
        rescale_res (float, optional): Resolution to rescale
            all images to. Defaults to None, no rescaling.
        segm_path (str, optional): Path to a bids root
            from which to look for GT segmentations. Defaults to None.
        use_json (bool, optional): Whether to use json files with metadata.
        device (str, optional): Device to use. Defaults to "cuda".
        """
        super().__init__(bids_path, subjects, rescale_res, device)
        self.use_json = use_json
        self.im_paths = self._load_bids_path(
            self.bids_path, "T2w", load_json=self.use_json
        )
        if self.use_json:
            self.im_paths, self.json_paths = self.im_paths[0], self.im_paths[1]
        self.segm_path = Path(segm_path) if segm_path is not None else None

        if self.segm_path is not None:
            if not self.segm_path.exists():
                raise FileNotFoundError(
                    f"Provided segmentation path {self.segm_path} does not exist."
                )
            self.segm_paths = self._load_bids_path(
                self.segm_path, "dseg", load_json=False
            )
        else:
            self.segm_paths = None

        self.base_transforms = Compose(
            [
                SignalFillEmptyd(
                    keys=["image", "label"],
                    replacement=0,
                    allow_missing_keys=True,
                ),
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    allow_smaller=True,
                    allow_missing_keys=True,
                    margin=0,
                ),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(256, 256, 256),
                    mode="constant",
                    allow_missing_keys=True,
                ),
                CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(256, 256, 256),
                    allow_missing_keys=True,
                ),
            ]
        )

        self.target_key = target_key
        if target_key:
            self.transform_target = transform_target
            self.target_threshold = target_threshold
            assert self.transform_target in [
                "binarize",
                "soft_binarize",
                "continuous",
            ], "transform_target must be either 'binarize', 'soft_binarize' or 'continuous'"

        self.augmentations = augmentations

    def scale_data(self, data):
        for key, val in data.items():
            shape = val.shape
            val = val.view(shape[0], -1)
            val -= val.min(1, keepdim=True)[0]
            val /= val.max(1, keepdim=True)[0]
            data[key] = val.view(shape)
        return data

    def scale_target(self, target):
        if self.transform_target == "binarize":
            target_trf = (target > self.target_threshold).type(torch.float32)
            target_trf = torch.tensor([1 - target_trf, target_trf])
        elif self.transform_target == "soft_binarize":
            # Transform from 0 - 4 to 0-1 in a non-linear way: Provide more meaningful dynamics for the soft labels.
            # ToDo. Try a simple linear attenuation and see if it changes anything.
            # trf = lambda x: (
            #     self.target_threshold / (1 + torch.exp(-4 * (x - 1)))
            #     if x < self.target_threshold
            #     else self.target_threshold / (1 + torch.exp(-2 * (x - 1)))
            # )
            # Change it with a symmetric sigmoid; 3 is a good slope.

            trf = lambda x: 1 / (
                1 + torch.exp(-3 * (x - self.target_threshold))
            )
            target_trf = trf(target)
            target_trf = torch.tensor([1 - target_trf, target_trf])
        return target_trf

    def __len__(self):
        return len(self.sub_ses_num)

    def __getitem__(self, idx):
        MAX_SHAPE = 160
        with torch.inference_mode():
            image = self.im_paths[idx]
            affine = image.affine
            shape = [1, *image.shape[-3:]]

            extras = {
                "nearest": [],
                "linear": [],
            }  # + [seeds] if needed for debug
            if self.segm_path is not None:
                segmentation = self.segm_paths[idx]
                extras["nearest"].append(
                    torch.tensor(segmentation.get_fdata()).long()
                )
            import pdb
            import nibabel as nib
            from fetal_brain_utils.cropping import (
                get_cropped_stack_based_on_mask,
            )

            mask = nib.Nifti1Image(
                (image.get_fdata() > 0).astype(int), affine, dtype=np.int8
            )
            imagec = get_cropped_stack_based_on_mask(image, mask)
            affine = imagec.affine
            data = {
                "image": torch.tensor(imagec.get_fdata())
                .view(1, *imagec.shape[-3:])
                .contiguous()
                .to(self.device)
            }
            # print(data["image"].shape, nib.affines.voxel_sizes(affine))
            data = self.scale_data(data)
            if self.augmentations is not None:
                data = self.augmentations(data)

            if self.segm_path is not None:
                data["label"] = extras["nearest"][0].view(shape).long()
            # data = self.base_transforms(data)

            shape = data["image"].shape[-3:]

            max_shape = max(max(shape), MAX_SHAPE)

            # Make it even

            # Pad data to have a square shape
            # Do a center padding to have a square shape
            # Do a bounding box around the non-zero elements

            pad = [int((max_shape - s) / 2) for s in shape[::-1]]
            pad = [p for p in pad for _ in range(2)]
            for i, s in enumerate(shape[::-1]):
                is_odd = s % 2
                if is_odd:
                    pad[i * 2 + 1] += 1

            data["image"] = torch.nn.functional.pad(
                data["image"], pad, mode="constant", value=0
            )
            zooms = nib.affines.voxel_sizes(affine)
            new_shape = (max_shape,) * 3
            affine = nib.affines.rescale_affine(
                affine, shape, zooms=zooms, new_shape=new_shape
            )
            target_shape = (MAX_SHAPE,) * 3
            ## ISSUE: there is a reshaping even when the max_shape is < MAX_SHAPE
            if max_shape > MAX_SHAPE:
                new_zooms = (
                    data["image"].shape[-1] * zooms[0] / MAX_SHAPE,
                ) * 3

                affine = nib.affines.rescale_affine(
                    affine,
                    new_shape,
                    zooms=new_zooms,
                    new_shape=target_shape,
                )

                data["image"] = torch.nn.functional.interpolate(
                    data["image"].view(1, *data["image"].shape),
                    size=target_shape,
                    mode="trilinear",
                )

            samples = [{"input": data["image"].float().view(1, *target_shape)}]
            subjects = {
                "name": "_".join(
                    [x for x in self.sub_ses_num[idx] if x is not None]
                ),
                "image": data["image"].float().view(1, *target_shape),
                "aff": torch.tensor(affine).to(self.device),
                "shp": torch.tensor(image.shape).to(self.device),
            }

            # print(subjects["name"])
            if self.segm_path is not None:
                subjects["seg"] = data["label"].view(shape)[None, 0].long()

            if self.use_json:
                metadata = self.json_paths[idx]
                metadata = {
                    k: torch.tensor([v]).to(self.device)
                    for k, v in metadata.items()
                }
                subjects.update(**metadata)
            if self.target_key:
                target = subjects[self.target_key]
                subjects["target_raw"] = target
                subjects["target_binary"] = (
                    target > self.target_threshold
                ).type(torch.LongTensor)
                target_trf = self.scale_target(target)
                subjects["target"] = target_trf

        return subjects, samples
