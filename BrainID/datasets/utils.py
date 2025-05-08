import torch
from BrainID.datasets.fetal_id_synth import (
    RandomBlockPatchFetalDataset,
    BlockRandomSampler,
)
from torch.utils.data import DataLoader


def get_loader(dataset, cfg):
    if cfg.train_patch:
        dataset = RandomBlockPatchFetalDataset(
            dataset=dataset,
            patch_size=cfg.patch_size,
            boundary=cfg.patch_boundary,
            patch_per_subject=cfg.patch_per_subject,
        )

    if cfg.train_patch:
        sampler_train = BlockRandomSampler(dataset, cfg.patch_per_subject)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler_train, cfg.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        # collate_fn=utils.collate_fn, # apply custom data cooker if needed
        num_workers=cfg.num_workers,
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )
    return data_loader_train
