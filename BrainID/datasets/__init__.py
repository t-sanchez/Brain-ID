"""
Datasets interface.
"""

from .synth import BaseSynth
from .id_synth import IDSynth, DeformIDSynth
from .id_synth_eval import IDSynthEval, DeformIDSynthEval
from .supervised import ContrastSpecificDataset
from .fetal_id_synth import FetalIDSynth

dataset_options = {
    "train": {
        "synth": BaseSynth,
        "synth_id": IDSynth,
        "synth_id_deform": DeformIDSynth,
        "supervised": ContrastSpecificDataset,
    },
    "test": {
        "synth_id": IDSynthEval,
        "synth_id_deform": DeformIDSynthEval,
        "supervised": ContrastSpecificDataset,
    },
}


dataset_paths = {
    "synth": {
        "train": "/media/tsanchez/tsanchez_data/data/brain-ID/synth/",
        "test": "/media/tsanchez/tsanchez_data/data/brain-ID/synth/T1",
    },
    "ADNI": {
        "T1": "/path/to/data/synth/T1",
        "Seg": "/path/to/data/synth/label_maps_segmentation",
    },
    "ADNI3": {
        "T1": "/path/to/ADNI3/T1",
        "FLAIR": "/path/to/ADNI3/FLAIR",
        "Seg": "/path/to/ADNI3/T1-SynthSeg",
    },
    "ADHD200": {
        "T1": "/path/to/ADHD200/T1",
        "Seg": "/path/to/ADHD200/T1-SynthSeg",
    },
    "AIBL": {
        "T1": "/path/to/AIBL/T1",
        "T2": "/path/to/AIBL/T2",
        "FLAIR": "/path/to/AIBL/FLAIR",
        "Seg": "/path/to/AIBL/T1-SynthSeg",
    },
    "HCP": {
        "T1": "/path/to/HCP/T1",
        "T2": "/path/to/HCP/T2",
        "Seg": "/path/to/HCP/T1-SynthSeg",
    },
    "OASIS3": {
        "CT": "/path/to/OASIS3/CT",
        "T1": "/path/to/OASIS3/T1toCT",
        "Seg": "/path/to/OASIS3/T1toCT-SynthSeg",
    },
}


def get_dir(dataset, modality, split, task):
    if "synth" in dataset:
        return dataset_paths["synth"][split], None
    else:
        if "seg" in task or "bf" in task:
            return (
                dataset_paths[dataset][modality],
                dataset_paths[dataset]["Seg"],
            )
        elif "anat" in task:
            return (
                dataset_paths[dataset][modality],
                dataset_paths[dataset]["T1"],
            )
        elif "sr" in task:
            return (
                dataset_paths[dataset][modality],
                dataset_paths[dataset][modality],
            )
        else:
            raise ValueError("Unsupported task type:", task)


def build_dataset_single(dataset_name, split, args, device):
    """Helper function to build dataset for different splits ('train' or 'test')."""
    data_dir, gt_dir = get_dir(args.dataset, args.modality, split, args.task)
    print(f"data_dir: {data_dir} -- gt_dir: {gt_dir}")
    if "supervised" in dataset_name:
        return dataset_options[split][dataset_name](
            args, data_dir, gt_dir, device
        )
    else:
        return dataset_options[split][dataset_name](args, data_dir, device)


from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def build_fetal_dataset(config_dir):
    """Helper function to build a fetal dataset."""
    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
    ):
        cfg = compose(
            config_name="dhcp_onlinesynth",
        )
        dm = instantiate(cfg)
        train_args = dm.get_dataset_args(
            dm.train_path,
            dm.train_transform,
            dm.train_splits,
        )
        del dm
        return FetalIDSynth(**train_args)


def build_dataset_multi(dataset_name_list, split, args, device):
    """Helper function to build dataset for different splits ('train' or 'test')."""
    datasets = {}
    for name in dataset_name_list:
        datasets[name] = build_dataset_single(name, split, args, device)
    return datasets
