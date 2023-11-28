

"""
Datasets interface.
"""
from .synth import BaseSynth
from .id_synth import IDSynth, DeformIDSynth
from .id_synth_eval import IDSynthEval, DeformIDSynthEval
from .supervised import ContrastSpecificDataset


dataset_options = { 
    'train':{
        'synth': BaseSynth,  
        'synth_id': IDSynth,  
        'synth_id_deform': DeformIDSynth,
        'supervised': ContrastSpecificDataset,
    },
    'test':{
        'synth_id': IDSynthEval,  
        'synth_id_deform': DeformIDSynthEval,
        'supervised': ContrastSpecificDataset,
    }
}


dataset_paths = {
    'synth': {
        'train': '/autofs/space/yogurt_001/users/pl629/data/synth',
        'test': '/autofs/space/yogurt_001/users/pl629/data/synth/images',
    },
    'ADNI': {
        'T1': '/autofs/space/yogurt_001/users/pl629/data/synth/images',
        'Seg': '/autofs/space/yogurt_001/users/pl629/data/synth/label_maps_segmentation',
    },
    'ADNI3': {
        'T1': '/autofs/space/yogurt_004/users/pl629/ADNI3/T1',
        'FLAIR': '/autofs/space/yogurt_004/users/pl629/ADNI3/FLAIR',
        'Seg': '/autofs/space/yogurt_004/users/pl629/ADNI3/T1-SynthSeg',
    },
    'ADHD200': {
        'T1': '/autofs/space/yogurt_004/users/pl629/ADHD200/T1',
        'Seg': '/autofs/space/yogurt_004/users/pl629/ADHD200/T1-SynthSeg',
    },
    'AIBL': {
        'T1': '/autofs/space/yogurt_004/users/pl629/AIBL/T1',
        'T2': '/autofs/space/yogurt_004/users/pl629/AIBL/T2',
        'FLAIR': '/autofs/space/yogurt_004/users/pl629/AIBL/FLAIR',
        'Seg': '/autofs/space/yogurt_004/users/pl629/AIBL/T1-SynthSeg',
    },
    'HCP': {
        'T1': '/autofs/space/yogurt_004/users/pl629/HCP/T1',
        'T2': '/autofs/space/yogurt_004/users/pl629/HCP/T2',
        'Seg': '/autofs/space/yogurt_004/users/pl629/HCP/T1-SynthSeg',
    },
    'OASIS3': {
        'CT': '/autofs/space/yogurt_004/users/pl629/OASIS3/CT',
        'T1': '/autofs/space/yogurt_004/users/pl629/OASIS3/T1toCT',
        'Seg': '/autofs/space/yogurt_004/users/pl629/OASIS3/T1toCT-SynthSeg',
    },
}


def get_dir(dataset, modality, split, task): 
    if 'synth' in dataset:
        return dataset_paths['synth'][split], None
    else: # TODO: split train/test partition
        if 'seg' in task or 'bf' in task:
            return dataset_paths[dataset][modality], dataset_paths[dataset]['Seg']
        elif 'anat' in task: 
            return dataset_paths[dataset][modality], dataset_paths[dataset]['T1']
        elif 'sr' in task:
            return dataset_paths[dataset][modality], dataset_paths[dataset][modality]
        else:
            raise ValueError('Unsupported task type:', task)




def build_dataset_single(dataset_name, split, args, device):
    """Helper function to build dataset for different splits ('train' or 'test')."""
    data_dir, gt_dir = get_dir(args.dataset, args.modality, split, args.task)
    if 'supervised' in dataset_name:
        return dataset_options[split][dataset_name](args, data_dir, gt_dir, device)
    else:
        return dataset_options[split][dataset_name](args, data_dir, device)
    


def build_dataset_multi(dataset_name_list, split, args, device):
    """Helper function to build dataset for different splits ('train' or 'test')."""
    datasets = {}
    for name in dataset_name_list:
        datasets[name] = build_dataset_single(name, split, args, device)
    return datasets


