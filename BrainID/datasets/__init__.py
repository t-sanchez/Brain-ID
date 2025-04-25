# """
# Datasets interface.
# """

# from .fetal_id_synth import FetalIDSynth, RealDataset


# fetal_dataset_options = {
#     "train": {
#         "synth_id": FetalIDSynth,
#         "real_dataset": RealDataset,
#     },
#     "test": {
#         "synth_id": FetalIDSynth,
#         "real_dataset": RealDataset,
#     },
# }


# def build_fetal_dataset(
#     dm,
#     dataset_name,
#     train=True,
#     target_key=None,
#     transform_target=None,
#     target_threshold=None,
#     device=None,
# ):
#     """Helper function to build a fetal dataset."""

#     import pdb; pdb.set_trace()
#     if train:
#         split = "train"
#         args = dm.get_dataset_args(
#             dm.train_path,
#             dm.train_splits,
#             dm.train_transform,
#         )
#         del dm
#     else:
#         split = "test"
#         args = dm.get_dataset_args(
#             dm.train_path,
#             dm.valid_splits,
#             dm.val_transform,
#         )

#     dataset = fetal_dataset_options[split][dataset_name.__dict__[split]]
#     if target_key is not None:
#         args["target_key"] = target_key
#         args["transform_target"] = transform_target
#         args["target_threshold"] = target_threshold
#     if device is not None:
#         args["device"] = device
#     return dataset(**args)
