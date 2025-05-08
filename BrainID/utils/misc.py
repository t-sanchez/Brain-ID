"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

import os
import math
import time
import datetime
import pickle
import subprocess
import warnings
from argparse import Namespace
from typing import List, Optional
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk

# from utils.process_cfg import load_config
from BrainID.utils.process_cfg import build_out_dir
from collections import defaultdict, deque


import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor
from visdom import Visdom
from types import SimpleNamespace
from omegaconf import DictConfig, ListConfig
import datetime
import pytz


def get_params_groups(model):
    all = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        all.append(param)
    return [{"params": all}]


def dictconfig_to_namespace(cfg: DictConfig) -> SimpleNamespace:
    """
    Recursively converts a DictConfig to a SimpleNamespace.
    """

    def convert(value):
        if isinstance(value, DictConfig):
            return dictconfig_to_namespace(value)
        elif isinstance(value, ListConfig):
            return [convert(item) for item in value]
        elif isinstance(value, list):  # Handle lists with nested DictConfigs
            return [convert(item) for item in value]
        return value

    return SimpleNamespace(
        **{key: convert(value) for key, value in cfg.items()}
    )


"""if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size"""


def get_output_dir(cfg):
    tz_CH = pytz.timezone("Europe/Zurich")
    out_dir = os.path.join(
        cfg.out_dir,
        "Test",
        cfg.exp_name,
        cfg.job_name,
        datetime.datetime.now(tz_CH).strftime("%m%d-%H%M"),
    )
    return out_dir


def make_dir(dir_name, parents=True, exist_ok=True):
    dir_name = Path(dir_name)
    dir_name.mkdir(parents=parents, exist_ok=exist_ok)
    return dir_name


def read_image(img_path, save_path=None):
    img = nib.load(img_path)
    nda = img.get_fdata()
    affine = img.affine
    if save_path:
        ni_img = nib.Nifti1Image(nda, affine)
        nib.save(ni_img, save_path)
    return nda, affine


def save_image(nda, affine, save_path):
    ni_img = nib.Nifti1Image(nda, affine)
    nib.save(ni_img, save_path)
    return save_path


def img2nda(img_path, save_path=None):
    img = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(img)
    if save_path:
        np.save(save_path, nda)
    return nda, img.GetOrigin(), img.GetSpacing(), img.GetDirection()


def to3d(img_path, save_path=None):
    nda, o, s, d = img2nda(img_path)
    save_path = img_path if save_path is None else save_path
    if len(o) > 3:
        nda2img(nda, o[:3], s[:3], d[:3] + d[4:7] + d[8:11], save_path)
    return save_path


def nda2img(
    nda,
    origin=None,
    spacing=None,
    direction=None,
    save_path=None,
    isVector=None,
):
    if type(nda) == torch.Tensor:
        nda = nda.cpu().detach().numpy()
    nda = np.squeeze(np.array(nda))
    isVector = isVector if isVector else len(nda.shape) > 3
    img = sitk.GetImageFromArray(nda, isVector=isVector)
    if origin:
        img.SetOrigin(origin)
    if spacing:
        img.SetSpacing(spacing)
    if direction:
        img.SetDirection(direction)
    if save_path:
        sitk.WriteImage(img, save_path)
    return img


def cropping(img_path, tol=0, crop_range_lst=None, spare=0, save_path=None):

    img = sitk.ReadImage(img_path)
    orig_nda = sitk.GetArrayFromImage(img)
    if len(orig_nda.shape) > 3:  # 4D data: last axis (t=0) as time dimension
        nda = orig_nda[..., 0]
    else:
        nda = np.copy(orig_nda)

    if crop_range_lst is None:
        # Mask of non-black pixels (assuming image has a single channel).
        mask = nda > tol
        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)
        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top
        # add sparing gap if needed
        x0 = x0 - spare if x0 > spare else x0
        y0 = y0 - spare if y0 > spare else y0
        z0 = z0 - spare if z0 > spare else z0
        x1 = x1 + spare if x1 < orig_nda.shape[0] - spare else x1
        y1 = y1 + spare if y1 < orig_nda.shape[1] - spare else y1
        z1 = z1 + spare if z1 < orig_nda.shape[2] - spare else z1

        # Check the the bounding box #
        # print('    Cropping Slice [%d, %d)' % (x0, x1))
        # print('    Cropping Row [%d, %d)' % (y0, y1))
        # print('    Cropping Column [%d, %d)' % (z0, z1))

    else:
        [[x0, y0, z0], [x1, y1, z1]] = crop_range_lst

    cropped_nda = orig_nda[x0:x1, y0:y1, z0:z1]
    new_origin = [
        img.GetOrigin()[0] + img.GetSpacing()[0] * z0,
        img.GetOrigin()[1] + img.GetSpacing()[1] * y0,
        img.GetOrigin()[2] + img.GetSpacing()[2] * x0,
    ]  # numpy reverse to sitk'''
    cropped_img = sitk.GetImageFromArray(
        cropped_nda, isVector=len(orig_nda.shape) > 3
    )
    cropped_img.SetOrigin(new_origin)
    # cropped_img.SetOrigin(img.GetOrigin())
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetDirection(img.GetDirection())
    if save_path:
        sitk.WriteImage(cropped_img, save_path)

    return cropped_img, [[x0, y0, z0], [x1, y1, z1]], new_origin




#########################################
#########################################


def viewVolume(x, aff=None, prefix="", postfix="", names=[], save_dir="/tmp"):

    if aff is None:
        aff = np.eye(4)
    else:
        if type(aff) == torch.Tensor:
            aff = aff.cpu().detach().numpy()

    if type(x) is dict:
        names = list(x.keys())
        x = [x[k] for k in x]

    if type(x) is not list:
        x = [x]

    # cmd = 'source /usr/local/freesurfer/nmr-dev-env-bash && freeview '

    for n in range(len(x)):
        vol = x[n]
        if vol is not None:
            if type(vol) == torch.Tensor:
                vol = vol.cpu().detach().numpy()
            vol = np.squeeze(np.array(vol))
            try:
                save_path = os.path.join(
                    save_dir, prefix + names[n] + postfix + ".nii.gz"
                )
            except:
                save_path = os.path.join(
                    save_dir, prefix + str(n) + postfix + ".nii.gz"
                )
            MRIwrite(vol, aff, save_path)
            # cmd = cmd + ' ' + save_path

    # os.system(cmd + ' &')
    return save_path


###############################3


def MRIwrite(volume, aff, filename, dtype=None):

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if aff is None:
        aff = np.eye(4)
    header = nib.Nifti1Header()
    nifty = nib.Nifti1Image(volume, aff, header)

    nib.save(nifty, filename)


###############################


def MRIread(filename, dtype=None, im_only=False):

    assert filename.endswith((".nii", ".nii.gz", ".mgz")), (
        "Unknown data file: %s" % filename
    )

    x = nib.load(filename)
    volume = x.get_fdata()
    aff = x.affine

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if im_only:
        return volume
    else:
        return volume, aff


##############


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(
        np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0
    )
    return img_ras_axes


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size,), dtype=torch.uint8, device="cuda")
        )
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return (
            subprocess.check_output(command, cwd=cwd).decode("ascii").strip()
        )

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = {
        k: torch.stack([dict[k] for dict in batch]) for k in batch[0]
    }  # switch from batch of dict to dict of batch
    return batch
    # v = {k: [dic[k] for dic in LD] for k in LD[0]}


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes



def nested_dict_to_device(dictionary, device):

    if isinstance(dictionary, dict):
        output = {}
        for key, value in dictionary.items():
            output[key] = nested_dict_to_device(value, device)
        return output

    if isinstance(dictionary, str):
        return dictionary
    elif isinstance(dictionary, list):
        return [nested_dict_to_device(d, device) for d in dictionary]
    else:
        try:
            return dictionary.to(device)
        except:
            return dictionary

# def preprocess_cfg(cfg_files, cfg_dir=""):
#     config = load_config(cfg_files[0], cfg_files[1:], cfg_dir)

#     args = nested_dict_to_namespace(config)
#     return args



######################### Synth #########################


def myzoom_torch_slow(X, factor, device, aff=None):

    if len(X.shape) == 3:
        X = X[..., None]

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(
        delta[0],
        delta[0] + newsize[0] / factor[0],
        1 / factor[0],
        dtype=torch.float,
        device=device,
    )[: newsize[0]]
    vy = torch.arange(
        delta[1],
        delta[1] + newsize[1] / factor[1],
        1 / factor[1],
        dtype=torch.float,
        device=device,
    )[: newsize[1]]
    vz = torch.arange(
        delta[2],
        delta[2] + newsize[2] / factor[2],
        1 / factor[2],
        dtype=torch.float,
        device=device,
    )[: newsize[2]]

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0] - 1)] = X.shape[0] - 1
    vy[vy > (X.shape[1] - 1)] = X.shape[1] - 1
    vz[vz > (X.shape[2] - 1)] = X.shape[2] - 1

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0] - 1)] = X.shape[0] - 1
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1] - 1)] = X.shape[1] - 1
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2] - 1)] = X.shape[2] - 1
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros(
        [newsize[0], newsize[1], newsize[2], X.shape[3]],
        dtype=torch.float,
        device=device,
    )

    for channel in range(X.shape[3]):
        Xc = X[:, :, :, channel]

        tmp1 = torch.zeros(
            [newsize[0], Xc.shape[1], Xc.shape[2]],
            dtype=torch.float,
            device=device,
        )
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] + wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros(
            [newsize[0], newsize[1], Xc.shape[2]],
            dtype=torch.float,
            device=device,
        )
        for j in range(newsize[1]):
            tmp2[:, j, :] = (
                wfy[j] * tmp1[:, fy[j], :] + wcy[j] * tmp1[:, cy[j], :]
            )
        for k in range(newsize[2]):
            Y[:, :, k, channel] = (
                wfz[k] * tmp2[:, :, fz[k]] + wcz[k] * tmp2[:, :, cz[k]]
            )

    if Y.shape[3] == 1:
        Y = Y[:, :, :, 0]

    if aff is not None:
        aff_new = aff.copy()
        for c in range(3):
            aff_new[:-1, c] = aff_new[:-1, c] / factor
        aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (
            0.5 - 0.5 / (factor * np.ones(3))
        )
        return Y, aff_new
    else:
        return Y


def myzoom_torch(X, factor, aff=None):

    if len(X.shape) == 3:
        X = X[..., None]

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(
        delta[0],
        delta[0] + newsize[0] / factor[0],
        1 / factor[0],
        dtype=torch.float,
        device=X.device,
    )[: newsize[0]]
    vy = torch.arange(
        delta[1],
        delta[1] + newsize[1] / factor[1],
        1 / factor[1],
        dtype=torch.float,
        device=X.device,
    )[: newsize[1]]
    vz = torch.arange(
        delta[2],
        delta[2] + newsize[2] / factor[2],
        1 / factor[2],
        dtype=torch.float,
        device=X.device,
    )[: newsize[2]]

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0] - 1)] = X.shape[0] - 1
    vy[vy > (X.shape[1] - 1)] = X.shape[1] - 1
    vz[vz > (X.shape[2] - 1)] = X.shape[2] - 1

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0] - 1)] = X.shape[0] - 1
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1] - 1)] = X.shape[1] - 1
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2] - 1)] = X.shape[2] - 1
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros(
        [newsize[0], newsize[1], newsize[2], X.shape[3]],
        dtype=torch.float,
        device=X.device,
    )

    tmp1 = torch.zeros(
        [newsize[0], X.shape[1], X.shape[2], X.shape[3]],
        dtype=torch.float,
        device=X.device,
    )
    for i in range(newsize[0]):
        tmp1[i, :, :] = wfx[i] * X[fx[i], :, :] + wcx[i] * X[cx[i], :, :]
    tmp2 = torch.zeros(
        [newsize[0], newsize[1], X.shape[2], X.shape[3]],
        dtype=torch.float,
        device=X.device,
    )
    for j in range(newsize[1]):
        tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] + wcy[j] * tmp1[:, cy[j], :]
    for k in range(newsize[2]):
        Y[:, :, k] = wfz[k] * tmp2[:, :, fz[k]] + wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:, :, :, 0]

    if aff is not None:
        aff_new = aff.copy()
        aff_new[:-1] = aff_new[:-1] / factor
        aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (
            0.5 - 0.5 / (factor * np.ones(3))
        )
        return Y, aff_new
    else:
        return Y


def myzoom_torch_test(X, factor, aff=None):
    time.sleep(3)

    start_time = time.time()
    Y2 = myzoom_torch_slow(X, factor, aff)
    print("slow", X.shape[-1], time.time() - start_time)

    time.sleep(3)

    start_time = time.time()
    Y1 = myzoom_torch(X, factor, aff)
    print("fast", X.shape[-1], time.time() - start_time)

    time.sleep(3)

    print("diff", (Y2 - Y1).mean(), (Y2 - Y1).max())
    return Y1


def myzoom_torch_anisotropic_slow(X, aff, newsize, device):

    if len(X.shape) == 3:
        X = X[..., None]

    factors = np.array(newsize) / np.array(X.shape[:-1])
    delta = (1.0 - factors) / (2.0 * factors)

    vx = torch.arange(
        delta[0],
        delta[0] + newsize[0] / factors[0],
        1 / factors[0],
        dtype=torch.float,
        device=device,
    )[: newsize[0]]
    vy = torch.arange(
        delta[1],
        delta[1] + newsize[1] / factors[1],
        1 / factors[1],
        dtype=torch.float,
        device=device,
    )[: newsize[1]]
    vz = torch.arange(
        delta[2],
        delta[2] + newsize[2] / factors[2],
        1 / factors[2],
        dtype=torch.float,
        device=device,
    )[: newsize[2]]

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0] - 1)] = X.shape[0] - 1
    vy[vy > (X.shape[1] - 1)] = X.shape[1] - 1
    vz[vz > (X.shape[2] - 1)] = X.shape[2] - 1

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0] - 1)] = X.shape[0] - 1
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1] - 1)] = X.shape[1] - 1
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2] - 1)] = X.shape[2] - 1
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros(
        [newsize[0], newsize[1], newsize[2], X.shape[3]],
        dtype=torch.float,
        device=device,
    )

    dtype = X.dtype
    for channel in range(X.shape[3]):
        Xc = X[:, :, :, channel]

        tmp1 = torch.zeros(
            [newsize[0], Xc.shape[1], Xc.shape[2]], dtype=dtype, device=device
        )
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] + wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros(
            [newsize[0], newsize[1], Xc.shape[2]], dtype=dtype, device=device
        )
        for j in range(newsize[1]):
            tmp2[:, j, :] = (
                wfy[j] * tmp1[:, fy[j], :] + wcy[j] * tmp1[:, cy[j], :]
            )
        for k in range(newsize[2]):
            Y[:, :, k, channel] = (
                wfz[k] * tmp2[:, :, fz[k]] + wcz[k] * tmp2[:, :, cz[k]]
            )

    if Y.shape[3] == 1:
        Y = Y[:, :, :, 0]

    if aff is not None:
        aff_new = aff.copy()
        for c in range(3):
            aff_new[:-1, c] = aff_new[:-1, c] / factors[c]
        aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (
            0.5 - 0.5 / factors
        )
        return Y, aff_new
    else:
        return Y


def myzoom_torch_anisotropic(X, aff, newsize):

    device = X.device

    if len(X.shape) == 3:
        X = X[..., None]

    factors = np.array(newsize) / np.array(X.shape[:-1])
    delta = (1.0 - factors) / (2.0 * factors)

    vx = torch.arange(
        delta[0],
        delta[0] + newsize[0] / factors[0],
        1 / factors[0],
        dtype=torch.float,
        device=device,
    )[: newsize[0]]
    vy = torch.arange(
        delta[1],
        delta[1] + newsize[1] / factors[1],
        1 / factors[1],
        dtype=torch.float,
        device=device,
    )[: newsize[1]]
    vz = torch.arange(
        delta[2],
        delta[2] + newsize[2] / factors[2],
        1 / factors[2],
        dtype=torch.float,
        device=device,
    )[: newsize[2]]

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0] - 1)] = X.shape[0] - 1
    vy[vy > (X.shape[1] - 1)] = X.shape[1] - 1
    vz[vz > (X.shape[2] - 1)] = X.shape[2] - 1

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0] - 1)] = X.shape[0] - 1
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1] - 1)] = X.shape[1] - 1
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2] - 1)] = X.shape[2] - 1
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros(
        [newsize[0], newsize[1], newsize[2], X.shape[3]],
        dtype=torch.float,
        device=device,
    )

    dtype = X.dtype
    for channel in range(X.shape[3]):
        Xc = X[:, :, :, channel]

        tmp1 = torch.zeros(
            [newsize[0], Xc.shape[1], Xc.shape[2]], dtype=dtype, device=device
        )
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] + wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros(
            [newsize[0], newsize[1], Xc.shape[2]], dtype=dtype, device=device
        )
        for j in range(newsize[1]):
            tmp2[:, j, :] = (
                wfy[j] * tmp1[:, fy[j], :] + wcy[j] * tmp1[:, cy[j], :]
            )
        for k in range(newsize[2]):
            Y[:, :, k, channel] = (
                wfz[k] * tmp2[:, :, fz[k]] + wcz[k] * tmp2[:, :, cz[k]]
            )

    if Y.shape[3] == 1:
        Y = Y[:, :, :, 0]

    if aff is not None:
        aff_new = aff.copy()
        for c in range(3):
            aff_new[:-1, c] = aff_new[:-1, c] / factors[c]
        aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (
            0.5 - 0.5 / factors
        )
        return Y, aff_new
    else:
        return Y


def torch_resize(
    I,
    aff,
    resolution,
    power_factor_at_half_width=5,
    dtype=torch.float32,
    slow=False,
):

    if torch.is_grad_enabled():
        with torch.no_grad():
            return torch_resize(
                I, aff, resolution, power_factor_at_half_width, dtype, slow
            )

    slow = slow or (I.device == "cpu")
    voxsize = np.sqrt(np.sum(aff[:-1, :-1] ** 2, axis=0))
    newsize = np.round(I.shape[0:3] * (voxsize / resolution)).astype(int)
    factors = np.array(I.shape[0:3]) / np.array(newsize)
    k = np.log(power_factor_at_half_width) / np.pi
    sigmas = k * factors
    sigmas[sigmas <= k] = 0

    if len(I.shape) not in (3, 4):
        raise Exception("torch_resize works with 3D or 3D+label volumes")
    no_channels = len(I.shape) == 3
    if no_channels:
        I = I[:, :, :, None]
    if torch.is_tensor(I):
        I = I.permute([3, 0, 1, 2])
    else:
        I = I.transpose([3, 0, 1, 2])

    It_lowres = None
    for c in range(len(I)):
        It = torch.as_tensor(I[c], device=I.device, dtype=dtype)[None, None]
        # Smoothen if needed
        for d in range(3):
            It = It.permute([0, 1, 3, 4, 2])
            if sigmas[d] > 0:
                sl = np.ceil(sigmas[d] * 2.5).astype(int)
                v = np.arange(-sl, sl + 1)
                gauss = np.exp((-((v / sigmas[d]) ** 2) / 2))
                kernel = gauss / np.sum(gauss)
                kernel = torch.tensor(kernel, device=I.device, dtype=dtype)
                if slow:
                    It = conv_slow_fallback(It, kernel)
                else:
                    kernel = kernel[None, None, None, None, :]
                    It = torch.conv3d(
                        It,
                        kernel,
                        bias=None,
                        stride=1,
                        padding=[0, 0, int((kernel.shape[-1] - 1) / 2)],
                    )

        It = torch.squeeze(It)
        It, aff2 = myzoom_torch_anisotropic(It, aff, newsize)
        It = It.detach()
        if torch.is_tensor(I):
            It = It.to(I.device)
        else:
            It = It.cpu().numpy()
        if len(I) == 1:
            It_lowres = It[None]
        else:
            if It_lowres is None:
                if torch.is_tensor(It):
                    It_lowres = It.new_empty([len(I), *It.shape])
                else:
                    It_lowres = np.empty_like(It, shape=[len(I), *It.shape])
            It_lowres[c] = It

        torch.cuda.empty_cache()

    if not no_channels:
        if torch.is_tensor(I):
            It_lowres = It_lowres.permute([1, 2, 3, 0])
        else:
            It_lowres = It_lowres.transpose([1, 2, 3, 0])
    else:
        It_lowres = It_lowres[0]

    return It_lowres, aff2


###############################


@torch.jit.script
def conv_slow_fallback(x, kernel):
    """1D Conv along the last dimension with padding"""
    y = torch.zeros_like(x)
    x = torch.nn.functional.pad(x, [(len(kernel) - 1) // 2] * 2)
    x = x.unfold(-1, size=len(kernel), step=1)
    x = x.movedim(-1, 0)
    for i in range(len(kernel)):
        y = y.addcmul_(x[i], kernel[i])
    return y


#######


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=3):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            volume = torch.swapaxes(volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = (
                ras_axes_flo[i],
                ras_axes_flo[swapped_axis_idx],
            )

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            volume = torch.flip(volume, [i])
            aff_flo[:, i] = -aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (
                volume.shape[i] - 1
            )

    if return_aff:
        return volume, aff_flo
    else:
        return volume



def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def has_batchnorms(model):
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


###############################

# map SynthSeg right to left labels for contrast synthesis
right_to_left_dict = {
    41: 2,
    42: 3,
    43: 4,
    44: 5,
    46: 7,
    47: 8,
    49: 10,
    50: 11,
    51: 12,
    52: 13,
    53: 17,
    54: 18,
    58: 26,
    60: 28,
}
