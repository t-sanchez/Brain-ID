"""
Model heads
"""

import torch
import torch.nn as nn


class IndepHead(nn.Module):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-independent tasks
    """

    def __init__(
        self,
        f_maps_list,
        out_channels,
        is_3d,
        out_feat_level=-1,
        *kwargs,
    ):
        super(IndepHead, self).__init__()
        self.out_feat_level = out_feat_level
        layers = (
            []
        )  # additional layers (same-size-output 3x3 conv) before final_conv, if len( f_maps_list ) > 1
        for i, in_feature_num in enumerate(f_maps_list[:-1]):
            layer = ConvBlock(
                in_feature_num, f_maps_list[i + 1], stride=1, is_3d=is_3d
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        conv = nn.Conv3d if is_3d else nn.Conv2d
        self.out_names = out_channels.keys()
        for out_name, out_channels_num in out_channels.items():
            self.add_module(
                "final_conv_%s" % out_name,
                conv(f_maps_list[-1], out_channels_num, 1),
            )

    def forward(self, x, *kwargs):
        x = x[self.out_feat_level]
        for layer in self.layers:
            x = layer(x)
        out = {}
        for name in self.out_names:
            out[name] = getattr(self, f"final_conv_{name}")(x)
        return out


class DepHead(nn.Module):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-dependent tasks
    """

    def __init__(
        self,
        args,
        f_maps_list,
        out_channels,
        is_3d,
        out_feat_level=-1,
        *kwargs,
    ):
        super(DepHead, self).__init__()
        self.out_feat_level = out_feat_level

        f_maps_list[0] += 1  # add one input image/contrast channel

        layers = (
            []
        )  # additional layers (same-size-output 3x3 conv) before final_conv, if len( f_maps_list ) > 1
        for i, in_feature_num in enumerate(f_maps_list[:-1]):
            layer = ConvBlock(
                in_feature_num, f_maps_list[i + 1], stride=1, is_3d=is_3d
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        conv = nn.Conv3d if is_3d else nn.Conv2d
        self.out_names = out_channels.keys()
        for out_name, out_channels_num in out_channels.items():
            self.add_module(
                "final_conv_%s" % out_name,
                conv(
                    f_maps_list[-1],
                    out_channels_num,
                    2 if args.losses.uncertainty is not None else 1,
                ),
            )

    def forward(self, x, image):
        x = x[self.out_feat_level]
        x = torch.cat([x, image], dim=1)
        for layer in self.layers:
            x = layer(x)
        out = {}
        for name in self.out_names:
            out[name] = getattr(self, f"final_conv_{name}")(x)
        return out


class MultiInputDepHead(DepHead):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-dependent tasks
    """

    def __init__(
        self,
        args,
        f_maps_list,
        out_channels,
        is_3d,
        out_feat_level=-1,
        *kwargs,
    ):
        super(MultiInputDepHead, self).__init__(
            args, f_maps_list, out_channels, is_3d, out_feat_level
        )

    def forward(self, feat_list, image_list):
        outs = []
        for i, x in enumerate(feat_list):
            x = x[self.out_feat_level]
            x = torch.cat([x, image_list[i]], dim=1)
            for layer in self.layers:
                x = layer(x)
            out = {}
            for name in self.out_names:
                out[name] = getattr(self, f"final_conv_{name}")(x)
            outs.append(out)
        return outs


class MultiInputIndepHead(IndepHead):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-independent tasks
    """

    def __init__(
        self,
        args,
        f_maps_list,
        out_channels,
        is_3d,
        out_feat_level=-1,
        *kwargs,
    ):
        super(MultiInputIndepHead, self).__init__(
            args, f_maps_list, out_channels, is_3d, out_feat_level
        )

    def forward(self, feat_list, *kwargs):
        outs = []
        for x in feat_list:
            x = x[self.out_feat_level]
            for layer in self.layers:
                x = layer(x)
            out = {}
            for name in self.out_names:
                out[name] = getattr(self, f"final_conv_{name}")(x)
            outs.append(out)
        return outs


class ConvBlock(nn.Module):
    """
    Specific same-size-output 3x3 convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, stride=1, is_3d=True):
        super().__init__()

        conv = nn.Conv3d if is_3d else nn.Conv2d
        self.main = conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


################################


import numpy as np
import torch
from torch import nn

"""
Using the CNN architecture of ClinicaDL (https://github.com/aramis-lab/clinicadl)
"""


class Reshape(nn.Module):
    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(*self.size)


# pre-commit comment
class PadMaxPool3d(nn.Module):
    def __init__(
        self, kernel_size, stride, return_indices=False, return_pad=False
    ):
        super(PadMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(
            kernel_size, stride, return_indices=return_indices
        )
        self.pad = nn.ConstantPad3d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [
            self.stride - f_maps.size(i + 2) % self.stride for i in range(3)
        ]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[2], 0, coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return (
                    output,
                    indices,
                    (coords[2], 0, coords[1], 0, coords[0], 0),
                )
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output


class PadMaxPool2d(nn.Module):
    def __init__(
        self, kernel_size, stride, return_indices=False, return_pad=False
    ):
        super(PadMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool2d(
            kernel_size, stride, return_indices=return_indices
        )
        self.pad = nn.ConstantPad2d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [
            self.stride - f_maps.size(i + 2) % self.stride for i in range(2)
        ]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, indices, (coords[1], 0, coords[0], 0)
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[1], 0, coords[0], 0)
            else:
                return output


def get_layers_fn(is_3d, norm="batch"):
    if is_3d:
        if norm == "batch":
            return nn.Conv3d, nn.BatchNorm3d, PadMaxPool3d
        elif norm == "instance":
            return nn.Conv3d, nn.InstanceNorm3d, PadMaxPool3d
    else:
        if norm == "batch":
            return nn.Conv2d, nn.BatchNorm2d, PadMaxPool2d
        elif norm == "instance":
            return nn.Conv2d, nn.InstanceNorm2d, PadMaxPool2d


class ConvNormPoolBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        is_3d=True,
        conv_kernel=3,
        conv_padding=1,
        conv_stride=1,
        pool_kernel=2,
        pool_stride=2,
        norm="instance",
    ):
        super().__init__()
        conv, norm, pool = get_layers_fn(is_3d, norm)
        self.block = nn.Sequential(
            conv(
                in_channels,
                out_channels,
                conv_kernel,
                stride=conv_stride,
                padding=conv_padding,
            ),
            norm(out_channels),
            nn.ReLU(),
            pool(kernel_size=pool_kernel, stride=pool_stride),
        )

    def forward(self, x):
        return self.block(x)


class DropoutLinearReLUBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class FCDropout(nn.Module):

    def __init__(self, n_input, output_size, fc_maps_list, dropout=0.5):
        super().__init__()
        fc_maps_list = [n_input] + list(fc_maps_list)
        self.fc = [nn.Flatten()]
        for i, features in enumerate(fc_maps_list[:-1]):
            self.fc.append(
                DropoutLinearReLUBlock(features, fc_maps_list[i + 1], dropout)
            )
        self.fc = nn.Sequential(
            *self.fc, nn.Linear(fc_maps_list[-1], output_size)
        )

    def forward(self, x):
        return self.fc(x)

    def predict(self, x):
        return self.forward(x)


class ScalarHead(nn.Module):
    """
    Task-specific head that takes a list of sample features as inputs
    """

    def __init__(
        self,
        in_shape,
        n_classes,
        out_channels,
        f_maps_list,
        fc_maps_list,
        is_3d=True,
        dropout=0.5,
        contrast_dependent=True,
        out_feat_level=-1,
        conv_kernel=3,
        conv_padding=1,
        conv_stride=1,
        pool_kernel_stride=2,
        *kwargs,
    ):
        super().__init__()
        self.out_feat_level = out_feat_level
        self.contrast_dependent = contrast_dependent
        if self.contrast_dependent:
            f_maps_list[0] += 1  # add one input image/contrast channel

        layers = []
        for i, in_feature_num in enumerate(f_maps_list[:-1]):
            layer = ConvNormPoolBlock(
                in_feature_num,
                f_maps_list[i + 1],
                is_3d=is_3d,
                conv_kernel=conv_kernel,
                conv_padding=conv_padding,
                conv_stride=conv_stride,
                pool_kernel=pool_kernel_stride,
                pool_stride=pool_kernel_stride,
            )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)
        self.out_name = list(out_channels.keys())[0]
        n_input_fc = f_maps_list[-1] * np.prod(
            [
                np.array(in_shape[-3:])
                // (
                    (pool_kernel_stride * conv_stride)
                    ** (len(f_maps_list) - 1)
                )
            ]
        )

        self.in_shape = in_shape
        self.fc = FCDropout(
            n_input_fc,
            n_classes,
            fc_maps_list,
            dropout=dropout,
        )

    def forward(self, x, image):
        x = x[self.out_feat_level]
        if self.out_feat_level < -1:
            scale_factor = 2 ** (-1 - self.out_feat_level)
            image = torch.nn.functional.interpolate(
                image, scale_factor=1 / scale_factor, mode="trilinear"
            )

        assert (
            list(x.shape[-3:]) == self.in_shape[-3:]
        ), f"Input shape mismatch: expected {self.in_shape[-3:]} but got {x.shape[-3:]}."
        if self.contrast_dependent:
            x = torch.cat([x, image], dim=1)
        x = self.layers(x)
        x = self.fc(x)
        return {self.out_name: x}


class ScalarAggHead(nn.Module):
    """
    Task-specific head that takes a list of sample features as inputs
    """

    def __init__(
        self,
        n_classes,
        out_channels,
        f_maps_list,
        fc_maps_list,
        is_3d=True,
        dropout=0.5,
        contrast_dependent=True,
        out_feat_level=-1,
    ):
        super().__init__()
        self.out_feat_level = out_feat_level
        self.contrast_dependent = contrast_dependent
        if self.contrast_dependent:
            # Add one input image/contrast channel to each feature map
            f_maps_list = [f + 1 for f in f_maps_list]
        in_channels = sum(f_maps_list)

        if is_3d:
            self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Layers will just be a concatenation of the pooled features and two fc layers
        self.out_name = list(out_channels.keys())[0]
        self.fc = FCDropout(
            in_channels,
            n_classes,
            fc_maps_list,
            dropout=dropout,
        )

    def forward(self, x, image):
        # Take the input x

        # Apply channel-wise sigmoid to the input x
        # x = [torch.sigmoid(feat) for feat in x]
        # Apply channel-wise 0-1 normalization to the input x without sigmoids

        x = [
            (xi - xi.amin(dim=(2, 3, 4), keepdim=True))
            / (
                xi.amax(dim=(2, 3, 4), keepdim=True)
                - xi.amin(dim=(2, 3, 4), keepdim=True)
                + 1e-8
            )
            for xi in x
        ]

        if self.contrast_dependent:
            # Take the input image and downsample by a factor 2 for each feature map
            image_scaled = [
                torch.nn.functional.interpolate(
                    image, scale_factor=1 / (2**i), mode="trilinear"
                )
                for i in range(len(x))
            ]

            # Concatenate the input x and the scaled image
            x = [
                torch.cat([feat, img], dim=1)
                for feat, img in zip(x, image_scaled[::-1])
            ]

        # Take x and do global pooling of each element in the list
        x = [self.global_pool(feat) for feat in x]
        # Concatenate the pooled features
        x = torch.cat(x, dim=1)

        x = self.fc(x)

        return {self.out_name: x}


################################


def get_head(args, f_maps_list, out_channels, is_3d, out_feat_level):
    task = args.task

    if "feat" in task:
        return IndepHead(
            args, f_maps_list, out_channels, is_3d, out_feat_level
        )
    else:
        if "qc" in task:
            return ScalarHead(
                args,
                args.in_shape,
                args.n_classes,
                out_channels,
                f_maps_list,
                args.fc_maps_list,
                is_3d,
                out_feat_level=out_feat_level,
                pool_kernel_stride=args.pool_kernel_stride,
            )
        elif "sr" in task or "bf" in task:
            return MultiInputDepHead(
                args, f_maps_list, out_channels, is_3d, out_feat_level
            )
        else:
            return MultiInputIndepHead(
                args, f_maps_list, out_channels, is_3d, out_feat_level
            )
