import numpy as np
import torch
from torch import nn
from BrainID import N_IQMS
from monai.networks import nets

"""
Using the CNN architecture of ClinicaDL (https://github.com/aramis-lab/clinicadl)
"""

import csv


class Reshape(nn.Module):
    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(*self.size)

class FeatureDropout(nn.Module):
    """Channel-wise dropout for [B, F] IQM tensors. Expectation-preserving."""
    def __init__(self, p: float = 0.2):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        B, F = x.shape
        keep = 1.0 - self.p
        mask = (torch.rand(B, F, device=x.device) < keep).float()
        return x * mask / keep
    
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


def get_layers_fn(input_size, norm="batch"):
    if len(input_size) == 4:
        if norm == "batch":
            return nn.Conv3d, nn.BatchNorm3d, PadMaxPool3d
        elif norm == "instance":
            return nn.Conv3d, nn.InstanceNorm3d, PadMaxPool3d

    elif len(input_size) == 3:
        if norm == "batch":
            return nn.Conv2d, nn.BatchNorm2d, PadMaxPool2d
        elif norm == "instance":
            return nn.Conv2d, nn.InstanceNorm2d, PadMaxPool2d
    else:
        raise ValueError(
            f"The input is neither a 2D or 3D image.\n "
            f"Input shape is {input_size - 1}."
        )


class Conv5_FC3(nn.Module):
    """
    It is a convolutional neural network with 5 convolution and 3 fully-connected layer.
    It reduces the 2D or 3D input image to an array of size output_size.
    """

    def __init__(
        self,
        input_size,
        nconv=8,
        norm="batch",
        gpu=True,
        output_size=2,
        dropout=0.5,
        feature_drop=0.2,
        use_iqms=False,
    ):
        super().__init__()
        input_size = list(input_size)
        conv, norm, pool = get_layers_fn(input_size, norm)
        self.convolutions = nn.Sequential(
            conv(input_size[0], nconv, 3, padding=1),
            norm(nconv),
            nn.ReLU(),
            pool(2, 2),
            conv(nconv, nconv * 2, 3, padding=1),
            norm(nconv * 2),
            nn.ReLU(),
            pool(2, 2),
            conv(nconv * 2, nconv * 4, 3, padding=1),
            norm(nconv * 4),
            nn.ReLU(),
            pool(2, 2),
            conv(nconv * 4, nconv * 8, 3, padding=1),
            norm(nconv * 8),
            nn.ReLU(),
            pool(2, 2),
            conv(nconv * 8, nconv * 16, 3, padding=1),
            norm(nconv * 16),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d(1),
        )
        self.use_iqms = use_iqms
        # self.prepro_fc_iqms = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(N_IQMS, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        # )

        # add_linear = 128 if self.use_iqms else 0
        n_linear = nconv * 16
        # Compute the size of the first FC layer
        # input_tensor = torch.zeros(input_size).unsqueeze(0)
        # output_convolutions = self.convolutions(input_tensor)
        self.counter = 0
        if self.use_iqms:
            self.feature_drop = feature_drop
            self.cnn_proj = nn.Sequential(
                nn.LayerNorm(nconv * 16),
                nn.Linear(nconv * 16, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.iqm_proj = nn.Sequential(
                FeatureDropout(p=self.feature_drop),
                nn.LayerNorm(N_IQMS),
                nn.Linear(N_IQMS, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            n_linear = 256
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(n_linear, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def _freeze_cnn(self):
        for param in self.convolutions.parameters():
            param.requires_grad = False

    def _unfreeze_cnn(self):
        for param in self.convolutions.parameters():
            param.requires_grad = True

    def forward(self, x, iqms=None):
        x = self.convolutions(x)
        if self.use_iqms:

            x = torch.cat(
                (self.cnn_proj(x.view(1, -1)), self.iqm_proj(iqms)), dim=1
            )

        x = self.fc(x)
        return x

    def predict(self, x):
        return self.forward(x)


class Conv3_FC2(nn.Module):
    """
    It is a convolutional neural network with 5 convolution and 3 fully-connected layer.
    It reduces the 2D or 3D input image to an array of size output_size.
    """

    def __init__(
        self,
        input_size,
        norm="batch",
        gpu=True,
        output_size=2,
        dropout=0.5,
        use_iqms=False,
    ):
        super().__init__()
        input_size = list(input_size)
        conv, norm, pool = get_layers_fn(input_size, norm)
        self.convolutions = nn.Sequential(
            conv(input_size[0], 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),
            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),
            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            # Do global max pooling
            nn.AdaptiveMaxPool3d(1),
        )
        self.use_iqms = use_iqms
        n_linear = 128
        if self.use_iqms:
            self.cnn_proj = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.iqm_proj = nn.Sequential(
                FeatureDropout(p=dropout),
                nn.LayerNorm(N_IQMS),
                nn.Linear(N_IQMS, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # self.cnn_proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            # self.iqm_proj = nn.Sequential(nn.Linear(N_IQMS, 128), nn.ReLU())
            n_linear = 256

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(n_linear, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x, iqms=None):
        x = self.convolutions(x)
        if self.use_iqms:
            x = torch.cat(
                (self.cnn_proj(x.view(1, -1)), self.iqm_proj(iqms)), dim=1
            )

        x = self.fc(x)
        return x

    def predict(self, x):
        return self.forward(x)


class DenseNet121(nets.DenseNet121):
    def __init__(self, use_iqms: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_feature_dim = N_IQMS

        # Get the current Linear layer input/output sizes
        original_linear: nn.Linear = self.class_layers[-1]
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.use_iqms = use_iqms
        # n_linear = 128

        self.cnn_proj = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(in_features, 128), nn.ReLU()
        )
        self.iqm_proj = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(N_IQMS, 128), nn.ReLU()
        )

        self.class_layers[-1] = nn.Linear(256, out_features)

    def forward(
        self, x: torch.Tensor, iqms: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.features(x)

        # Apply ReLU, Adaptive Pool, and Flatten
        for name, layer in self.class_layers.named_children():
            if name == "out":
                break
            x = layer(x)

        x = x.view(x.size(0), -1)

        # Concatenate external features if provided
        if iqms is not None:
            x = torch.cat(
                [self.cnn_proj(x.view(1, -1)), self.iqm_proj(iqms)], dim=1
            )

        x = self.class_layers[-1](x)
        return x


class ModelWrapper(nn.Module):

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.use_iqms = model.use_iqms

    def freeze_cnn(self):
        self.model._freeze_cnn()

    def unfreeze_cnn(self):
        self.model._unfreeze_cnn()

    def forward(self, input_list, iqms=None):
        outs = []
        if not isinstance(input_list, list):
            input_list = [input_list]
            if iqms is not None:
                iqms = [iqms]

        for i, x in enumerate(input_list):
            x = self.model(x, iqms[i] if self.use_iqms else None)
            outs.append({"pred": x})
        return outs, input_list

    def predict(self, x):
        return self.forward(x)


class IQMOnlyHead(nn.Module):
    def __init__(
        self,
        d_iqm: int,
        d_hidden: int = 128,
        dropout: float = 0.2,
        feat_dropout: float = 0.0,  # 0 disables feature dropout
        output_size: int = 2,
        use_iqms: bool = True
    ):
        super().__init__()
        layers = []
        if feat_dropout > 0:
            layers.append(FeatureDropout(p=feat_dropout))
        layers.extend([
            nn.LayerNorm(d_iqm),
            nn.Linear(d_iqm, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, output_size)
        ])
        self.net = nn.Sequential(*layers)
        self.use_iqms = use_iqms
        assert self.use_iqms, "IQMOnlyHead is designed to work with IQMs only"
    def forward(self, x, iqms=None):
        if iqms is None:
            raise ValueError("IQMOnlyHead expects iqms tensor (shape [B, d_iqm])")
        return self.net(iqms).squeeze(1)