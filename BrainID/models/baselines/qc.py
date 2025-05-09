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
        self, input_size, norm="batch", gpu=True, output_size=2, dropout=0.5
    ):
        super().__init__()
        input_size = list(input_size)
        conv, norm, pool = get_layers_fn(input_size, norm)
        self.convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),
            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),
            conv(16, 32, 3, padding=1),
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
            pool(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = self.convolutions(input_tensor)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
        )

    def forward(self, input_list):
        outs = []
        if not isinstance(input_list, list):
            input_list = [input_list]
        for x in input_list:
            x = self.convolutions(x)
            x = self.fc(x)
            outs.append({"pred": x})
        return outs, x

    def predict(self, x):
        return self.forward(x)
    



