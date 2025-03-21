"""
Wrapper interface.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb


class UncertaintyProcessor(nn.Module):
    def __init__(self, output_names):
        super(UncertaintyProcessor, self).__init__()
        self.output_names = output_names

    def forward(self, outputs, *kwargs):
        for output_name in self.output_names:
            for output in outputs:
                output[output_name + "_sigma"] = output[output_name][:, 1][
                    :, None
                ]
                output[output_name] = output[output_name][:, 0][:, None]
        return outputs


class SegProcessor(nn.Module):
    def __init__(self):
        super(SegProcessor, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs, *kwargs):
        for output in outputs:
            output["seg"] = self.softmax(output["seg"])
        return outputs


class ContrastiveProcessor(nn.Module):
    def __init__(self):
        """
        Ref: https://openreview.net/forum?id=2oCb0q5TA4Y
        """
        super(ContrastiveProcessor, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs, *kwargs):
        for output in outputs:
            output["feat"][-1] = F.normalize(output["feat"][-1], dim=1)
        return outputs


class BFProcessor(nn.Module):
    def __init__(self):
        super(BFProcessor, self).__init__()

    def forward(self, outputs, *kwargs):
        for output in outputs:
            output["bias_field"] = torch.exp(output["bias_field_log"])
        return outputs


##############################################################################


class MultiJoiner(nn.Module):
    def __init__(self, backbone, head, freeze_feat=False):
        super(MultiJoiner, self).__init__()

        self.backbone = backbone
        self.head = head
        self.freeze_feat = freeze_feat

    def format_head_input(self, feat, x):
        out = self.pick_head_input(feat, x)
        if not isinstance(out[0], list):
            # Need the first head input to be feat, a list of tensors.
            # As the head input will be unpacked in the head, we need to wrap it in a list.
            out = [out]
        return out

    def pick_head_input(self, feat, x):
        raise NotImplementedError

    def forward(self, input_list):
        outs = []

        for x in input_list:
            if self.freeze_feat:
                with torch.no_grad():
                    feat = self.backbone.get_feature(x["input"])
            else:
                feat = self.backbone.get_feature(x["input"])
            out = {"feat": feat}
            if self.head is not None:

                head_input = self.format_head_input(feat, x)
                out.update(self.head(*head_input))
            outs.append(out)
        return outs, [input["input"] for input in input_list]

    def train(self):
        self.backbone.train()
        if self.head is not None:
            self.head.train()

    def eval(self):
        self.backbone.eval()
        if self.head is not None:
            self.head.eval()


class MultiInputIndepJoiner(MultiJoiner):
    """
    Perform forward pass separately on each augmented input.
    """

    def __init__(self, backbone, head, freeze_feat=False):
        super(MultiInputIndepJoiner, self).__init__(
            backbone, head, freeze_feat
        )

    def pick_head_input(self, feat, x):
        return feat


class MultiInputDepJoiner(MultiJoiner):
    """
    Perform forward pass separately on each augmented input.
    """

    def __init__(self, backbone, head, freeze_feat=False):
        super(MultiInputDepJoiner, self).__init__(backbone, head, freeze_feat)

    def pick_head_input(self, feat, x):
        return feat, x["input"]


################################


def get_processors(args, task, device):
    processors = []
    if args.losses.uncertainty is not None:
        processors.append(UncertaintyProcessor(args.output_names).to(device))
    if "contrastive" in task:
        processors.append(ContrastiveProcessor().to(device))
    if "seg" in task:
        processors.append(SegProcessor().to(device))
    if "bf" in task:
        processors.append(BFProcessor().to(device))
    return processors


def get_joiner(task, backbone, head, freeze_feat=False):
    if "sr" in task or "bf" in task or "qc" in task:
        return MultiInputDepJoiner(backbone, head, freeze_feat)
    else:
        return MultiInputIndepJoiner(backbone, head, freeze_feat)
