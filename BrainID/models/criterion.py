"""
Criterion modules.
"""

import numpy as np
import torch
import torch.nn as nn
import pdb
from BrainID.models.losses import GradientLoss, gaussian_loss, laplace_loss
from monai.losses import DiceCELoss

uncertainty_loss = {"gaussian": gaussian_loss, "laplace": laplace_loss}


class SetScalarCriterion(nn.Module):

    def __init__(self, loss_dict, device):
        super(SetScalarCriterion, self).__init__()
        # self.args = args
        self.weight_dict = loss_dict
        self.loss_names = list(loss_dict.keys())

        self.loss_map = {
            "l1": self.loss_l1,
            "l2": self.loss_l2,
            "ce": self.loss_ce,
        }
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.device = device

    def loss_ce(self, outputs, targets, *kwargs):
        loss_ce = self.ce(outputs["pred"], targets["label"].to(self.device))
        return {"loss_ce": loss_ce}

    def loss_l1(self, outputs, targets, *kwargs):
        loss_l1 = self.l1(outputs["pred"], targets["label"].to(self.device))
        return {"loss_l1": loss_l1}

    def loss_l2(self, outputs, targets, *kwargs):
        loss_l2 = self.l2(outputs["pred"], targets["label"].to(self.device))
        return {"loss_l2": loss_l2}

    def get_loss(self, loss_name, outputs_list, targets, samples_list):
        assert (
            loss_name in self.loss_map
        ), f"do you really want to compute {loss_name} loss?"
        total_loss = 0.0
        for i_sample, outputs in enumerate(outputs_list):
            total_loss += self.loss_map[loss_name](
                outputs, targets, samples_list[i_sample]
            )["loss_" + loss_name]
        return {"loss_" + loss_name: total_loss / len(outputs_list)}

    def get_losses(self, outputs, targets, *kwargs):
        losses = {}
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, outputs, targets, *kwargs))
        return losses

    def aggregate_losses(self, losses_dict):
        weight_dict = {"loss_" + k: v for k, v in self.weight_dict.items()}
        losses = sum(
            losses_dict[k] * weight_dict[k]
            for k in losses_dict.keys()
            if k in weight_dict
        )
        return losses

    def forward(self, outputs_list, targets, *kwargs):
        losses_dict = self.get_losses(outputs_list, targets, *kwargs)
        losses_dict["loss"] = self.aggregate_losses(losses_dict)
        return losses_dict


class SetCriterion(nn.Module):
    """
    This class computes the loss for BrainID.
    """

    def __init__(self, loss_dict, params_dict, device):
        """Create the criterion.
        Parameters:
            args: general exp cfg
            loss_weights: dict containing as key the names of the losses and as values their
                         relative weight.
            loss_names: list of all the losses to be applied. See get_loss for list of
                    available loss_names.
        """
        super(SetCriterion, self).__init__()
        self.weight_dict = loss_dict
        self.loss_names = list(loss_dict.keys())
        self.mse = nn.MSELoss()
        self.dice_ce = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )

        self.loss_regression_type = (
            params_dict.uncertainty
            if params_dict.uncertainty is not None
            else "l1"
        )
        self.loss_regression = (
            uncertainty_loss[params_dict.uncertainty]
            if params_dict.uncertainty is not None
            else nn.L1Loss()
        )

        self.grad = GradientLoss("l1")

        self.bflog_loss = (
            nn.L1Loss()
            if params_dict.bias_field_log_type == "l1"
            else self.mse
        )

        if "contrastive" in self.loss_names:
            self.temp_alpha = params_dict.contrastive_temperatures.alpha
            self.temp_beta = params_dict.contrastive_temperatures.beta
            self.temp_gamma = params_dict.contrastive_temperatures.gamma

        self.loss_map = {
            # "seg_ce": self.loss_seg_ce,
            # "seg_dice": self.loss_seg_dice,
            "seg_dice_ce": self.loss_seg_dice_ce,
            "dist": self.loss_dist,
            "sr": self.loss_sr,
            "sr_grad": self.loss_sr_grad,
            "image": self.loss_image,
            "image_grad": self.loss_image_grad,
            "bias_field_log": self.loss_bias_field_log,
            "supervised_seg": self.loss_supervised_seg,
            "contrastive": self.loss_feat_contrastive,
        }

    def loss_feat_contrastive(self, outputs, *kwargs):
        """
        outputs: [feat1, feat2]
        feat shape: (b, feat_dim, s, r, c)
        """
        feat1, feat2 = outputs[0]["feat"][-1], outputs[1]["feat"][-1]
        num = torch.sum(torch.exp(feat1 * feat2 / self.temp_alpha), dim=1)
        den = torch.zeros_like(feat1[:, 0])
        for i in range(feat1.shape[1]):
            den1 = torch.exp(feat1[:, i] ** 2 / self.temp_beta)
            den2 = torch.exp(
                (
                    torch.sum(feat1[:, i][:, None] * feat1, dim=1)
                    - feat1[:, i] ** 2
                )
                / self.temp_gamma
            )
            den += den1 + den2
        loss_contrastive = torch.mean(-torch.log(num / den))
        return {"loss_contrastive": loss_contrastive}

    def loss_seg_ce(self, outputs, targets, *kwargs):
        """
        Cross entropy of segmentation
        """
        nlabels = torch.unique(targets["label"][targets["label" > 0]])
        nlabels = nlabels[nlabels != 0]

        loss_seg_ce = torch.mean(
            -torch.sum(
                torch.log(torch.clamp(outputs["seg"], min=1e-5))
                * self.weights_ce
                * targets["label"],
                dim=1,
            )
        )
        return {"loss_seg_ce": loss_seg_ce}

    def loss_seg_dice_ce(self, outputs, targets, *kwargs):
        """
        Dice of segmentation
        """

        loss_seg_dice_ce = self.dice_ce(outputs["seg"], targets["label"])
        return {"loss_seg_dice_ce": loss_seg_dice_ce}

    def loss_dist(self, outputs, targets, *kwargs):
        loss_dist = self.mse(outputs["dist"], targets["dist"])
        return {"loss_image": loss_dist}

    def loss_sr(self, outputs, targets, samples):
        if self.loss_regression_type != "l1":
            loss_sr = self.loss_regression(
                outputs["image"], outputs["image_sigma"], samples["orig"]
            )
        else:
            loss_sr = self.loss_regression(outputs["image"], samples["orig"])
        return {"loss_sr": loss_sr}

    def loss_sr_grad(self, outputs, targets, samples):
        loss_sr_grad = self.grad(outputs["image"], samples["orig"])
        return {"loss_sr_grad": loss_sr_grad}

    def loss_image(self, outputs, targets, *kwargs):
        if self.loss_regression_type != "l1":
            loss_image = self.loss_regression(
                outputs["image"], outputs["image_sigma"], targets["image"]
            )
        else:
            loss_image = self.loss_regression(
                outputs["image"], targets["image"]
            )
        return {"loss_image": loss_image}

    def loss_image_grad(self, outputs, targets, *kwargs):
        loss_image_grad = self.grad(outputs["image"], targets["image"])
        return {"loss_image_grad": loss_image_grad}

    def loss_bias_field_log(self, outputs, targets, samples):
        bf_soft_mask = 1.0 - targets["seg"][:, 0]
        loss_bias_field_log = self.bflog_loss(
            outputs["bias_field_log"] * bf_soft_mask,
            samples["bias_field_log"] * bf_soft_mask,
        )
        return {"loss_bias_field_log": loss_bias_field_log}

    def loss_supervised_seg(self, outputs, targets, *kwargs):
        """
        Supervised segmentation differences (for dataset_name == synth)
        """
        onehot_withoutcsf = targets["seg"].clone()
        onehot_withoutcsf = onehot_withoutcsf[:, self.csf_v, ...]
        onehot_withoutcsf[:, 0, :, :, :] = (
            onehot_withoutcsf[:, 0, :, :, :]
            + targets["seg"][:, self.csf_ind, :, :, :]
        )

        loss_supervised_seg = torch.sum(
            self.weights_dice_sup
            * (
                1.0
                - 2.0
                * (
                    (outputs["supervised_seg"] * onehot_withoutcsf).sum(
                        dim=[2, 3, 4]
                    )
                )
                / torch.clamp(
                    (outputs["supervised_seg"] + onehot_withoutcsf).sum(
                        dim=[2, 3, 4]
                    ),
                    min=1e-5,
                )
            )
        )

        return {"loss_supervised_seg": loss_supervised_seg}

    def get_loss(self, loss_name, outputs, targets, *kwargs):
        assert (
            loss_name in self.loss_map
        ), f"do you really want to compute {loss_name} loss?"
        return self.loss_map[loss_name](outputs, targets, *kwargs)

    def get_losses(self, outputs, targets, *kwargs):
        losses = {}
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, outputs, targets, *kwargs))
        return losses

    def aggregate_losses(self, losses_dict):
        weight_dict = {"loss_" + k: v for k, v in self.weight_dict.items()}
        losses = sum(
            losses_dict[k] * weight_dict[k]
            for k in losses_dict.keys()
            if k in weight_dict
        )
        return losses

    def forward(self, outputs, targets, *kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """

        losses_dict = self.get_losses(outputs, targets, *kwargs)
        losses_dict["loss"] = self.aggregate_losses(losses_dict)
        return losses_dict


class SetMultiCriterion(SetCriterion):
    """
    This class computes the loss for BrainID with a list of results as inputs.
    """

    def __init__(self, loss_dict, params_dict, device):
        """Create the criterion.
        Parameters:
            args: general exp cfg
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            loss_names: list of all the losses to be applied. See get_loss for list of
                    available loss_names.
        """
        super(SetMultiCriterion, self).__init__(loss_dict, params_dict, device)

    def get_loss(self, loss_name, outputs_list, targets, samples_list):
        assert (
            loss_name in self.loss_map
        ), f"do you really want to compute {loss_name} loss?"
        total_loss = 0.0
        all_samples = len(outputs_list)
        for i_sample, outputs in enumerate(outputs_list):
            total_loss += self.loss_map[loss_name](
                outputs, targets, samples_list[i_sample]
            )["loss_" + loss_name]
        return {"loss_" + loss_name: total_loss / all_samples}

    def get_losses(self, outputs_list, targets, samples_list):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss_name in self.loss_names:
            losses.update(
                self.get_loss(loss_name, outputs_list, targets, samples_list)
            )
        return losses

    def forward(self, outputs_list, targets, samples_list):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        losses_dict = self.get_losses(outputs_list, targets, samples_list)
        loss = self.aggregate_losses(losses_dict)
        losses_dict["loss"] = loss
        return losses_dict
