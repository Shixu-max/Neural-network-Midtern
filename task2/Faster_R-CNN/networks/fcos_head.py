import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss

from . import det_utils


def _loss_inter_union(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

    return intsctk, unionk


def _upcast_non_float(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.dtype not in (torch.float32, torch.float64):
        return t.float()
    return t


def generalized_box_iou_loss(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-7,
) -> torch.Tensor:
    """
    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``, and The two boxes should have the
    same dimensions.

    Args:
        boxes1 (Tensor[N, 4] or Tensor[4]): first set of boxes
        boxes2 (Tensor[N, 4] or Tensor[4]): second set of boxes
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: No reduction will be
            applied to the output. ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``
        eps (float): small number to prevent division by zero. Default: 1e-7

    Returns:
        Tensor: Loss tensor with the reduction option applied.

    Reference:
        Hamid Rezatofighi et al.: Generalized Intersection over Union:
        A Metric and A Loss for Bounding Box Regression:
        https://arxiv.org/abs/1902.09630
    """

    # Original implementation from https://github.com/facebookresearch/fvcore/blob/bfff2ef/fvcore/nn/giou_loss.py

    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)
    intsctk, unionk = _loss_inter_union(boxes1, boxes2)
    iouk = intsctk / (unionk + eps)

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))

    loss = 1 - miouk

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class FCOSHead(nn.Module):
    """
    A regression and classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer of head. Default: 4.
    """

    __annotations__ = {
        "box_coder": det_utils.BoxLinearCoder,
    }

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: Optional[int] = 4) -> None:
        super().__init__()
        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)
        self.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes, num_convs)
        self.regression_head = FCOSRegressionHead(in_channels, num_anchors, num_convs)

    def compute_loss(
            self,
            targets: List[Dict[str, Tensor]],
            head_outputs: Dict[str, Tensor],
            anchors: List[Tensor],
            matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:

        cls_logits = head_outputs["cls_logits"]  # [N, HWA, C]
        bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, HWA, 1]

        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image),))
                gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4))
            else:
                gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)]
                gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]
            gt_classes_targets[matched_idxs_per_image < 0] = -1  # background
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)

        # List[Tensor] to Tensor conversion of  `all_gt_boxes_target`, `all_gt_classes_targets` and `anchors`
        all_gt_boxes_targets, all_gt_classes_targets, anchors = (
            torch.stack(all_gt_boxes_targets),
            torch.stack(all_gt_classes_targets),
            torch.stack(anchors),
        )

        # compute foregroud
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # amp issue: pred_boxes need to convert float
        pred_boxes = self.box_coder.decode(bbox_regression, anchors)

        # regression loss: GIoU loss
        loss_bbox_reg = generalized_box_iou_loss(
            pred_boxes[foregroud_mask],
            all_gt_boxes_targets[foregroud_mask],
            reduction="sum",
        )

        # ctrness loss

        bbox_reg_targets = self.box_coder.encode(anchors, all_gt_boxes_targets)

        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )
        pred_centerness = bbox_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[foregroud_mask], gt_ctrness_targets[foregroud_mask], reduction="sum"
        )

        return {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits = self.classification_head(x)
        bbox_regression, bbox_ctrness = self.regression_head(x)
        return {
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "bbox_ctrness": bbox_ctrness,
        }


class FCOSClassificationHead(nn.Module):
    """
    A classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature.
        num_anchors (int): number of anchors to be predicted.
        num_classes (int): number of classes to be predicted.
        num_convs (Optional[int]): number of conv layer. Default: 4.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
        norm_layer: Module specifying the normalization layer to use.
    """

    def __init__(
            self,
            in_channels: int,
            num_anchors: int,
            num_classes: int,
            num_convs: int = 4,
            prior_probability: float = 0.01,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

    def forward(self, x: List[Tensor]) -> Tensor:
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class FCOSRegressionHead(nn.Module):
    """
    A regression head for use in FCOS, which combines regression branch and center-ness branch.
    This can obtain better performance.

    Reference: `FCOS: A simple and strong anchor-free object detector <https://arxiv.org/abs/2006.09214>`_.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 4.
        norm_layer: Module specifying the normalization layer to use.
    """

    def __init__(
            self,
            in_channels: int,
            num_anchors: int,
            num_convs: int = 4,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.bbox_ctrness = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)
        for layer in [self.bbox_reg, self.bbox_ctrness]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.zeros_(layer.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        all_bbox_regression = []
        all_bbox_ctrness = []

        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = nn.functional.relu(self.bbox_reg(bbox_feature))
            bbox_ctrness = self.bbox_ctrness(bbox_feature)

            # permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)
            all_bbox_regression.append(bbox_regression)

            # permute bbox ctrness output from (N, 1 * A, H, W) to (N, HWA, 1).
            bbox_ctrness = bbox_ctrness.view(N, -1, 1, H, W)
            bbox_ctrness = bbox_ctrness.permute(0, 3, 4, 1, 2)
            bbox_ctrness = bbox_ctrness.reshape(N, -1, 1)
            all_bbox_ctrness.append(bbox_ctrness)

        return torch.cat(all_bbox_regression, dim=1), torch.cat(all_bbox_ctrness, dim=1)
