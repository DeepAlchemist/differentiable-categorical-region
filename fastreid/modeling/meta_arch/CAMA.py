# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import math
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

def binary_conversion_v0(activ_map,
                         v_filter,
                         s_filter,
                         block_size=3):
    """ Generate the binary mask (activate bg)

    Args:
        activ_map(4D Tensor): (b 1 h w)
    """
    b, ch, h, w = activ_map.size()
    soft_mask = activ_map
    activ_map = torch.mean(activ_map, -1, keepdim=True)  # (b 1 h 1)
    activ_map = F.conv2d(input=activ_map,
                         weight=v_filter,
                         padding=(block_size // 2, 0))  # (b 1 h 1)
    if block_size % 2 == 0:
        activ_map = activ_map[:, :, :-1]

    max_index = torch.argmax(activ_map.view(b, h), dim=1)  # (b,)

    # generate the stripe mask
    stripe_masks = torch.zeros([b, h], device=activ_map.device)
    batch_index = torch.arange(0, b, dtype=torch.long)
    stripe_masks[batch_index, max_index] = 1

    stripe_masks = F.max_pool2d(input=stripe_masks[:, None, :, None],
                                kernel_size=(block_size, 1),
                                stride=(1, 1),
                                padding=(block_size // 2, 0))  # (b 1 h 1)
    if block_size % 2 == 0:
        stripe_masks = stripe_masks[:, :, 1:]

    stripe_masks = stripe_masks.view(b, 1, h, 1).expand_as(soft_mask)  # (b 1 h w)

    # process soft mask
    soft_mask = soft_mask * 0.1
    soft_mask = torch.softmax(soft_mask.view(b, -1), dim=1).view(b, 1, h, w)

    # smoothing
    soft_mask = F.conv2d(input=soft_mask,
                         weight=s_filter,
                         padding=block_size // 2)  # (b 1 h w)
    return stripe_masks, soft_mask

def binary_conversion_v1(activ_map,
                         v_filter,
                         s_filter,
                         block_size=3):
    """ Generate the binary mask (cross-entropy loss can not converge)
        The number of stripes is 3, with kernel_size=(6, 8), stride=5, padding=0,
        The number of stripes is 4, with kernel_size=(4, 8), stride=4, padding=0
        Use sub-sampled soft mask
    Args:
        activ_map(4D Tensor): (b 1 h w)
    """
    activ_map = F.conv2d(input=activ_map,
                         weight=s_filter,
                         stride=(4, 1),
                         padding=0)  # (b 1 h w)

    b, ch, h, w = activ_map.size()
    max_index = torch.argmax(activ_map.view(b, -1), dim=1)  # (b h*w)

    # generate the stripe mask
    stripe_masks = torch.zeros([b, h * w], device=activ_map.device)
    batch_index = torch.arange(0, b, dtype=torch.long)
    stripe_masks[batch_index, max_index] = 1
    stripe_masks = stripe_masks.view(b, 1, h, w)  # (b 1 h w)

    # process soft mask
    activ_map = activ_map * 0.1
    activ_map = torch.softmax(activ_map.view(b, -1), dim=1).view(b, 1, h, w)

    return stripe_masks, activ_map

def binary_conversion_v2(activ_map,
                         v_filter,
                         s_filter,
                         block_size=3):
    """ Generate the binary mask (activate bg)
        The number of stripes is 3, with kernel_size=(6, 8), stride=5, padding=0
    Args:
        activ_map(4D Tensor): (b 1 h w)
    """
    soft_mask = activ_map
    activ_map = F.conv2d(input=activ_map,
                         weight=s_filter,
                         stride=(5, 1),
                         padding=0)  # (b 1 h w)

    b, ch, h, w = activ_map.size()
    max_index = torch.argmax(activ_map.view(b, -1), dim=1)  # (b h*w)

    # generate the stripe mask
    stripe_masks = torch.zeros([b, h * w], device=activ_map.device)
    batch_index = torch.arange(0, b, dtype=torch.long)
    stripe_masks[batch_index, max_index] = 1
    stripe_masks = stripe_masks.view(b, 1, h, w)  # (b 1 h w)
    stripe_masks = F.interpolate(stripe_masks, size=soft_mask.size()[-2:], mode='bilinear', align_corners=False)

    # process soft mask
    h, w = soft_mask.size(2), soft_mask.size(3)
    soft_mask = soft_mask * 0.1
    soft_mask = torch.softmax(soft_mask.view(b, -1), dim=1).view(b, 1, h, w)

    return stripe_masks, soft_mask

def binary_conversion_v3(activ_map,
                         v_filter,
                         s_filter,
                         block_size=3):
    """ Generate the binary mask (activate bg)
        The number of stripes is 6, with kernel_size=(3, 8), stride=3, padding=1
    Args:
        activ_map(4D Tensor): (b 1 h w)
    """
    soft_mask = activ_map
    activ_map = F.conv2d(input=activ_map,
                         weight=s_filter,
                         stride=(3, 1),
                         padding=1)  # (b 1 h w)

    b, ch, h, w = activ_map.size()
    max_index = torch.argmax(activ_map.view(b, -1), dim=1)  # (b h*w)

    # generate the stripe mask
    stripe_masks = torch.zeros([b, h * w], device=activ_map.device)
    batch_index = torch.arange(0, b, dtype=torch.long)
    stripe_masks[batch_index, max_index] = 1
    stripe_masks = stripe_masks.view(b, 1, h, w)  # (b 1 h w)
    stripe_masks = F.interpolate(stripe_masks, size=soft_mask.size()[-2:], mode='bilinear', align_corners=False)

    # process soft mask
    h, w = soft_mask.size(2), soft_mask.size(3)
    soft_mask = soft_mask * 0.1
    soft_mask = torch.softmax(soft_mask.view(b, -1), dim=1).view(b, 1, h, w)

    return stripe_masks, soft_mask

def overlap_activ_penalty(activ_maps):
    num_branch, b, h, w = activ_maps.size()
    loss = 0
    cnt = 0
    for ii in range(num_branch - 1):
        ref_map = activ_maps[ii].detach()  # (b h w)
        for jj in range(ii + 1, num_branch):
            cnt += 1
            cur_map = activ_maps[jj]
            with torch.no_grad():
                keep = (ref_map.sum(-1).sum(-1) > 0) & (cur_map.sum(-1).sum(-1) > 0)
            loss += torch.sum(ref_map[keep] * cur_map[keep])
    loss = loss / cnt if cnt > 1 else loss

    return loss

def generate_activ_maps(feat_maps, weights, targets, v_filter, s_filter, block_size):
    """
    feat_maps(List(Tensor(b c h w))): len=num_branch
    weights(List(2D Tensor)): (num_class d)
    targets(Tensor):

    Returns:
        n_branch_activ_maps(List(3D Tensor)): (b h w) len=num_branch
    """
    n_branch_activ_maps = []
    # Traversing branches
    for ii, feat_map in enumerate(feat_maps):
        # [b d] to [b d 1 1]
        W = weights[ii][targets, :][:, :, None, None]
        # [b d h w] to [b h w]
        activ_maps = (feat_map * W).sum(dim=1)
        n_branch_activ_maps.append(activ_maps)

    activ_maps = torch.stack(n_branch_activ_maps, dim=0)  # (num_branch b h w)

    num_branch, b, h, w = activ_maps.size()
    stripe_masks, soft_masks = binary_conversion_v1(activ_maps.view(-1, 1, h, w),
                                                    v_filter, s_filter, block_size)
    stripe_mask = stripe_masks.view(num_branch, b, stripe_masks.size(-2), stripe_masks.size(-1))  # retain foreground
    soft_masks = soft_masks.view(num_branch, b, soft_masks.size(-2), soft_masks.size(-1))

    return stripe_mask, soft_masks

class MultiBranch(nn.Module):
    def __init__(self, module, num_branch):
        super().__init__()
        self.num_branch = num_branch

        for idx in range(num_branch):
            name = 'branch' + str(idx)
            m = deepcopy(module)
            self.add_module(name, m)

    def forward(self, x):
        features = []
        for idx in range(self.num_branch):
            module = getattr(self, 'branch' + str(idx))
            feature = module(x)
            features.append(feature)
        return features

def split_backbone(backbone):
    layer4 = backbone.layer4
    backbone.layer4 = nn.Identity()
    return backbone, layer4

@META_ARCH_REGISTRY.register()
class CAMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self.block_size = 3
        self.register_buffer(
            "vertical_filter", torch.ones([1, 1, self.block_size, 1], requires_grad=False)
        )
        # self.register_buffer(
        #     "smooth_filter", torch.ones([1, 1, self.block_size, self.block_size], requires_grad=False)
        # )
        self.register_buffer(
            "smooth_filter", torch.ones([1, 1, 4, 8], requires_grad=False)
        )

        self.num_branch = cfg.MODEL.NUM_BRANCH

        # backbone
        backbone = build_backbone(cfg)
        self.backbone, layer4 = split_backbone(backbone)
        self.multi_branch = MultiBranch(layer4, self.num_branch)

        # head
        self.heads = build_heads(cfg)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        branch_features = self.multi_branch(features)  # List(Tensor(b c h w)) len=num_branch

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(branch_features, targets)
            branch_features = outputs.pop('branch_features')

            ret = {"outputs": outputs,
                   "targets": targets}

            # overlap activation penalty
            if self._cfg.MODEL.LOSSES.OAP.SCALE > 0:
                cls_weights = [cls.weight for cls in self.heads.classifier]
                stripe_mask, soft_mask = generate_activ_maps(branch_features,
                                                             cls_weights,
                                                             targets,
                                                             self.vertical_filter,
                                                             self.smooth_filter,
                                                             self.block_size)
                activ_maps = stripe_mask * soft_mask
                ret.update({"activ_map": activ_maps,
                            "soft_map": soft_mask})
            return ret

        else:
            outputs = self.heads(branch_features)
            return outputs

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs = outs["outputs"]
        gt_labels = outs["targets"]
        # model predictions
        pred_class_logits = [o.detach() for o in outputs['pred_class_logits']]
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits[-1], gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_cls = [cross_entropy_loss(
                logits,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE for logits in cls_outputs]
            # print("classification loss: {}".format(" ".join(["%.3f" % l.item() for l in loss_cls])))
            loss_dict['loss_cls'] = sum(loss_cls)

        if "TripletLoss" in loss_names:
            loss_tri = [triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE]
            loss_dict['loss_tri'] = sum(loss_tri)

        if "CircleLoss" in loss_names:
            loss_cir = [circle_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE]
            loss_dict['loss_cir'] = sum(loss_cir)

        # overlap activation penalty
        if self._cfg.MODEL.LOSSES.OAP.SCALE > 0:
            activ_maps = outs['activ_map']
            loss_dict['loss_oap'] = overlap_activ_penalty(
                activ_maps
            ) * self._cfg.MODEL.LOSSES.OAP.SCALE
        return loss_dict
