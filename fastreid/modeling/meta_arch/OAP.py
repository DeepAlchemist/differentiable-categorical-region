# encoding: utf-8
"""
# Last Change:  2022-07-09 15:28:02
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import os
import cv2
import numpy as np
from PIL import Image
#from skimage import filters
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from fastreid.layers.cbam import CBAM, SpatialGate

class Backbone(nn.Module):
    def __init__(self, network, with_attn=False, with_csa=False):
        super().__init__()
        self.with_attn = with_attn

        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1  # inp_c:   64,  oup_c:  256
        self.layer2 = network.layer2  # inp_c:  256,  oup_c:  512
        self.layer3 = network.layer3  # inp_c:  512,  oup_c: 1024
        self.layer4 = network.layer4  # inp_c: 1024,  oup_c: 2048

        if self.with_attn:
            self.attn3 = DilationAttn(in_c=512, reduction=2, stride=2)

        if with_csa:
            hook_names = ["layer1.2.csa.SpatialGate",
                          "layer2.3.csa.SpatialGate",
                          "layer3.5.csa.SpatialGate",
                          "layer4.2.csa.SpatialGate"]
            self.prefix_to_tensor = {}
            for n in hook_names:
                self.prefix_to_tensor.update({"vis_" + n: None})
            for name, prefix in zip(hook_names, self.prefix_to_tensor.keys()):
                self.module_forward_hook(name, prefix)

        # Temporary
        self.norm_attn = SoftNorm(tau=0.1, num_stripe=3)
        hook_names = ["norm_attn"]
        self.prefix_to_tensor = {}
        for n in hook_names:
            self.prefix_to_tensor.update({"vis_" + n: None})
        for name, prefix in zip(hook_names, self.prefix_to_tensor.keys()):
            self.module_forward_hook(name, prefix)

    def module_forward_hook(self, model_name, prefix, hook_output=True):
        def hook_in(module, input, output):
            self.prefix_to_tensor[prefix] = input

        def hook_out(module, input, output):
            self.prefix_to_tensor[prefix] = output

        # get the selected module
        model_name = model_name.split('.')
        model = self
        for name in model_name:
            assert hasattr(model, name), \
                '{} has no attr {}'.format(model.__class__.__name__, name)
            model = getattr(model, name)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module

        # register a forward hook for the selected module
        assert isinstance(model, torch.nn.Module), \
            'register_forward_hook is an attribute of nn.Module'
        # import pdb; pdb.set_trace()
        model.register_forward_hook(hook_out if hook_output else hook_in)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x2 = self.layer2(x)

        # Attention unit
        x = self.layer3(x2)
        if self.with_attn:
            attn = self.attn3(x2)  # (b 1 h w)
            x = x * attn
        #
        x = self.layer4(x)
        attn = self.norm_attn(x)
        x = x * attn
        return x

class DilationAttn(nn.Module):
    r""" Discriminative Feature Learning with Consistent Attention Regularization.
    in_c=512, reduction=2, stride=2
    """

    def __init__(self, in_c, reduction, stride):
        super().__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c, in_c // reduction, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_c // reduction),
            nn.ReLU(),
            nn.Conv2d(in_c // reduction, in_c // reduction // 2, kernel_size=1),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_c, in_c // reduction, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_c // reduction),
            nn.ReLU(),
            nn.Conv2d(in_c // reduction, in_c // reduction // 2, kernel_size=1),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_c, in_c // reduction, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(in_c // reduction),
            nn.ReLU(),
            nn.Conv2d(in_c // reduction, in_c // reduction // 2, kernel_size=1),
        )

        self.conv_agg = nn.Sequential(
            nn.Conv2d(in_c // reduction // 2, 1, kernel_size=1, stride=stride)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_agg(self.conv3(x) + self.conv5(x) + self.conv7(x))
        x = x.sigmoid()  # (b, 1, h, w)
        return x

class GlobalBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        # spatial attention
        self.convert = nn.Sequential(
            nn.BatchNorm2d(1),
        )
        self.p1 = nn.Parameter(torch.ones(1) * 3, requires_grad=True)
        self.p2 = nn.Parameter(torch.ones(1) * 3, requires_grad=True)
        # init
        for m in self.convert.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # hook something for vis
        self.prefix_to_tensor = {"vis_global": 0}

    def forward(self, x1, x2):
        norm1 = x1.clamp(min=self.eps).pow(self.p1).sum(dim=1, keepdim=True).pow(1. / self.p1)  # (b 1 h w)
        norm2 = x2.clamp(min=self.eps).pow(self.p2).sum(dim=1, keepdim=True).pow(1. / self.p2)
        norm = (norm1 + norm2) / 2.
        norm = self.convert(norm)  # (b 1 h w)
        norm = torch.sigmoid(norm)
        self.prefix_to_tensor['vis_global'] = norm.detach()
        return norm

class MultiSegHead(nn.Module):
    r"""
    """

    def __init__(self, num_parts, tau=10):
        super().__init__()
        self.tau = tau
        self.seg_head1 = SegHead(num_parts=num_parts, in_c=1024)
        self.seg_head2 = SegHead(num_parts=num_parts, in_c=2048)
        self.prefix_to_tensor = {}

    def forward(self, x1, x2):
        parts1 = self.seg_head1(x1)
        parts2 = self.seg_head2(x2)
        parts = (parts1 + parts2) / 2.  # (b num_parts h w)
        # parts = torch.sigmoid(parts)
        parts = F.softmax(parts * self.tau, dim=1)  # (b num_parts h w)

        # record for visualization
        with torch.no_grad():
            vis = parts.split(dim=1, split_size=1)
            for i, v in enumerate(vis):
                self.prefix_to_tensor.update({
                    "vis_part" + str(i): v.detach()  # torch.sigmoid(v.detach())
                })
        return parts

class SegHead(nn.Module):
    r"""
    """

    def __init__(self, num_parts=3, in_c=1024):
        super().__init__()
        self.num_parts = num_parts
        self.eps = 1e-6

        for idx in range(num_parts):
            name = "fc" + str(idx)
            m = nn.Conv2d(in_c, 1, kernel_size=1, bias=False)
            self.add_module(name, m)

        for idx in range(num_parts):
            name = "bn" + str(idx)
            m = nn.BatchNorm2d(1)
            self.add_module(name, m)

        self.p = nn.Parameter(torch.ones(1) * 3, requires_grad=True)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.clamp(min=self.eps)
        parts = []
        for idx in range(self.num_parts):
            fc = getattr(self, 'fc' + str(idx)).weight  # (1 in_c 1 1)
            fc = torch.sigmoid(fc)
            part = (x * fc).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)
            bn = getattr(self, 'bn' + str(idx))
            part = bn(part)
            parts.append(part)  # (b 1 h w)
        parts = torch.cat(parts, dim=1)  # (b num_parts h w)
        return parts

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SegHeadV2(nn.Module):
    r"""
    """

    def __init__(self, num_parts=3, in_c=1024):
        super().__init__()
        self.num_parts = num_parts
        self.eps = 1e-6

        for idx in range(num_parts):
            name = "fc" + str(idx)
            m = SELayer(in_c)
            self.add_module(name, m)

        # for idx in range(num_parts):
        #     name = "bn" + str(idx)
        #     m = nn.BatchNorm2d(1)
        #     self.add_module(name, m)

        self.p = nn.Parameter(torch.ones(1) * 3, requires_grad=True)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.clamp(min=self.eps)
        parts = []
        for idx in range(self.num_parts):
            se_layer = getattr(self, 'fc' + str(idx))
            ch_gate = se_layer(x)  # (b in_c 1 1)
            part = (x * ch_gate).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)
            # bn = getattr(self, 'bn' + str(idx))
            # part = bn(part)
            parts.append(part)  # (b 1 h w)
        parts = torch.cat(parts, dim=1)  # (b num_parts h w)
        return parts

class SoftNorm(nn.Module):
    r"""
    """

    def __init__(self, tau, eps=1e-6):
        super().__init__()
        self.tau = tau
        self.eps = eps

        # generate mask
        self.v_size = 3
        self.register_buffer(
            "v_f", torch.ones([1, 1, self.v_size, 1], requires_grad=False)  # vertical filter
        )

        # spatial attention
        self.convert = nn.Sequential(
            # nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
        )
        self.p = nn.Parameter(torch.ones(1) * 3, requires_grad=True)

        # record
        self.logit = 0

        # init
        for m in self.convert.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def generate_mask(self, x):
        r"""
        Args:
            x (4D tensor): (b 1 h w)
        """
        b, _, h, w = x.size()
        x = torch.sum(x, -1, keepdim=True)  # (b 1 h 1)
        x = F.conv2d(input=x,
                     weight=self.v_f,
                     padding=(self.v_size // 2, 0))  # (b 1 h 1)
        if self.v_size % 2 == 0:
            x = x[:, :, :-1]

        # generate gumbel stripe mask
        if self.training:
            stripe_mask = F.gumbel_softmax(x.view(b, h), tau=self.tau, hard=True)
        # generate the stripe mask
        else:
            max_index = torch.argmax(x.view(b, h), dim=1)  # (b,)
            stripe_mask = torch.zeros([b, h], device=x.device)
            batch_index = torch.arange(0, b, dtype=torch.long)
            stripe_mask[batch_index, max_index] = 1
        #
        k_size = 15  # [7 5]
        stripe_mask = F.max_pool2d(input=stripe_mask[:, None, :, None],
                                   kernel_size=(k_size, 1),
                                   stride=(1, 1),
                                   padding=(k_size // 2, 0))  # (b 1 h 1)
        if k_size % 2 == 0:
            stripe_mask = stripe_mask[:, :, 1:]

        stripe_mask = stripe_mask.expand([b, 1, stripe_mask.size(2), w])  # (b 1 h w)

        self.logit = stripe_mask.squeeze(1)  # (b h w)

        return stripe_mask.detach()  # detach yields better performance

    def _forward(self, x):
        norm = x.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)
        # generate mask
        # mask = self.generate_mask(norm)
        # sigmoid norm
        norm = self.convert(norm)
        norm = torch.sigmoid(norm)
        # generate attention
        # norm = norm * mask
        return norm

    def forward(self, x1, x2):
        norm1 = x1.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)  # (b 1 h w)
        norm2 = x2.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)
        norm = (norm1 + norm2) / 2.
        norm = self.convert(norm)  # (b 1 h w)
        norm = torch.sigmoid(norm)
        return norm

class LocalBranch(nn.Module):
    def __init__(self, num_branch, tau):
        super().__init__()
        self.num_branch = num_branch

        for idx in range(num_branch):
            name = 'local' + str(idx)
            # m = DilationAttn(in_c=in_c, reduction=2, stride=stride)
            m = SoftNorm(tau=tau)
            self.add_module(name, m)

        # hook something for vis
        hook_names = ["local" + str(idx) for idx in range(num_branch)]
        self.prefix_to_tensor = {}
        for n in hook_names:
            self.prefix_to_tensor.update({"vis_" + n: None})
        for name, prefix in zip(hook_names, self.prefix_to_tensor.keys()):
            self.module_forward_hook(name, prefix)

    def module_forward_hook(self, model_name, prefix, hook_output=True):
        def hook_in(module, input, output):
            self.prefix_to_tensor[prefix] = input

        def hook_out(module, input, output):
            self.prefix_to_tensor[prefix] = output

        # get the selected module
        model_name = model_name.split('.')
        model = self
        for name in model_name:
            assert hasattr(model, name), \
                '{} has no attr {}'.format(model.__class__.__name__, name)
            model = getattr(model, name)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module

        # register a forward hook for the selected module
        assert isinstance(model, torch.nn.Module), \
            'register_forward_hook is an attribute of nn.Module'
        # import pdb; pdb.set_trace()
        model.register_forward_hook(hook_out if hook_output else hook_in)

    def forward(self, x1, x2):
        att_map = []
        for idx in range(self.num_branch):
            # attention map
            m = getattr(self, 'local' + str(idx))
            att = m(x1, x2)
            att_map.append(att)
        return att_map

    @staticmethod
    def apply_attention(feature, att_maps):
        ret = [att * feature for att in att_maps]
        return ret

def _split_backbone(backbone):
    stem = nn.Sequential()
    stem.add_module('conv1', nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool))
    stem.add_module('conv2', backbone.layer1)
    stem.add_module('conv3', backbone.layer2)
    stem.add_module('conv4', backbone.layer3)

    depth_layer4 = len(backbone.layer4)
    middle = nn.Sequential(*[backbone.layer4[i] for i in range(0, depth_layer4 - 1)])
    transform = backbone.layer4[-1]
    return stem, middle, transform

@META_ARCH_REGISTRY.register()
class OAP(nn.Module):
    r"""
    Spatial Diversity Constraint
    """

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        self.n_local_branch = cfg.MODEL.NUM_BRANCH - 1  # global plus local
        # backbone
        backbone = build_backbone(cfg)
        self.stem, self.middle, self.transform = _split_backbone(backbone)
        if cfg.MODEL.GLOBAL_ATTN:
            self.global_branch = GlobalBranch()
        if self.n_local_branch > 0:
            self.local_branch = LocalBranch(self.n_local_branch, tau=cfg.MODEL.LOSSES.OAP.TAU)
            self.local_transform = deepcopy(self.transform)
        # head
        self.heads = build_heads(cfg)

        ### segmentation head ###
        num_parts = 4
        self.segment_head = MultiSegHead(num_parts=num_parts, tau=10)
        self.part_transform = deepcopy(self.transform)
        cfg.defrost()
        cfg.MODEL.NUM_BRANCH = num_parts - 1
        cfg.freeze()
        self.part_heads = build_heads(cfg)
        self.load_ckpt()
        for layer in cfg.MODEL.FREEZE_LAYERS:
            m = getattr(self, layer)
            for subm in m.modules():
                subm.requires_grad = False
            m.eval()
        #########################

    def load_ckpt(self, ):
        ckpt_path = '/home/caffe/code/fastReID/logs/mk/OAP/oap-glbFg/model_final.pth'
        ckpt_state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        ckpt_state_dict = ckpt_state_dict["model"]
        model_state_dict = self.state_dict()
        for k in list(ckpt_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(ckpt_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    print(  # self.logger.warning
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(
                            k, shape_checkpoint, shape_model
                        )
                    )
                    ckpt_state_dict.pop(k)

        incompatible = self.load_state_dict(
            ckpt_state_dict, strict=False
        )
        print(incompatible)
        return

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        with torch.no_grad():
            # global
            stem_feat_map = self.stem(images)
            mid_feat_map = self.middle(stem_feat_map)
            att_map = self.global_branch(stem_feat_map, mid_feat_map) \
                if hasattr(self, "global_branch") else 1.
            global_feat_map = self.transform(mid_feat_map * att_map)  # (b c h w)
            branch_feat_map = [global_feat_map]

            # local
            if hasattr(self, "local_branch"):
                att_map = self.local_branch(stem_feat_map, mid_feat_map)
                att_feat_map = self.local_branch.apply_attention(mid_feat_map, att_map)
                local_feat_map = [self.local_transform(f) for f in att_feat_map]
                branch_feat_map += local_feat_map

        ### segmentation head ###
        part_prob_maps = self.segment_head(stem_feat_map.detach(), mid_feat_map.detach())  # (b num_parts h w)
        part_related = {"part_prob_maps": part_prob_maps,
                        "fg_prob_map": att_map[0].detach()}
        part_feat_map = []
        for idx in range(1, part_prob_maps.size(1)):
            f = (part_prob_maps[:, idx:idx + 1, :, :] * mid_feat_map).contiguous()
            part_feat = self.part_transform(f)  # (b c h w)
            part_feat_map.append(part_feat)
        #########################

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            # outputs = self.heads(branch_feat_map, targets)
            # ret = {"outputs": outputs,
            #        "targets": targets}

            ### segmentation head ###
            ret = {"targets": targets}
            part_outputs = self.part_heads(part_feat_map, targets)
            ret.update({"part_outputs": part_outputs})
            ##########################

            ret.update(part_related)
            # heatmap related
            if hasattr(self, "global_branch"):
                ret.update(self.global_branch.prefix_to_tensor)
            if self.n_local_branch > 0 and hasattr(self.local_branch, 'prefix_to_tensor'):
                ret.update(self.local_branch.prefix_to_tensor)

                ### segmentation head ###
                binary_masks = {}
                for k, v in self.local_branch.prefix_to_tensor.items():
                    binary_masks[k + "_fg"], binary_masks[k + "_bg"] = \
                        gen_fg_and_bg_mask(v, fg_thresh=0.55, bg_thresh=0.25)
                ret.update(binary_masks)
                #########################

            ### segmentation head ###
            if hasattr(self, "segment_head"):
                ret.update(self.segment_head.prefix_to_tensor)
            #########################
            return ret

        else:
            outputs = self.heads(branch_feat_map)
            part_feat = self.part_heads(part_feat_map)
            outputs = torch.cat([outputs, part_feat], dim=1)
            return outputs

    def _losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs = outs["outputs"]
        gt_labels = outs["targets"]
        # model predictions
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_cls = [cross_entropy_loss(
                logits,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE for logits in cls_outputs]
            loss_dict['loss_cls'] = sum(loss_cls) / len(loss_cls)

        if "TripletLoss" in loss_names:
            loss_dict['loss_tri'] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict['loss_cir'] = circle_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        # overlap activation penalty
        if self.n_local_branch > 1:
            loss_oap = self.oap_loss()
            loss_dict['loss_oap'] = loss_oap * self._cfg.MODEL.LOSSES.OAP.SCALE
        return loss_dict

    def losses(self, outs):
        # fmt: off
        part_outputs = outs["part_outputs"]
        gt_labels = outs["targets"]
        # model predictions
        pred_class_logits = part_outputs['pred_class_logits'].detach()
        cls_outputs = part_outputs['cls_outputs']
        pred_features = part_outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_cls = [cross_entropy_loss(
                logits,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) for logits in cls_outputs]
            loss_dict['loss_cls'] = sum(loss_cls) / len(loss_cls)

        ### segmentation ###
        loss_dict.update(self.part_loss(outs['part_prob_maps'], outs['fg_prob_map']))
        ####################
        return loss_dict

    def part_loss(self, prob_maps, weights):
        """
        Args:
            prob_maps: (b num_parts h w)
            weights: (b 1 h w)
        """
        # logits = F.softmax(logits, dim=1)  # (b num_parts h w)

        fg_mask, bg_mask = gen_fg_and_bg_mask(weights, fg_thresh=0.55, bg_thresh=0.25)
        # 1. segment background
        bg_logit = prob_maps[:, :1, :, :]
        loss_bg = F.binary_cross_entropy(bg_logit[fg_mask == 1], 1. - fg_mask[fg_mask == 1]) + \
                  F.binary_cross_entropy(bg_logit[bg_mask == 1], bg_mask[bg_mask == 1])
        # loss_bg = F.binary_cross_entropy(bg_logit, 1. - fg_mask, weight=weights)

        # 2. segment foreground
        predicts = prob_maps.max(dim=1, keepdim=True)[0]  # (b 1 h w)
        loss_fg = (-1 * predicts.log()).mean()  # * weights

        # 3. overlap penalty
        prob_stripes = []
        for i in range(1, prob_maps.size(1)):
            cur = prob_maps[:, i:i + 1, :, :]
            prob_stripes.append(horizontal_stripe_v2(cur, tau=0.1))  # (b h w)
        prob_stripes = torch.stack(prob_stripes, dim=0)  # (num_parts-1 b h w)
        loss_op = map_overlap_activ_penalty(prob_stripes)

        # 4. collection
        loss = dict(loss_bg=loss_bg * 0.1,
                    loss_fg=loss_fg,
                    loss_op=loss_op * 5, )
        return loss

    def _part_loss(self, logits, weights):
        # logits = F.softmax(logits, dim=1)  # (b num_parts h w)
        logits = logits.max(dim=1, keepdim=True)[0]  # (b 1 h w)
        loss = (-1 * logits.log() * weights).mean()
        return loss

    def oap_loss(self):
        # opt1 map
        logits = []
        for i in range(self.n_local_branch):
            m = getattr(self.local_branch, 'local' + str(i))
            logit = m.logit  # (b h w)
            logits.append(logit)
        logits = torch.stack(logits, dim=0)  # (num_branch b h w)
        loss = map_overlap_activ_penalty(logits, detach=False)  # detach=False yields better performance
        return loss

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

def map_overlap_activ_penalty(activ_maps, detach=True):
    r"""
    Args:
        activ_maps: (num_branch b h w)
    """
    num_branch, b, h, w = activ_maps.size()
    loss = 0
    cnt = 0
    for ii in range(num_branch - 1):
        ref_map = activ_maps[ii].detach() if detach else activ_maps[ii]  # (b h w)
        for jj in range(ii + 1, num_branch):
            cnt += 1
            cur_map = activ_maps[jj]
            loss += torch.sum(ref_map * cur_map)
    loss = loss / cnt if cnt > 1 else loss
    loss = loss / (b * h * w)
    return loss

def blur_and_threshold(masks):
    """
    Args:
        masks(4D tensor): (b 1 h w)
    """
    device = masks.device
    masks = masks.split(dim=0, split_size=1)
    masks = [m.squeeze().unsqueeze(-1).cpu().numpy() for m in masks]  # (h w 1)
    post_masks = []
    for m in masks:
        # blur = cv2.GaussianBlur(m, ksize=(5, 5), sigmaX=0)  # (h w)
        blur = m[:, :, 0]
        # thr = filters.threshold_otsu(blur)
        # thr = np.mean(blur)
        thr = blur.ravel()[blur.ravel().argsort()[80]]
        mask = torch.from_numpy(blur > thr).float()
        post_masks.append(mask)
    post_masks = torch.stack(post_masks, dim=0)[:, None, :, :]  # (b 1 h w)
    post_masks = post_masks.to(device)
    return post_masks

def gen_fg_and_bg_mask(prob_map, fg_thresh=0.55, bg_thresh=0.25, eps=1e-6):
    """
    Args:
        prob_map(4D tensor): (b 1 h w)
    """
    num_pixels = prob_map.size(2) * prob_map.size(3)
    num_fg = round(num_pixels * fg_thresh)
    num_bg = round(num_pixels * bg_thresh)
    topK = prob_map.view(prob_map.size(0), -1).topk(k=num_fg + num_bg, dim=-1)[0]  # (b topK)
    fg_thresh = topK[:, num_fg][:, None, None, None]  # (b,) to (b 1 1 1)
    bg_thresh = topK[:, -1][:, None, None, None]  # (b,) to (b 1 1 1)
    fg_mask = ((prob_map - fg_thresh) > eps).float()  # (b 1 h w)
    bg_mask = ((bg_thresh - prob_map) > eps).float()  # (b 1 h w)
    return fg_mask, bg_mask

def horizontal_stripe(prob_map, kernel_size=3, tau=1.):
    r"""
    Args:
        prob_map (4D tensor): (b 1 h w)
    """
    batchsize, _, height, width = prob_map.size()
    kernel = torch.ones([1, 1, kernel_size, 1], requires_grad=False, device=prob_map.device)  # filter

    prob_map = torch.sum(prob_map, -1, keepdim=True)  # (b 1 h 1)
    prob_map = F.conv2d(input=prob_map,
                        weight=kernel,
                        padding=(kernel_size // 2, 0))  # (b 1 h 1)
    if kernel_size % 2 == 0:
        prob_map = prob_map[:, :, :-1]

    # generate gumbel stripe
    stripe = F.gumbel_softmax(prob_map.view(batchsize, height), tau=tau, hard=True)  # (b h)

    #
    k_size = 5  # [7 5]
    stripe = F.max_pool2d(input=stripe[:, None, :, None],
                          kernel_size=(k_size, 1),
                          stride=(1, 1),
                          padding=(k_size // 2, 0))  # (b 1 h 1)
    if k_size % 2 == 0:
        stripe = stripe[:, :, 1:]

    stripe = stripe.squeeze(1).expand([batchsize, stripe.size(2), width])  # (b h w)
    return stripe

def horizontal_stripe_v2(prob_map, kernel_size=5, tau=1.):
    r"""Hard Stripes
        kernel=8, stride=8, padding=0, out_size=3
    Args:
        h_block (4D tensor): (b 1 h w)
    """
    batchsize, _, height, width = prob_map.size()
    kernel = torch.ones([1, 1, kernel_size, 1], requires_grad=False, device=prob_map.device)  # filter

    x = torch.sum(prob_map, dim=-1, keepdim=True)  # (b 1 h 1)
    x = F.conv2d(input=x,
                 weight=kernel,
                 stride=(kernel_size, 1),
                 padding=(kernel_size // 2, 0))  # (b 1 h/r 1)
    if kernel_size % 2 == 0:
        x = x[:, :, :-1]

    # generate gumbel stripe
    stripe = F.gumbel_softmax(x.view(batchsize, x.size(2)), tau=tau, hard=True)  # (b h/r)
    stripe = stripe[:, None, :, None]  # (b 1 h/r 1)
    stripe = F.interpolate(stripe, size=(height, width), mode='nearest')  # (b 1 h w)
    stripe = stripe.squeeze(1)
    return stripe

def write_masks(masks, im_names, save_dir="/mnt/data2/caffe/person_reid/market1501_mask/"):
    """
    Args:
        masks (4D tensor): (b 1 h w) value 0-1
        im_names (List):
        save_dir:
    """
    dst_size = (128, 64)
    masks = F.interpolate(masks, size=dst_size, mode='bilinear', align_corners=False)
    # masks = masks.permute(0, 2, 3, 1).contiguous().split(dim=0, split_size=1)  # List[tensor(h w 1)]
    masks = masks.squeeze(1).split(dim=0, split_size=1)  # List[tensor(1 h w)]
    for mask, name in zip(masks, im_names):
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
        mask = mask.squeeze(0).cpu().numpy().round().astype(np.uint8)
        mask = Image.fromarray(mask, mode='L')
        im_path = os.path.join(save_dir, name)
        mask.save(im_path)
    return
