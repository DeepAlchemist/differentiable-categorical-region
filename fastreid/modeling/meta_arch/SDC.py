# encoding: utf-8
import math
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from fastreid.layers.se_layer import ChannelGate

from .build import META_ARCH_REGISTRY
from tests.gumbel_test import gumbel_softmax


def l2_normalize(x, detach=False):
    r'''
    Args:
        x (4D Tensor): (b c h w)
    '''
    if detach:
        with torch.no_grad():
            norm = x.norm(p=2, dim=1, keepdim=True)  # (b 1 h w)
        x = x / norm.detach()
    else:
        norm = x.norm(p=2, dim=1, keepdim=True)  # (b 1 h w)
        x = x / norm
    return x


class SelfNormAttn(nn.Module):
    r"""
    """

    def __init__(self, tau, num_stripe):
        super().__init__()
        self.tau = tau
        self.num_stripe = num_stripe

        if num_stripe > 1:
            self.register_buffer(
                "filter", torch.ones([1, 1, 8, 8], requires_grad=False)
            )
            self.v_stride = 8
            self.padding = 0

            self.stripe_logit = None

    def vertical_max(self, x):
        r"""
        x (4D Tensor): (b 1 h w)
        """
        b = x.size(0)
        vert_activ = F.conv2d(input=x,
                              weight=self.filter,
                              stride=(self.v_stride, 1),
                              padding=(self.padding, 0))  # (b 1 h 1)
        self.stripe_logit = vert_activ[:, 0, :, 0]  # (b h)

        max_index = torch.argmax(vert_activ.view(b, -1), dim=1)  # (b,)

        # generate the stripe mask
        stripe_mask = torch.zeros([b, vert_activ.size(2)], device=vert_activ.device)
        batch_index = torch.arange(0, b, dtype=torch.long)
        stripe_mask[batch_index, max_index] = 1
        stripe_mask = stripe_mask.view(b, x.size(1), vert_activ.size(2), 1)  # (b 1 h 1)
        stripe_mask = F.interpolate(stripe_mask, size=x.size()[-2:],
                                    mode='nearest')  # (b 1 h w)
        return stripe_mask

    def forward(self, x):
        feat_norm = x.norm(p=2, dim=1, keepdim=True)  # (b 1 h w)

        soft_norm = torch.softmax(feat_norm.view(x.size(0), -1) * self.tau, dim=-1)  # (b -1)
        soft_norm = soft_norm.view(x.size(0), x.size(2), x.size(3))[:, None, :, :]  # (b h w) to (b 1 h w)

        if self.num_stripe > 1:
            stripe_mask = self.vertical_max(feat_norm)  # (b 1 h w)
            soft_norm = soft_norm * stripe_mask  # (b 1 h w)

        return soft_norm


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
        self.batch_norm = nn.BatchNorm2d(1)
        nn.init.constant_(self.batch_norm.weight, 1)
        nn.init.constant_(self.batch_norm.bias, 0)
        # self.ch_gate = ChannelGate(2048)
        self.p = nn.Parameter(torch.ones(1) * 3, requires_grad=True)

        # record
        self.logit = 0

    def generate_mask_v0(self, x):
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
        k_size = 7 # [7 5]
        stripe_mask = F.max_pool2d(input=stripe_mask[:, None, :, None],
                                   kernel_size=(k_size, 1),
                                   stride=(1, 1),
                                   padding=(k_size // 2, 0))  # (b 1 h 1)
        if k_size % 2 == 0:
            stripe_mask = stripe_mask[:, :, 1:]

        stripe_mask = stripe_mask.expand([b, 1, h, w])  # (b 1 h w)

        self.logit = stripe_mask.squeeze(1)  # (b h w)

        return stripe_mask.detach()  # detach yields better performance

    def generate_mask_v1(self, x):
        r"""Hard Stripes
            kernel=8, stride=8, padding=0, out_size=3
        Args:
            h_block (4D tensor): (b 1 h w)
        """
        b, _, h, w = x.size()
        x = torch.sum(x, dim=-1, keepdim=True)  # (b 1 h 1)
        h_block = F.conv2d(input=x,
                           weight=self.v_f,
                           stride=(self.v_size, 1),
                           padding=0)  # (b 1 h/r 1)
        self.logit = h_block[:, 0, :, 0]  # (b h/r)

        max_index = torch.argmax(h_block.view(b, -1), dim=1)  # (b,)

        # generate the stripe mask
        mask = torch.zeros([b, h_block.size(2)], device=h_block.device)
        batch_index = torch.arange(0, b, dtype=torch.long)
        mask[batch_index, max_index] = 1
        mask = mask[:, None, :, None]  # (b 1 h/r 1)
        mask = F.interpolate(mask, size=(h, w), mode='nearest')  # (b 1 h w)
        return mask

    def generate_mask_v2(self, x):
        r"""Adaptive Stripes
        Args:
            h_block (4D tensor): (b 1 h w)
        """
        b, _, h, w = x.size()
        x = torch.sum(x, -1, keepdim=True)  # (b 1 h 1)
        x = F.conv2d(input=x,
                     weight=self.v_f,
                     stride=(self.v_size, 1),
                     padding=0)  # (b 1 h 1)
        if self.v_size % 2 == 0:
            x = x[:, :, :-1]

        # generate gumbel stripe mask
        if self.training:
            stripe_mask = F.gumbel_softmax(x.view(b, -1), tau=self.tau, hard=True)
        # generate the stripe mask
        else:
            max_index = torch.argmax(x.view(b, -1), dim=1)  # (b,)
            stripe_mask = torch.zeros([b, x.size(2)], device=x.device)
            batch_index = torch.arange(0, b, dtype=torch.long)
            stripe_mask[batch_index, max_index] = 1

        stripe_mask = F.interpolate(stripe_mask[:, None, :, None], size=(h, 1), mode='nearest')  # (b 1 h 1)
        #
        k_size = 5
        stripe_mask = F.max_pool2d(input=stripe_mask,
                                   kernel_size=(k_size, 1),
                                   stride=(1, 1),
                                   padding=(k_size // 2, 0))  # (b 1 h 1)
        if k_size % 2 == 0:
            stripe_mask = stripe_mask[:, :, 1:]

        stripe_mask = stripe_mask.expand([b, 1, h, w])  # (b 1 h w)

        self.logit = stripe_mask.squeeze(1)  # (b h w)

        return stripe_mask.detach()  # detach yields better performance

    def forward(self, x):
        # opt0
        # norm = x.norm(p=2, dim=1, keepdim=True)

        # opt1
        # ch_gate = self.ch_gate(x)  # (b c 1 1)
        # x = x.clamp(min=self.eps).pow(self.p)
        # x = x * ch_gate  # broadcasting
        # norm = x.sum(dim=1, keepdim=True).pow(1. / self.p)  # (b 1 h w)

        # opt2
        # ch_gate = self.ch_gate(x)  # (b c 1 1)
        # x = x * ch_gate
        # norm = x.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)

        # opt3
        norm = x.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)

        # opt4
        # soft_norm = torch.softmax(norm.view(x.size(0), -1) * self.tau, dim=-1)  # (b -1)
        # soft_norm = soft_norm.view(x.size(0), x.size(2), x.size(3))[:, None, :, :]  # (b h w) to (b 1 h w)

        # generate mask
        mask = self.generate_mask_v0(norm)
        # threshold norm
        norm = self.batch_norm(norm)
        norm = torch.sigmoid(norm)
        # generate attention
        norm = norm * mask
        return norm


class LocalBranch(nn.Module):
    def __init__(self, module, num_branch, tau):
        super().__init__()
        self.num_branch = num_branch

        for idx in range(num_branch):
            name = 'branch' + str(idx)
            m = deepcopy(module)
            self.add_module(name, m)

        for idx in range(num_branch):
            name = 'local' + str(idx)
            # m = SelfNormAttn(tau=tau, num_stripe=2)
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

    def forward(self, x):
        features = []
        for idx in range(self.num_branch):
            # 1.feature
            m = getattr(self, 'branch' + str(idx))
            feature = m(x)

            # 2.attention map
            m = getattr(self, 'local' + str(idx))
            attn = m(feature)

            # 3. l2 normalize and attention
            feature = l2_normalize(feature)
            feature = feature * attn # no detach raise nan error when use softmax attn

            features.append(feature)
        return features


class GlobalBranch(nn.Module):
    def __init__(self, module, pow=3):
        super().__init__()
        self.eps = 1e-6
        self.model = module
        self.prefix_to_tensor = {}

        # spatial attention
        self.batch_norm = nn.BatchNorm2d(1)
        nn.init.constant_(self.batch_norm.weight, 1)
        nn.init.constant_(self.batch_norm.bias, 0)
        self.p = pow # nn.Parameter(torch.ones(1) * pow, requires_grad=True)

    def forward(self, x):
        x = self.model(x)

        norm = x.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)
        norm = self.batch_norm(norm)
        norm = torch.sigmoid(norm)
        self.prefix_to_tensor.update({'vis_global': norm.detach()})

        # 1. divide l2 norm
        # x = l2_normalize(x)
        x = x * norm

        # opt2: averaging
        # num_pixels = x.size(2) * x.size(3)
        # x = x.div(num_pixels)
        return x


def split_backbone(backbone):
    # opt1
    # last_conv = backbone.layer4
    # backbone.layer4 = nn.Identity()
    # opt2
    last_conv = backbone.layer4[2]
    backbone.layer4[2] = nn.Identity()
    return backbone, last_conv


@META_ARCH_REGISTRY.register()
class SDC(nn.Module):
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
        self.backbone, last_conv = split_backbone(backbone)
        self.global_branch = GlobalBranch(last_conv)
        if self.n_local_branch > 0:
            self.local_branch = LocalBranch(last_conv, self.n_local_branch, cfg.MODEL.LOSSES.OAP.TAU)

        # head
        self.heads = build_heads(cfg)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        branch_features = []
        # global
        g_features = self.global_branch.forward(features)
        branch_features.append(g_features)

        # local
        if hasattr(self, "local_branch"):
            local_features = self.local_branch.forward(features)  # List(Tensor(b c h w)) len=num_branch
            branch_features += local_features

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(branch_features, targets)

            ret = {"outputs": outputs,
                   "targets": targets}

            # heatmap related
            if hasattr(self.global_branch, 'prefix_to_tensor'):
                ret.update(self.global_branch.prefix_to_tensor)
            if hasattr(self, 'local_branch') and hasattr(self.local_branch, 'prefix_to_tensor'):
                ret.update(self.local_branch.prefix_to_tensor)
            return ret

        else:
            outputs = self.heads(branch_features)
            return outputs

    def losses(self, outs):
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

    def oap_loss(self):
        # opt1 map
        logits = []
        for i in range(self.n_local_branch):
            m = getattr(self.local_branch, 'local' + str(i))
            logit = m.logit  # (b h w)
            logits.append(logit)
        logits = torch.stack(logits, dim=0)  # (num_branch b h w)
        loss = map_overlap_activ_penalty(logits, detach=False)  # detach=False yields better performance

        # opt2 stripe
        # logits = []
        # for i in range(self.n_local_branch):
        #     m = getattr(self.local_branch, 'local' + str(i))
        #     logit = m.logit  # (b num_stripe)
        #     # logit = gumbel_softmax(logit, temperature=5)
        #     logit = F.gumbel_softmax(logit, tau=0.1, hard=True)
        #     logits.append(logit)
        # loss = overlap_activ_penalty(logits)

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


def overlap_activ_penalty(logits):
    num_branch = len(logits)
    loss = 0
    cnt = 0
    for ii in range(num_branch - 1):
        ref_logit = logits[ii]  # (b num_stripe)
        for jj in range(ii + 1, num_branch):
            cnt += 1
            cur_logit = logits[jj]
            loss += torch.mean(ref_logit * cur_logit)

    loss = loss / cnt if cnt > 1 else loss
    return loss


def map_overlap_activ_penalty(activ_maps, detach=True):
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
