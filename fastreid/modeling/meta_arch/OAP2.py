# encoding: utf-8
import os
# Last Change:  2022-07-09 15:28:35
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
            binary_parts = prob_to_binary(parts.detach()).float()
            vis = binary_parts.split(dim=1, split_size=1)
            for i, v in enumerate(vis):
                self.prefix_to_tensor.update({
                    "vis_part" + str(i): v  # torch.sigmoid(v.detach())
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

class SoftNorm(nn.Module):
    def __init__(self, tau, eps=1e-6):
        super().__init__()
        self.tau = tau
        self.eps = eps
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

    def forward(self, x1, x2):
        norm1 = x1.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)  # (b 1 h w)
        norm2 = x2.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)
        norm = (norm1 + norm2) / 2.
        norm = self.convert(norm)  # (b 1 h w)
        norm = torch.sigmoid(norm)
        return norm

class FgBranch(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.fg = SoftNorm(tau=tau)
        # hook something for vis
        hook_names = ["fg"]
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
        att_map = self.fg(x1, x2)  # (b 1 h w)
        return att_map

    @staticmethod
    def apply_attention(feature, att_map):
        ret = att_map * feature
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
class OAP2(nn.Module):
    r"""
    Spatial Diversity Constraint
    """

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        self.n_branch = cfg.MODEL.NUM_BRANCH - 1  # global plus foreground plus local
        # backbone
        backbone = build_backbone(cfg)
        # global
        self.stem, self.middle, self.transform = _split_backbone(backbone)
        # foreground
        if self.n_branch > 0:
            self.fg_branch = FgBranch(tau=cfg.MODEL.LOSSES.OAP.TAU)
            self.fg_transform = deepcopy(self.transform)
        # local
        if self.n_branch > 2:
            self.n_parts = cfg.MODEL.NUM_BRANCH - 2
            # num_parts = bg + parts
            self.segment_head = MultiSegHead(num_parts=self.n_parts + 1, tau=10)  # (b num_parts h w)
            self.part_transform = deepcopy(self.transform)
        # head
        self.heads = build_heads(cfg)

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

        # global (stem middle transform)
        stem_feat_map = self.stem(images)
        mid_feat_map = self.middle(stem_feat_map)
        global_feat_map = self.transform(mid_feat_map)  # (b c h w)
        branch_feat_map = [global_feat_map]

        # foreground (fg_branch fg_transform)
        if hasattr(self, "fg_branch"):
            fg_map = self.fg_branch(stem_feat_map, mid_feat_map)
            att_feat_map = self.fg_branch.apply_attention(mid_feat_map, fg_map)
            fg_feat_map = self.fg_transform(att_feat_map)
            branch_feat_map.append(fg_feat_map)
            fg_related = {"fg_prob_map": fg_map}

        # local (segment_head part_transform)
        if hasattr(self, "segment_head"):
            part_prob_maps = self.segment_head(stem_feat_map.detach(), mid_feat_map.detach())  # (b num_parts h w)
            for idx in range(1, part_prob_maps.size(1)):  # 0 indicates background
                mask = part_prob_maps[:, idx:idx + 1, :, :]
                f = (mask * mid_feat_map).contiguous()
                part_feat_map = self.part_transform(f)  # (b c h w)
                branch_feat_map.append(part_feat_map)
            part_related = {"part_prob_maps": part_prob_maps}

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(branch_feat_map, targets)
            ret = {"outputs": outputs,
                   "targets": targets}

            if hasattr(self, 'fg_branch'):
                ret.update(fg_related)

                ### ground-truth foreground mask ### 20201202
                fg_gt_mask = self.read_foreground(batched_inputs)  # (b 1 h w)
                ret.update({"fg_gt_mask": fg_gt_mask})
                ####################################

            if hasattr(self, "segment_head"):
                ret.update(part_related)

            # heatmap related
            if hasattr(self, 'fg_branch') and hasattr(self.fg_branch, 'prefix_to_tensor'):
                ret.update(self.fg_branch.prefix_to_tensor)

                # vis binary mask

                ### ground-truth foreground mask ### 20201202
                # binary_masks = {}
                # for k, v in self.fg_branch.prefix_to_tensor.items():
                #     binary_masks["vis_binary_fg"], binary_masks["vis_binary_bg"] = \
                #         gen_fg_and_bg_mask(v, fg_thresh=0.55, bg_thresh=0.3)
                # ret.update(binary_masks)
                ####################################

                ### ground-truth foreground mask ### 20201202
                ret.update({"vis_fg_gt": ret['fg_gt_mask']})
                ####################################

            if hasattr(self, "segment_head"):
                ret.update(self.segment_head.prefix_to_tensor)
            return ret

        else:
            outputs = self.heads(branch_feat_map)
            return outputs

    def losses(self, outs):
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

        # local
        if hasattr(self, "segment_head"):
            # opt 1
            # loss_dict.update(self.part_loss(outs['part_prob_maps'], outs['fg_prob_map'].detach()))

            # opt 2
            ### ground-truth foreground mask ### 20201202
            loss_dict.update(self.part_loss_with_gt(
                outs['fg_prob_map'],
                outs['part_prob_maps'],
                outs['fg_gt_mask'],
            ))
            ####################################

            # concentration loss
            # feat_dim = self._cfg.MODEL.BACKBONE.FEAT_DIM
            # part_feats = pred_features.split(dim=1, split_size=feat_dim)[-self.n_parts:]  # List[tensor(b feat_dim)]
            # part_feats = torch.stack(part_feats, dim=0)  # (num_parts b c)
            # loss_dict.update(self.concentration_loss(part_feats, gt_labels))
        return loss_dict

    def part_loss_with_gt(self, fg_prob_map, prob_maps, fg_gt):
        """
        Args:
            fg_prob_map: (b 1 h w)
            prob_maps: (b num_parts h w)
            fg_gt: (b 1 h w)
        """
        # 1. segment background and foreground
        loss_fg = F.binary_cross_entropy(fg_prob_map, fg_gt)

        # bg_prob = prob_maps[:, :1, :, :]
        # loss_fg += F.binary_cross_entropy(bg_prob, 1. - fg_gt)  # background

        # 2. overlap penalty
        prob_stripes = []
        for i in range(1, prob_maps.size(1)):
            cur = prob_maps[:, i:i + 1, :, :]
            prob_stripes.append(horizontal_stripe(cur, tau=0.1))  # (b h w)
        prob_stripes = torch.stack(prob_stripes, dim=0)  # (num_parts-1 b h w)
        loss_op = map_overlap_activ_penalty(prob_stripes)

        # 3. segment parts
        prob_stripes = prob_stripes.detach().permute(1, 0, 2, 3)  # (b num_parts-1 h w)
        seg_label = torch.cat([1 - fg_gt, prob_stripes], dim=1)  # (b num_parts h w)
        loss_seg = (-seg_label * prob_maps.log()).mean()

        # 4. collection
        loss = dict(loss_fg=loss_fg * 0.1,
                    loss_seg=loss_seg,
                    loss_op=loss_op * 5, )
        return loss

    def part_loss(self, prob_maps, fg_prob):
        """
        Args:
            prob_maps: (b num_parts h w), softMaxed probability
            fg_prob: (b 1 h w)
        """
        fg_mask, bg_mask = gen_fg_and_bg_mask(fg_prob, fg_thresh=0.55, bg_thresh=0.3)  # (b 1 h w)

        # 1. segment background
        bg_prob = prob_maps[:, :1, :, :]
        loss_bg = F.binary_cross_entropy(bg_prob[fg_mask == 1], 1. - fg_mask[fg_mask == 1]) + \
                  F.binary_cross_entropy(bg_prob[bg_mask == 1], bg_mask[bg_mask == 1])  # background

        # 2. overlap penalty
        prob_stripes = []
        for i in range(1, prob_maps.size(1)):
            cur = prob_maps[:, i:i + 1, :, :]
            prob_stripes.append(horizontal_stripe(cur, tau=0.1))  # (b h w)
        prob_stripes = torch.stack(prob_stripes, dim=0)  # (num_parts-1 b h w)
        loss_op = map_overlap_activ_penalty(prob_stripes)

        # 3. segment foreground
        prob_stripes = prob_stripes.detach().permute(1, 0, 2, 3)  # (b num_parts-1 h w)
        fg_prob_stripes = prob_stripes * fg_mask
        # bg_prob_stripes = prob_stripes * bg_mask
        part_prob_maps = prob_maps[:, 1:, :, :]  # (b num_parts-1 h w)
        loss_seg = (-1 * part_prob_maps[fg_prob_stripes == 1].log()).mean()

        # loss_seg = F.binary_cross_entropy(part_prob_maps[fg_prob_stripes == 1],
        #                                   fg_prob_stripes[fg_prob_stripes == 1]) \
        #            + F.binary_cross_entropy(part_prob_maps[bg_prob_stripes == 1],
        #                                     1 - bg_prob_stripes[bg_prob_stripes == 1])  # background

        # 4. collection
        loss = dict(loss_bg=loss_bg * 0.1,
                    loss_seg=loss_seg,
                    loss_op=loss_op * 5, )
        return loss

    def part_loss_soft(self, prob_maps, fg_prob_map):
        """
        Args:
            prob_maps: (b num_parts h w)
            fg_prob_map: (b 1 h w)
        """
        # normalize prob map
        max_val = fg_prob_map.flatten(1).max(dim=-1)[0][:, None, None, None]
        fg_prob_map = fg_prob_map / max_val

        # 1. segment background
        pred_bg_prob = prob_maps[:, :1, :, :]
        loss_bg = F.binary_cross_entropy(pred_bg_prob, 1. - fg_prob_map)

        # 2. overlap penalty
        prob_stripes = []
        for i in range(1, prob_maps.size(1)):
            cur = prob_maps[:, i:i + 1, :, :]
            prob_stripes.append(horizontal_stripe(cur, tau=0.1))  # (b h w)
        prob_stripes = torch.stack(prob_stripes, dim=0)  # (num_parts-1 b h w)
        loss_op = map_overlap_activ_penalty(prob_stripes)

        # 3. segment foreground
        prob_stripes = prob_stripes.detach().permute(1, 0, 2, 3)  # (b num_parts-1 h w)
        weights = prob_stripes * fg_prob_map  # (b num_parts-1 h w)
        part_prob_maps = prob_maps[:, 1:, :, :]  # (b num_parts-1 h w)
        loss_seg = F.binary_cross_entropy(part_prob_maps, weights)

        # 4. collection
        loss = dict(loss_bg=loss_bg * 0.1,
                    loss_seg=loss_seg,
                    loss_op=loss_op * 5, )
        return loss

    def concentration_loss(self, part_feats, targets):
        n_parts, batch_size, ch = part_feats.size()
        loss_ctr = 0
        cnt = 0
        unique_tgt = set(targets.clone().cpu().numpy().tolist())
        for tgt in unique_tgt:
            feats = part_feats[:, targets == tgt, :]  # (num_parts, *, c)
            tgts = torch.tensor(range(n_parts), device=feats.device)[:, None].expand([n_parts, feats.size(1)])
            loss_ctr += triplet_loss(feats.view(-1, ch),
                                     tgts.flatten(0),
                                     margin=0.3,
                                     norm_feat=False,
                                     hard_mining=False)
            cnt += 1
        if cnt > 1:
            loss_ctr = loss_ctr / cnt
        loss = {'loss_ctr': loss_ctr}
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

    def read_foreground(self, batched_inputs, dst_size=(24, 8)):
        r"""
        Read foreground mask.
        """
        mask_dir = '/mnt/data2/caffe/person_reid/Market-1501-v15.09.15-foreground/bounding_box_train'
        assert isinstance(batched_inputs, dict)
        img_paths = batched_inputs["img_paths"]
        img_name = [os.path.basename(img).split('.')[0] for img in img_paths]  # List['xxx']
        img_name = [img + ".png" for img in img_name]  # List['xxx.png']
        img_paths = [os.path.join(mask_dir, img) for img in img_name]
        masks = [torch.from_numpy(np.asarray(Image.open(img))) for img in img_paths]  # List[Tensor(h w)]
        masks = torch.stack(masks, dim=0).float().unsqueeze(dim=1)  # (b h w) to (b 1 h w), uint8 to float
        masks = F.interpolate(masks, size=dst_size, mode='bilinear', align_corners=True)
        masks = masks.to(self.device)  # range 0-7,
        masks = (masks > 0).float()  # binary
        return masks

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

def horizontal_stripe(prob_map, kernel_size=3, tau=1., normalize=False):
    r"""
    Args:
        prob_map (4D tensor): (b 1 h w)
    """
    batchsize, _, height, width = prob_map.size()
    # normalize
    if normalize:
        max_val = prob_map.view(batchsize, -1).max(dim=-1, keepdim=True)[0]  # (b 1)
        prob_map = prob_map / max_val[..., None, None]  # broadcasting

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
    r"""Hard Stripes with padding
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

def horizontal_stripe_v3(prob_map, kernel_size=6, tau=1.):
    r"""Hard Stripes without padding
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
                 padding=(0, 0))  # (b 1 h/r 1)

    # generate gumbel stripe
    stripe = F.gumbel_softmax(x.view(batchsize, x.size(2)), tau=tau, hard=True)  # (b h/r)
    stripe = stripe[:, None, :, None]  # (b 1 h/r 1)
    stripe = F.interpolate(stripe, size=(height, width), mode='nearest')  # (b 1 h w)
    stripe = stripe.squeeze(1)
    return stripe

def prob_to_binary(probs):
    """
    Args:
        probs (4D tensor): (b num_parts h w)

    Returns: (b num_parts h w)
    """
    _, max_idx = probs.max(dim=1)  # (b h w)
    binary_mask = F.one_hot(max_idx)  # (b h w num_parts)
    binary_mask = binary_mask.permute(0, 3, 1, 2).contiguous()
    return binary_mask

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
