# encoding: utf-8
# Last Change:  2022-08-28 18:11:32


import os
import cv2
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from fastreid.layers.cbam import CBAM, SpatialGate

class MultiSegHead(nn.Module):
    r"""
    """
    def __init__(self, num_parts, tau=10):
        super(MultiSegHead, self).__init__()
        self.tau = tau
        self.seg_head = SegHead(num_parts=num_parts, in_c=1024)
        self.prefix_to_tensor = {}

    def forward(self, x):
        parts = self.seg_head(x)  # (b num_parts h w)
        parts = F.softmax(parts * self.tau, dim=1)  # (b num_parts h w)

        # record for visualization
        with torch.no_grad():
            #binary_parts = onehoting_probmap(parts.detach()).float()
            #vis = binary_parts.split(dim=1, split_size=1)
            
            vis = parts.detach().split(dim=1, split_size=1)  # keepdim is True
            for i, v in enumerate(vis):
                p_mask = v.sum(2, keepdim=True)  # (b 1 1 w)
                v = v/p_mask  # (b 1 h w) normalize along h_dim

                self.prefix_to_tensor.update({
                    "vis_part" + str(i): v  # torch.sigmoid(v.detach())
                })
        return parts


class SegHead(nn.Module):
    def __init__(self, dst_size=(24, 8), num_parts=3, in_c=1024):
        super(SegHead, self).__init__()
        self.dst_size = dst_size
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
        x = x.clamp(min=self.eps)  # (b c h w)
        parts = []
        for idx in range(self.num_parts):
            fc = getattr(self, 'fc' + str(idx)).weight  # (1 in_c 1 1)
            fc = torch.sigmoid(fc)
            part = (x * fc).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)  # (b 1 h w)

            bn = getattr(self, 'bn' + str(idx))
            part = bn(part)
            parts.append(part)  # (b 1 h w)
        parts = torch.cat(parts, dim=1)  # (b num_parts h w)
        # parts = F.interpolate(parts, size=self.dst_size, mode='bilinear', align_corners=True)
        return parts

class SoftNorm(nn.Module):
    def __init__(self, dst_size=(24, 8), eps=1e-6):
        super(SoftNorm, self).__init__()
        self.dst_size = dst_size
        self.eps = eps
        # spatial attention
        self.convert = nn.Sequential(
            nn.BatchNorm2d(1),
        )
        self.p = nn.Parameter(torch.ones(1) * 3, requires_grad=True)

        # init
        for m in self.convert.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        norm = x.clamp(min=self.eps).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)  # (b 1 h w)
        # norm = F.interpolate(norm, size=self.dst_size, mode='bilinear', align_corners=True)

        norm = self.convert(norm)  # (b 1 h w)
        norm = torch.sigmoid(norm)
        return norm

class FgBranch(nn.Module):
    def __init__(self):
        super(FgBranch, self).__init__()
        self.fg = SoftNorm(dst_size=(24, 8))
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

    def forward(self, x):
        att_map = self.fg(x)  # (b 1 h w)
        return att_map

    @staticmethod
    def apply_attention(feature, att_map):
        ret = att_map * feature
        return ret

def split_backbone(backbone):
    stem = nn.Sequential()
    stem.add_module('conv1', nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool))
    stem.add_module('conv2', backbone.layer1)  # (256 96 32)
    stem.add_module('conv3', backbone.layer2)  # (512 48 16)

    depth_layer3 = len(backbone.layer3)  # 6
    layer3_bottom = nn.Sequential(*[backbone.layer3[i] for i in range(depth_layer3 // 2)])
    layer3_top = nn.Sequential(*[backbone.layer3[i] for i in range(depth_layer3 // 2, depth_layer3)])
    stem.add_module('conv4_bottom', layer3_bottom)  # (1024 24 8)

    middle = nn.Sequential()
    middle.add_module('conv4_top', layer3_top)
    depth_layer4 = len(backbone.layer4)
    for i in range(0, depth_layer4 - 1):
        middle.add_module('conv5_{}'.format(i), backbone.layer4[i])

    transform = backbone.layer4[-1]
    return stem, middle, transform

@META_ARCH_REGISTRY.register()
class GSS(nn.Module):
    r"""
    Diversity Regularization via Gumbel-Softmax Sampling
    """
    def __init__(self, cfg):
        super(GSS, self).__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        self.n_branch = 1  # global plus foreground plus local
        self.branch_lst = cfg.MODEL.BRANCH

        # backbone
        backbone = build_backbone(cfg)

        # globe
        self.stem, self.middle, self.transform = split_backbone(backbone)

        # foreground
        if "foreground" in self.branch_lst:
            self.fg_branch = FgBranch()
            self.fg_transform = deepcopy(self.transform)

        # local
        if "part" in self.branch_lst:
            self.n_parts = cfg.MODEL.NUM_PART
            # num_parts = bg + parts
            self.part_branch = MultiSegHead(num_parts=self.n_parts + 1, tau=1.)  # (b bg+num_parts h w)
            self.part_transform = deepcopy(self.transform)

        # head
        self.heads = build_heads(cfg)

        # prob maps
        if cfg.MODEL.BACKBONE.WITH_NLKEY:
            assert "part" in self.branch_lst

    def set_attr(self, model, fg_part_prob_maps):
        for n, m in model.named_children():
            if 'nl' in n:
                m.prob_maps = fg_part_prob_maps
            else:
                self.set_attr(m, fg_part_prob_maps)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        
        ### only for generating pseudo fg label, in test stage
        conv1_fm = self.stem.conv1(images)
        conv2_fm = self.stem.conv2(conv1_fm)
        conv3_fm = self.stem.conv3(conv2_fm)
        stem_feat_map = self.stem.conv4_bottom(conv3_fm)
        conv4_fm = self.middle.conv4_top(stem_feat_map)
        ### yangwenjie
        
        # global (stem middle transform)
        stem_feat_map = self.stem(images)

        if hasattr(self, "fg_branch"):
            fg_map = self.fg_branch(stem_feat_map)  # (b 1 h w)

        part_prob_maps = None
        if hasattr(self, "part_branch"):
            part_prob_maps = self.part_branch(stem_feat_map)  # (b bg+num_parts h w)

        # set prob_maps of non-local blocks
        if self._cfg.MODEL.BACKBONE.WITH_NLKEY:
            fg_part_prob_maps = torch.cat([fg_map, part_prob_maps[:, 1:, :, :]], dim=1).detach()
            self.set_attr(self.middle, fg_part_prob_maps)

        mid_feat_map = self.middle(stem_feat_map)
        global_feat_map = self.transform(mid_feat_map)  # (b c h w)
        branch_feat_map = [global_feat_map]

        # foreground (fg_branch fg_transform)
        if hasattr(self, "fg_branch"):
            sum_fg_map = fg_map.detach().sum(2,keepdim=True).sum(3,keepdim=True)  # (b 1 1 1)
            norm_fg_map = fg_map.detach()/sum_fg_map
            att_feat_map = self.fg_branch.apply_attention(mid_feat_map, norm_fg_map)
            fg_feat_map = self.fg_transform(att_feat_map)
            branch_feat_map.append(fg_feat_map)
            fg_related = {"fg_prob_map": fg_map}

        # local (part_branch part_transform)
        if hasattr(self, "part_branch"):
            for idx in range(1, part_prob_maps.size(1)):  # 0 indicates background
                mask = part_prob_maps[:, idx:idx + 1, :, :].detach()
                p_mask = mask.sum(2, keepdim=True)  # (b 1 1 w)
                mask = mask/p_mask  # (b 1 h w) normalize along h_dim
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

            outputs = self.heads(branch_feat_map, targets, None)
            ret = {"outputs": outputs,
                   "targets": targets}

            if hasattr(self, 'fg_branch'):
                ret.update(fg_related)
                ### foreground pseudo label
                #pseudo_fg_mask, _ = gen_fg_and_bg_mask(ret['fg_prob_map'], fg_thresh=0.55, bg_thresh=0.3)  # (b 1 h w)
                #ret['fg_pseudo_mask'] = pseudo_fg_mask
                ###
                
                ### foreground pseudo label
                fg_pseudo_mask = self.read_duke_pseudo_fg(batched_inputs)  # (b 1 h w)
                ret.update({"fg_pseudo_mask": fg_pseudo_mask})
                ###

            # visualizing heatmap 
            if hasattr(self, 'fg_branch') and hasattr(self.fg_branch, 'prefix_to_tensor'):
                ret.update(self.fg_branch.prefix_to_tensor)
                ret.update({"vis_fg_pseudo": ret['fg_pseudo_mask']})
                #ret.update({"vis_fg_gt": ret['fg_gt_mask']})

            if hasattr(self, "part_branch"):
                ret.update(part_related)
                ret.update(self.part_branch.prefix_to_tensor)

                cp_part_prob_maps = ret['part_prob_maps'].detach()
                for i in range(1, cp_part_prob_maps.size(1)):
                    _cur = cp_part_prob_maps[:, i:i + 1, :, :]
                    _stripe = horizontal_stripe(_cur, self.n_parts, tau=0.1)  # (b h w)
                    _stripe = _stripe[:, None, :, :]
                    ret.update({"vis_stripe"+str(i): _stripe})

            return ret

        else:  # test stage
            binary_part_maps = None
            if hasattr(self, "part_branch"):
                binary_part_maps = onehoting_probmap(part_prob_maps)
                binary_part_maps = binary_part_maps[:, 1:, :, :].float()  # (b num_parts h w)
            outputs = self.heads(branch_feat_map, part_prob_maps=binary_part_maps)
            ################################
            # visualizing probability maps #
            ################################
            # opt1: vis gt fg mask
            # fg_gt_mask = self.read_duke_foreground(batched_inputs)  # (b 1 h w)
            # output_prob_map = torch.cat([fg_gt_mask, fg_map, part_prob_maps], dim=1)
            
            # opt2: vis pseudo fg mask
            # pseudo_fg_mask, _ = gen_fg_and_bg_mask(fg_map, fg_thresh=0.55, bg_thresh=0.3)  # (b 1 h w)
            # output_prob_map = torch.cat([pseudo_fg_mask, fg_map, part_prob_maps], dim=1)
            
            # opt3: without fg mask
            # output_prob_map = torch.cat([fg_map, part_prob_maps], dim=1)
            
            # opt4: only fg prediction
            #output_prob_map = fg_map  # (b 1 h w)
            
            # opt5: only for generating pseudo fg label, in test stage
            def normalize(fm, p=1, dst_size=(128, 64)):
                norm = fm.clamp(min=1e-6).pow(p).sum(dim=1, keepdim=True).pow(1. / p)  # (b 1 h w)
                max_val = norm.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
                min_val = norm.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
                norm = (norm - min_val) / (max_val - min_val)
                norm = F.interpolate(norm, size=dst_size, mode='bilinear', align_corners=True)
                return norm
            
            norm2 = normalize(conv2_fm, p=10)
            norm5 = normalize(mid_feat_map, p=1)
            global_norm = normalize(global_feat_map, p=1)
            
            output_prob_map = norm2 + norm5 + global_norm
            output_prob_map = torch.cat([norm2,norm5,global_norm,output_prob_map], dim=1)
            #
            outputs.update({
                'fg_map': output_prob_map,
                #'part_maps': part_prob_maps
            })
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

        # foreground
        #if hasattr(self, "fg_branch"):
        #    loss_fg = F.binary_cross_entropy(outs['fg_prob_map'], outs['fg_gt_mask'])
        #    loss_dict.update(
        #        {"loss_fg": loss_fg * 0.1}
        #    )

        # local
        if hasattr(self, "part_branch"):
            # opt 1
            #loss_dict.update(self.part_loss(outs['part_prob_maps'], outs['fg_prob_map'].detach()))
            
            ### TODO-Add@20210305
            #pseudo_fg_mask, _ = gen_fg_and_bg_mask(outs['fg_prob_map'], fg_thresh=0.55, bg_thresh=0.3)  # (b 1 h w)
            #outs['fg_gt_mask'] = pseudo_fg_mask
            ###

            #loss_dict.update(part_loss_func(
            #    outs['fg_prob_map'],
            #    outs['part_prob_maps'],
            #    outs['fg_gt_mask'],
            #))
            ####################################
            
            ### Stripe-wise 
            loss_sr, prob_stripes = self.part_loss_stripe_wise_soft(outs["part_prob_maps"][:,1:,:,:])
            loss_dict.update(loss_sr)
            ### yangwenjie 20220709

            ### Pixel-wise
            #loss_pr = self.part_loss_pixel_wise(prob_stripes, outs['part_prob_maps'], outs["fg_pseudo_mask"])
            #loss_dict.update(loss_pr)
            ### yangwenjie 20220711
        return loss_dict

    def part_loss_stripe_wise(self, part_prob_maps):
        """
        Args:
            prob_maps: (b num_parts h w) without bg
        """
        num_part = part_prob_maps.size(1)
        # stripe-wise regularization (sr)
        prob_stripes = []
        for i in range(0, part_prob_maps.size(1)):
            cur = part_prob_maps[:, i:i + 1, :, :]
            prob_stripes.append(horizontal_stripe(cur, num_part, tau=0.1))  # (b h w)
        prob_stripes = torch.stack(prob_stripes, dim=0)  # (num_parts b h w)
        loss_sr = map_overlap_activ_penalty(prob_stripes)

        # collection
        loss = dict(
            loss_sr=loss_sr * 5,
        )
        return loss, prob_stripes
    
    def part_loss_stripe_wise_soft(self, part_prob_maps, topk=24):
        """
        Args:
            prob_maps: (b num_parts h w) without bg
        """
        # soft stripe-wise regularization (sr)
        #_b, _p, _h, _w = part_prob_maps.size()
        #fmt_maps = part_prob_maps.view(_b, _p, -1)
        #top_k = fmt_maps.topk(topk, dim=2)[0][:,:,-1:]  # (b num_part 1)
        #soft_prob_stripes = (fmt_maps - top_k).view(_b, _p, _h, _w)
        #soft_prob_stripes = torch.sigmoid(soft_prob_stripes).permute(1,0,2,3).contiguous()  # (num_parts b h w)

        part_prob_maps = torch.softmax(part_prob_maps*1., dim=2)
        soft_prob_stripes = part_prob_maps.permute(1,0,2,3).contiguous()
        loss_sr = map_overlap_activ_penalty(soft_prob_stripes)

        # collection
        loss = dict(
            loss_sr=loss_sr * 5,
        )
        return loss, soft_prob_stripes

    def part_loss_pixel_wise(self, prob_stripes, part_prob_maps, fg_mask):
        """ Pixel-wise Regularization
        Args:
            prob_stripes: (num_parts b h w)
            part_prob_maps: (b bg+num_parts h w)
            fg_mask: (b 1 h w)
        """
        prob_stripes = prob_stripes.detach().permute(1, 0, 2, 3)  # (b num_parts h w)
        bg_mask = gen_bg_mask(fg_mask, bg_thresh=0.1)
        seg_label = torch.cat([bg_mask, prob_stripes*fg_mask], dim=1)  # (b num_parts+1 h w)

        # Binary cross entropy loss
        #seg_label = F.softmax(seg_label, dim=1)  # BCEv2
        #loss_pr = (-seg_label[:,1:,:,:] * part_prob_maps[:,1:,:,:].log()).mean()
        #loss_pr = (-seg_label * part_prob_maps.log()).mean()

        # KL-Div loss
        log_part_prob_maps = torch.log(part_prob_maps[:,1:,:,:])
        loss_pr = F.kl_div(log_part_prob_maps, F.softmax(seg_label*10., dim=1)[:,1:,:,:])  # temperature=1

        loss = dict(
            loss_pr=loss_pr,
        )
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

    def read_mk_foreground(self, batched_inputs, dst_size=(24, 8)):
        r"""
        Read foreground mask.
        """
        mask_dir = '/mnt/data2/caffe/person_reid/Market-1501-v15.09.15-foreground/bounding_box_train'
        assert isinstance(batched_inputs, dict)
        img_paths = batched_inputs["img_paths"]
        img_flipped = batched_inputs["flipped"]

        img_name = [os.path.basename(img).split('.')[0] for img in img_paths]  # List['xxx']
        img_name = [img + ".png" for img in img_name]  # List['xxx.png']
        img_paths = [os.path.join(mask_dir, img) for img in img_name]
        masks = [Image.open(img) for img in img_paths]
        for ii, flip in enumerate(img_flipped):
            if flip:
                masks[ii] = torchvision.transforms.functional.hflip(masks[ii])
            masks[ii] = torch.from_numpy(np.asarray(masks[ii]))  # List[Tensor(h w)]
        masks = torch.stack(masks, dim=0).float().unsqueeze(dim=1)  # (b h w) to (b 1 h w), uint8 to float
        masks = F.interpolate(masks, size=dst_size, mode='bilinear', align_corners=True)
        masks = masks.to(self.device)  # range 0-7,
        masks = (masks > 0).float()  # binary
        return masks

    def read_duke_foreground(self, batched_inputs, dst_size=(24, 8)):
        r"""
        Read foreground mask.
        """
        mask_dir = '/mnt/data2/caffe/person_reid/dukemtmc/DukeMTMC-reID-foreground/bounding_box_train'
        assert isinstance(batched_inputs, dict)
        img_paths = batched_inputs["img_paths"]
        img_flipped = batched_inputs["flipped"]
        img_erased = batched_inputs["erased"]
        img_name = [os.path.basename(img).split('.')[0] for img in img_paths]  # List['xxx']
        img_name = [img + ".png" for img in img_name]  # List['xxx.png']
        img_paths = [os.path.join(mask_dir, img) for img in img_name]
        masks = [Image.open(img) for img in img_paths]
        for ii, flip in enumerate(img_flipped):
            if flip:
                masks[ii] = torchvision.transforms.functional.hflip(masks[ii])
            masks[ii] = torch.from_numpy(np.asarray(masks[ii]))  # List[Tensor(h w)]
            erased_i, erased_j = img_erased[ii].tolist()
            if erased_i > -1:
                masks[ii][erased_i:erased_j, :] = 0
            masks[ii] = F.interpolate(masks[ii][None, None, :, :].float(),
                                      size=dst_size, mode='bilinear',
                                      align_corners=True)
        masks = torch.cat(masks, dim=0)  # (1 1 h w) to (b 1 h w)
        masks = masks.to(self.device)  # range 0-7,
        masks = (masks > 0).float()  # binary
        return masks

    def read_duke_pseudo_fg(self, batched_inputs, dst_size=(24, 8)):  # TODO-Add@20210305
        mask_dir = './logs/duke/fgMap'
        assert isinstance(batched_inputs, dict)
        img_paths = batched_inputs["img_paths"]
        img_flipped = batched_inputs["flipped"]
        img_erased = batched_inputs["erased"]
        img_name = [os.path.basename(img).split('.')[0] for img in img_paths]  # List['xxx']
        img_name = [img + ".jpg" for img in img_name]  # List['xxx.png']
        img_paths = [os.path.join(mask_dir, img) for img in img_name]
        masks = [Image.open(img) for img in img_paths]
        for ii, flip in enumerate(img_flipped):
            if flip:
                masks[ii] = torchvision.transforms.functional.hflip(masks[ii])
            masks[ii] = torch.from_numpy(np.asarray(masks[ii], dtype=np.float))  # List[Tensor(h w)]
            if masks[ii].dim() == 3:  # (h w 3) or (h w)
                masks[ii] = masks[ii][:,:,0]
            assert masks[ii].dim()==2
            erased_i, erased_j = img_erased[ii].tolist()
            if erased_i > -1:
                masks[ii][erased_i:erased_j, :] = 0
            masks[ii] = F.interpolate(masks[ii][None, None, :, :].float(),
                                      size=dst_size, mode='nearest')
        masks = torch.cat(masks, dim=0)  # (1 1 h w) to (b 1 h w)
        masks = masks.to(self.device) / 255.  # binary
        #assert len(torch.unique(masks)) == 2
        return masks

    def read_msmt_foreground(self, batched_inputs, dst_size=(24, 8)):
        r"""
        Read foreground mask.
        """
        mask_dir = '/mnt/data2/caffe/person_reid/MSMT17_V1-foreground/train'
        assert isinstance(batched_inputs, dict)
        img_paths = batched_inputs["img_paths"]
        img_flipped = batched_inputs["flipped"]
        img_erased = batched_inputs["erased"]
        img_name = [os.path.basename(img).split('.')[0] for img in img_paths]  # List['xxx']
        img_name = [img + ".png" for img in img_name]  # List['xxx.png']
        id_name = [img.split('/')[-2] for img in img_paths]  # List['xxx']
        img_paths = [os.path.join(mask_dir, img_id, img) for img, img_id in zip(img_name, id_name)]
        masks = [Image.open(img) for img in img_paths]
        for ii, flip in enumerate(img_flipped):
            if flip:
                masks[ii] = torchvision.transforms.functional.hflip(masks[ii])
            masks[ii] = torch.from_numpy(np.asarray(masks[ii]))  # List[Tensor(h w)]
            erased_i, erased_j = img_erased[ii].tolist()
            if erased_i > -1:
                masks[ii][erased_i:erased_j, :] = 0
            masks[ii] = F.interpolate(masks[ii][None, None, :, :].float(),
                                      size=dst_size, mode='bilinear',
                                      align_corners=True)
        masks = torch.cat(masks, dim=0)  # (1 1 h w) to (b 1 h w)
        masks = masks.to(self.device)  # range 0-7,
        masks = (masks > 0).float()  # binary
        return masks

    def read_cuhk_foreground(self, batched_inputs, dst_size=(24, 8)):
        r"""
        Read foreground mask.
        """
        mask_dir = '/mnt/data2/caffe/person_reid/cuhk03_np-foreground/bounding_box_train'
        assert isinstance(batched_inputs, dict)
        img_paths = batched_inputs["img_paths"]
        img_flipped = batched_inputs["flipped"]

        img_name = [os.path.basename(img).split('.')[0] for img in img_paths]  # List['xxx']
        img_name = [img + ".png" for img in img_name]  # List['xxx.png']
        img_paths = [os.path.join(mask_dir, img) for img in img_name]
        masks = [Image.open(img) for img in img_paths]
        for ii, flip in enumerate(img_flipped):
            if flip:
                masks[ii] = torchvision.transforms.functional.hflip(masks[ii])
            masks[ii] = torch.from_numpy(np.asarray(masks[ii]))  # List[Tensor(h w)]
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

def map_overlap_activ_penalty_occlude(activ_stripes, activ_maps, detach=True):
    r"""
    Args:
        activ_stripes: (num_branch b h w)
        activ_maps: (num_branch b h w)
    """
    num_branch, b, h, w = activ_stripes.size()
    loss = 0
    cnt = 0
    for ii in range(num_branch - 1):
        ref_map = activ_stripes[ii].detach() if detach else activ_stripes[ii]  # (b h w)
        ref_prob = activ_maps[ii].detach()  # (b h w)
        for jj in range(ii + 1, num_branch):
            cnt += 1
            cur_map = activ_stripes[jj]
            loss += torch.sum(ref_map * cur_map * ref_prob)
    loss = loss / cnt if cnt > 1 else loss
    loss = loss / (b * h * w)
    return loss

def gen_fg_and_bg_mask(prob_map, fg_thresh=0.55, bg_thresh=0.25, eps=1e-6):
    """
    Args:
        prob_map(4D tensor): (b 1 h w)
    """
    prob_map = prob_map.detach()
    num_pixels = prob_map.size(2) * prob_map.size(3)
    num_fg = round(num_pixels * fg_thresh)
    num_bg = round(num_pixels * bg_thresh)
    topK = prob_map.view(prob_map.size(0), -1).topk(k=num_fg + num_bg, dim=-1)[0]  # (b topK)
    fg_thresh = topK[:, num_fg][:, None, None, None]  # (b,) to (b 1 1 1)
    bg_thresh = topK[:, -1][:, None, None, None]  # (b,) to (b 1 1 1)
    fg_mask = ((prob_map - fg_thresh) > eps).float()  # (b 1 h w)
    bg_mask = ((bg_thresh - prob_map) > eps).float()  # (b 1 h w)
    return fg_mask, bg_mask

def gen_bg_mask(prob_map, bg_thresh=0.15, eps=1e-12):
    """
    Args:
        prob_map(4D tensor): (b 1 h w)
    """
    prob_map = prob_map.detach()
    num_pixels = prob_map.size(2) * prob_map.size(3)
    num_thresh = round(num_pixels * (1-bg_thresh))
    
    topK = prob_map.view(prob_map.size(0), -1).topk(k=num_thresh, dim=-1)[0]  # (b topK)
    bg_thresh = topK[:, -1][:, None, None, None]  # (b,) to (b 1 1 1)
    
    bg_mask = ((bg_thresh - prob_map) > eps).float()  # (b 1 h w)
    bg_mask[:,:,:2,:] = 1
    bg_mask[:,:,-2:,:] = 1
    return bg_mask

def gen_fg_prob(prob_map, bg_thresh=0.3, eps=1e-12):
    """
    Args:
        prob_map(4D tensor): (b 1 h w)
    """
    prob_map = prob_map.detach()
    num_pixels = prob_map.size(2) * prob_map.size(3)
    num_thresh = round(num_pixels * (1-bg_thresh))
    
    topK = prob_map.view(prob_map.size(0), -1).topk(k=num_thresh, dim=-1)[0]  # (b topK)
    bg_thresh = topK[:, -1][:, None, None, None]  # (b,) to (b 1 1 1)
    
    fg_mask = ((prob_map - bg_thresh) > eps).float()  # (b 1 h w)
    prob_map = fg_mask * prob_map
    return prob_map

def horizontal_stripe(prob_map, num_part, kernel_size=3, tau=1., normalize=False):
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
    # NOTE: prob_map should be unnormalized log probabilities
    prob_map = prob_map.view(batchsize, height)
    anchor_map = torch.ones_like(prob_map) * 1e20
    anchor_map[:, 1:-1] = 0.
    prob_map = prob_map - anchor_map
    #stripe = F.gumbel_softmax(F.log_softmax(prob_map), tau=tau, hard=True)  # (b h)
    stripe = gumbel_softmax(prob_map)
    
    # Expand to stripe
    k_size = 5  # [7 5]
    #k_size = int((height-2) // num_part)
    stripe = F.max_pool2d(input=stripe[:, None, :, None],
                          kernel_size=(k_size, 1),
                          stride=(1, 1),
                          padding=(k_size // 2, 0))  # (b 1 h 1)
    if k_size % 2 == 0:
        stripe = stripe[:, :, 1:]

    stripe = stripe.squeeze(1).expand([batchsize, stripe.size(2), width])  # (b h w)
    return stripe

def onehoting_probmap(probs):
    """
    Args:
        probs (4D tensor): (b num_parts h w)

    Returns: (b num_parts h w)
    """
    _, max_idx = probs.max(dim=1)  # (b h w)
    binary_mask = F.one_hot(max_idx)  # (b h w num_parts)
    binary_mask = binary_mask.permute(0, 3, 1, 2).contiguous()
    return binary_mask

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, dim=-1):
    y = F.log_softmax(logits, dim=dim) + sample_gumbel(logits.size(), logits.device)
    return F.softmax(y / temperature, dim=dim)

def gumbel_softmax(logits, temperature=0.8, dim=-1):
    """
    input: [*, n_class] network output
    return: [*, n_class] an one-hot vector
    """
    y_soft = gumbel_softmax_sample(logits, temperature)
    # Straight through
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
        
    return ret

class SegHeadV2(nn.Module):
    r"""
    """

    def __init__(self, dst_size=(24, 8), num_parts=3, in_c=1024):
        super(SegHeadV2, self).__init__()
        self.dst_size = dst_size
        self.num_parts = num_parts
        self.eps = 1e-6
        
        out_c = 512
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)
        for idx in range(num_parts):
            name = "fc" + str(idx)
            m = nn.Conv2d(out_c, 1, kernel_size=1, bias=False)
            self.add_module(name, m)

        for idx in range(num_parts):
            name = "bn" + str(idx)
            m = nn.BatchNorm2d(1)
            self.add_module(name, m)

        self.p = nn.Parameter(torch.ones(1) * 5, requires_grad=False)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.clamp(min=self.eps)  # (b c h w)
        parts = []
        for idx in range(self.num_parts):
            fc = getattr(self, 'fc' + str(idx)).weight  # (1 in_c 1 1)
            fc = torch.sigmoid(fc)
            part = (x * fc).pow(self.p).sum(dim=1, keepdim=True).pow(1. / self.p)  # (b 1 h w)

            bn = getattr(self, 'bn' + str(idx))
            part = bn(part)
            parts.append(part)  # (b 1 h w)
        parts = torch.cat(parts, dim=1)  # (b num_parts h w)
        # parts = F.interpolate(parts, size=self.dst_size, mode='bilinear', align_corners=True)
        return parts

class SegHeadFC(nn.Module):

    def __init__(self, dst_size=(24, 8), num_parts=3, in_c=1024, dropout=0.5):
        super(SegHeadFC, self).__init__()
        self.dst_size = dst_size
        self.num_parts = num_parts
        self.eps = 1e-12

        self.dropout = nn.Dropout(dropout)
        reduce_dim = 512
        self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_c, reduce_dim, kernel_size=1, bias=True),
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU(inplace=True)
                )
        self.part_conv = nn.Sequential(
                nn.Conv2d(reduce_dim, num_parts, kernel_size=1, bias=True),
                #nn.BatchNorm2d(num_parts),
                #nn.ReLU(inplace=True)
                )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        featmap_w = x.size(3)
        x = self.reduce_conv(x)
        x = self.dropout(x)  # (b c h w)
        part_prob_map = self.part_conv(x)  # (b num_parts h w)
        return part_prob_map

def _split_backbone(backbone):
    stem = nn.Sequential()
    stem.add_module('conv1', nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool))
    stem.add_module('conv2', backbone.layer1)  # (256 96 32)
    stem.add_module('conv3', backbone.layer2)  # (512 48 16)

    depth_layer4 = len(backbone.layer4)
    middle = nn.Sequential()
    middle.add_module('conv4', backbone.layer3)
    for i in range(0, depth_layer4 - 1):
        middle.add_module('conv5_{}'.format(i), backbone.layer4[i])

    transform = backbone.layer4[-1]
    return stem, middle, transform

