# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY

@REID_HEADS_REGISTRY.register()
class MultiHeadV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':
            self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool':
            self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":
            self.pool_layer = nn.Identity()
        elif pool_type == "flatten":
            self.pool_layer = Flatten()
        elif pool_type == "sumpool":
            self.pool_layer = GlobalSumPool2d()
        else:
            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        bottleneck = nn.Sequential(*bottleneck)

        # identity classification layer
        # fmt: off
        if cls_type == 'linear':
            classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax':
            classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'amSoftmax':
            classifier = AMSoftmax(cfg, feat_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        # multiple branches
        self.cfg_num_branch = cfg.MODEL.NUM_BRANCH
        self.num_branch = min(cfg.MODEL.NUM_BRANCH, 2)

        self.cls_type = cls_type
        self.bottleneck = nn.ModuleList()
        self.classifier = nn.ModuleList()
        for _ in range(self.num_branch):
            m = deepcopy(bottleneck)
            m.apply(weights_init_kaiming)
            self.bottleneck.append(m)
            m = deepcopy(classifier)
            m.apply(weights_init_classifier)
            self.classifier.append(m)

        # part heads
        if cfg.MODEL.NUM_BRANCH > 2:
            self.num_branch = 3
            num_parts = cfg.MODEL.NUM_BRANCH - 2
            feat_dim = num_parts * feat_dim

            # 1. bottleneck
            part_bottleneck = []
            part_bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
            part_bottleneck = nn.Sequential(*part_bottleneck)
            part_bottleneck.apply(weights_init_kaiming)
            self.bottleneck.append(part_bottleneck)

            # 2. identity classification layer
            classifier = nn.Linear(feat_dim, num_classes, bias=False)
            self.classifier.append(classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        features (List): feature maps from the multiple branches
        """
        assert len(features) == self.cfg_num_branch
        if self.num_branch == 3:
            part_features = torch.cat(features[2:], dim=1)  # (b num_parts*c h w)
            features = features[:2] + [part_features]

        global_feat = [self.pool_layer(f) for f in features]
        if self.num_branch == 3:
            global_feat[-1] = torch.sign(global_feat[-1]) * torch.sqrt(torch.abs(global_feat[-1]) + 1e-12)

        features = [neck(f) for neck, f in zip(self.bottleneck, global_feat)]
        bn_feat = [f[..., 0, 0] for f in features]

        # Evaluation
        # fmt: off
        if not self.training:
            bn_feat = [F.normalize(f, dim=1) for f in bn_feat]
            bn_feat = torch.cat(bn_feat, dim=1)
            return bn_feat
        # fmt: on

        # Training
        if self.cls_type == 'linear':
            cls_outputs = [cls(f) for cls, f in zip(self.classifier, bn_feat)]
            pred_class_logits = sum([F.linear(f, cls.weight) for f, cls in zip(bn_feat, self.classifier)])
        else:
            cls_outputs = [cls(f, targets) for cls, f in zip(self.classifier, bn_feat)]
            pred_class_logits = sum([cls.s * F.linear(F.normalize(f), F.normalize(cls.weight))
                                     for cls, f in zip(self.classifier, bn_feat)])

        # fmt: off
        if self.neck_feat == "before":
            feat = torch.cat(global_feat, dim=1)[..., 0, 0]
        elif self.neck_feat == "after":
            feat = torch.cat(bn_feat, dim=1)
        else:
            raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": feat,
        }
