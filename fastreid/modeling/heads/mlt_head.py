# encoding: utf-8
"""
# Last Change:  2022-08-06 17:15:27
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
class MultiHead(nn.Module):
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
        self.cls_type = cls_type
        if "part" in cfg.MODEL.BRANCH:
            self.num_branch = len(cfg.MODEL.BRANCH)-1 + cfg.MODEL.NUM_PART
        else:
            self.num_branch = len(cfg.MODEL.BRANCH)

        self.bottleneck = nn.ModuleList()
        self.classifier = nn.ModuleList()
        for _ in range(self.num_branch):
            m = deepcopy(bottleneck)
            m.apply(weights_init_kaiming)
            self.bottleneck.append(m)
            m = deepcopy(classifier)
            m.apply(weights_init_classifier)
            self.classifier.append(m)

    def forward(self, features, targets=None, part_prob_maps=None):
        """
        See :class:`ReIDHeads.forward`.
        features (List): feature maps from the multiple branches
        """
        assert len(features) == self.num_branch
        # opt1: BN+Pooling
        # global_feat = [self.pool_layer(f) for f in features]
        # bn_feat = [neck(f) for neck, f in zip(self.bottleneck, features)]
        # bn_feat = [self.pool_layer(f)[..., 0, 0] for f in bn_feat]

        # opt2: Pooling+BN
        global_feat = [self.pool_layer(f) for f in features]
        features = [neck(f) for neck, f in zip(self.bottleneck, global_feat)]
        bn_feat = [f[..., 0, 0] for f in features]

        # Evaluation
        # fmt: off
        # opt1
        # if not self.training:
        #     bn_feat = [F.normalize(f, dim=1) for f in bn_feat]
        #     bn_feat = torch.cat(bn_feat, dim=1)
        #     return {"feature": bn_feat}

        # opt2
        # if not self.training:
        #     bn_feat = [F.normalize(f, dim=1) for f in bn_feat]
        #     if part_prob_maps is not None:  # (b num_parts h w)
        #         part_probs = part_prob_maps.sum(-1).sum(-1)  # (b num_parts)
        #         part_probs[part_probs > 1] = 1.  # (b num_parts)
        #         n_feats = len(bn_feat)
        #         n_part_feats = part_probs.size(-1)
        #         step = n_feats - n_part_feats
        #         for ii in range(n_part_feats):
        #             bn_feat[step + ii] = bn_feat[step + ii] * part_probs[:, ii:ii + 1].sqrt()
        #     bn_feat = torch.cat(bn_feat, dim=1)
        #     return bn_feat

        # opt3
        # if not self.training:
        #     if part_prob_maps is not None:  # (b num_parts h w)
        #         part_probs = part_prob_maps.sum(-1).sum(-1)  # (b num_parts)
        #         n_feats = len(bn_feat)
        #         n_part_feats = part_probs.size(-1)
        #         step = n_feats - n_part_feats
        #         for ii in range(n_part_feats):
        #             bn_feat[step + ii] = bn_feat[step + ii] * part_probs[:, ii:ii + 1].sqrt()
        #     bn_feat = torch.cat(bn_feat, dim=1)
        #     bn_feat = F.normalize(bn_feat, dim=1)
        #     return bn_feat

        # opt3
        if not self.training:
            bn_feat = torch.stack(bn_feat, dim=1)  # (b n*dim_feat)

            if part_prob_maps is not None:  # (b num_parts h w)
                p_vis = part_prob_maps.sum(-1).sum(-1)  # (b num_parts)
            else:
                p_vis = torch.ones([bn_feat.size(0), bn_feat.size(1)], dtype=torch.float, device=bn_feat.device)
            return {"feature": bn_feat,
                    "visibility": p_vis}

        # fmt: on

        # Training
        # if self.cls_type == 'linear':
        #     cls_outputs = sum([cls(f) for cls, f in zip(self.classifier, bn_feat)])
        #     pred_class_logits = sum([F.linear(f, cls.weight) for f, cls in zip(bn_feat, self.classifier)])
        # else:
        #     cls_outputs = sum([cls(f, targets) for cls, f in zip(self.classifier, bn_feat)])
        #     pred_class_logits = sum([cls.s * F.linear(F.normalize(f), F.normalize(cls.weight))
        #                              for cls, f in zip(self.classifier, bn_feat)])

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

        # if self.neck_feat == "before":
        #    global_feat = [feat[..., 0, 0] for feat in global_feat]
        #    if part_prob_maps is not None:  # (b num_parts h w)
        #        part_probs = part_prob_maps.sum(-1).sum(-1)  # (b num_parts)
        #        part_probs = part_probs / part_probs.sum(-1, keepdim=True) # (b num_parts)
        #        n_feats = len(global_feat)
        #        n_part_feats = part_probs.size(-1)
        #        step = n_feats - n_part_feats
        #        for ii in range(n_part_feats):
        #            global_feat[step + ii] = global_feat[step + ii] * part_probs[:, ii:ii + 1].sqrt()
        #    feat = torch.cat(global_feat, dim=1)

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
