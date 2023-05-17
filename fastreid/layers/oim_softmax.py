import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class OIMArcSoftmax(nn.Module):
    def __init__(self,
                 cfg,
                 in_feat,
                 num_classes):
        super().__init__()
        self.num_features = in_feat
        self.num_classes = num_classes
        self.momentum = 0.5
        self.s = cfg.MODEL.HEADS.SCALE
        self.m = cfg.MODEL.HEADS.MARGIN

        self.register_buffer(
            'weight', torch.zeros(num_classes, in_feat, requires_grad=False)
        )

        if self.m > 0:
            self.cos_m = math.cos(self.m)
            self.sin_m = math.sin(self.m)
            self.threshold = math.cos(math.pi - self.m)
            self.mm = math.sin(math.pi - self.m) * self.m
            self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        assert all(targets >= 0)

        features = F.normalize(features, dim=1)

        cos_theta = features.mm(self.weight.clone().t())

        if self.m > 0:
            target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            mask = cos_theta > cos_theta_m
            final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            cos_theta[mask] = hard_example * (self.t + hard_example)
            cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)

        pred_class_logits = cos_theta * self.s

        # Update
        with torch.no_grad():
            # opt1
            # self.weight[targets] = self.momentum * self.weight[targets] + \
            #                         (1. - self.momentum) * features.detach().clone()
            # self.weight[targets] = F.normalize(self.weight[targets], p=2, dim=1)

            # opt2
            unique_tgts = set(targets.clone().cpu().numpy().tolist())
            for tgt in unique_tgts:
                new_w = features.detach().clone()[targets == tgt].mean(dim=0)
                self.weight[tgt] = self.momentum * self.weight[tgt] + \
                                   (1. - self.momentum) * new_w
                self.weight[tgt] = F.normalize(self.weight[tgt], p=2, dim=0)

        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.num_features, self.num_classes, self.s, self.m
        )

class OIMCircleSoftmax(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self.s = cfg.MODEL.HEADS.SCALE
        self.m = cfg.MODEL.HEADS.MARGIN
        self.momentum = 0.5

        self.register_buffer(
            'weight', torch.zeros(num_classes, in_feat, requires_grad=False)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets):
        features = F.normalize(features)
        sim_mat = F.linear(features, F.normalize(self.weight))
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)

        y = F.one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = y * s_p + (1.0 - y) * s_n

        # Update
        with torch.no_grad():
            # opt1
            # idx = 0
            # for tgt in targets:
            #     self.weight[tgt] = self.momentum * self.weight[tgt] + \
            #                            (1. - self.momentum) * features[idx].detach().clone()
            #     self.weight[tgt] = F.normalize(self.weight[tgt], p=2, dim=1)
            #     idx += 1

            # opt2
            unique_tgts = set(targets.clone().cpu().numpy().tolist())
            for tgt in unique_tgts:
                new_w = features.detach().clone()[targets == tgt].mean(dim=0)
                self.weight[tgt] = self.momentum * self.weight[tgt] + \
                                   (1. - self.momentum) * new_w
                self.weight[tgt] = F.normalize(self.weight[tgt], p=2, dim=0)

        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self.s, self.m
        )
