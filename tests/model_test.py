import unittest

import torch

import sys

sys.path.append('.')
from fastreid.config import cfg
from fastreid.modeling.backbones import build_resnet_backbone, build_simple_resnet_backbone
from torch import nn

class MyTestCase(unittest.TestCase):
    def test_se_resnet101(self):
        cfg.MODEL.BACKBONE.NAME = 'resnet50'
        cfg.MODEL.BACKBONE.DEPTH = 50
        cfg.MODEL.BACKBONE.WITH_IBN = True
        cfg.MODEL.BACKBONE.WITH_SE = True
        cfg.MODEL.BACKBONE.PRETRAIN_PATH = '/export/home/lxy/.cache/torch/checkpoints/se_resnet101_ibn_a.pth.tar'

        net1 = build_resnet_backbone(cfg)
        net1.cuda()
        net2 = nn.DataParallel(se_resnet101_ibn_a())
        res = net2.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAIN_PATH)['state_dict'], strict=False)
        net2.cuda()
        x = torch.randn(10, 3, 256, 128).cuda()
        y1 = net1(x)
        y2 = net2(x)
        assert y1.sum() == y2.sum(), 'train mode problem'
        net1.eval()
        net2.eval()
        y1 = net1(x)
        y2 = net2(x)
        assert y1.sum() == y2.sum(), 'eval mode problem'

def set_attr(model):
    for n, m in model.named_children():
        if 'nl' in n:
            m.prob_maps = torch.rand(1, 4, 24, 8)
            print(m.prob_maps.shape)

def test_model():
    # fmt: off
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.BACKBONE.LAST_STRIDE = 1
    cfg.MODEL.BACKBONE.NORM = "BN"
    cfg.MODEL.BACKBONE.WITH_IBN = False
    cfg.MODEL.BACKBONE.WITH_SE = False
    cfg.MODEL.BACKBONE.WITH_NL = True
    cfg.MODEL.BACKBONE.DEPTH = "50x"
    # fmt: on
    model = build_simple_resnet_backbone(cfg)
    model.apply(set_attr)
    return

if __name__ == '__main__':
    # unittest.main()
    test_model()
