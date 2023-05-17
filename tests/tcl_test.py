from __future__ import absolute_import
from __future__ import division

import math
import copy
import numpy as np
from copy import deepcopy

import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models import resnet50


def inflate_conv(conv2d,
                 time_dim=3,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = nn.Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d

def inflate_linear(linear2d, time_dim):
    """
    Args:
        time_dim: final time dimension of the features
    """
    linear3d = nn.Linear(linear2d.in_features * time_dim,
                         linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = nn.Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d

def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d

def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
    padding = (time_padding, pool2d.padding, pool2d.padding)
    if time_stride is None:
        time_stride = time_dim
    stride = (time_stride, pool2d.stride, pool2d.stride)
    if isinstance(pool2d, nn.MaxPool2d):
        dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
        pool3d = nn.MaxPool3d(
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
            ceil_mode=pool2d.ceil_mode)
    elif isinstance(pool2d, nn.AvgPool2d):
        pool3d = nn.AvgPool3d(kernel_dim, stride=stride)
    else:
        raise ValueError(
            '{} is not among known pooling classes'.format(type(pool2d)))
    return pool3d

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d, inflate_time=False):
        super(Bottleneck3d, self).__init__()

        if inflate_time == True:
            self.conv1 = inflate_conv(bottleneck2d.conv1, time_dim=3, time_padding=1, center=True)
        else:
            self.conv1 = inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate_conv(downsample2d[0], time_dim=1,
                                 time_stride=time_stride),
            inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TCLNet(nn.Module):

    def __init__(self, num_classes, use_gpu, loss={'xent'}):
        super(TCLNet, self).__init__()
        self.loss = loss
        self.use_gpu = use_gpu
        resnet2d = resnet50(pretrained=True)

        self.conv1 = inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate_pool(resnet2d.maxpool, time_dim=1)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, enhance_idx=[3], channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3)
        layer4 = nn.Sequential(resnet2d.layer4[0], resnet2d.layer4[1])

        branch = nn.ModuleList([deepcopy(resnet2d.layer4[-1]) for _ in range(2)])

        self.TSE_Module = TSE(layer4=layer4, branch=branch, use_gpu=use_gpu)

        bn = []
        for _ in range(2):
            add_block = nn.BatchNorm1d(2048)
            add_block.apply(weights_init_kaiming)
            bn.append(add_block)
        self.bn = nn.ModuleList(bn)

        classifier = []
        for _ in range(2):
            add_block = nn.Linear(2048, num_classes)
            add_block.apply(weights_init_classifier)
            classifier.append(add_block)
        self.classifier = nn.ModuleList(classifier)

    def _inflate_reslayer(self, reslayer2d, enhance_idx=[], channels=0):
        reslayers3d = []
        for i, layer2d in enumerate(reslayer2d):
            layer3d = Bottleneck3d(layer2d)
            reslayers3d.append(layer3d)

            if i in enhance_idx:
                TSB_Module = TSB(in_channels=channels, use_gpu=self.use_gpu)
                reslayers3d.append(TSB_Module)

        return nn.Sequential(*reslayers3d)

    def pool(self, x):
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)  # [b, c]
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        b, c, t, h, w = x.size()

        x1_list = self.TSE_Module(x[:, :, :2])
        x2_list = self.TSE_Module(x[:, :, 2:])

        x1 = self.pool(x1_list[0])
        x2 = self.pool(x1_list[1])
        x3 = self.pool(x2_list[0])
        x4 = self.pool(x2_list[1])

        x1 = torch.stack((x1, x3), 1)  # [b, 2, c]
        x2 = torch.stack((x2, x4), 1)

        x1 = x1.mean(1)  # [b, c]
        x2 = x2.mean(1)

        if not self.training:
            x = torch.cat((x1, x2), 1)  # [b, c * 2]
            return x

        f1 = self.bn[0](x1)
        f2 = self.bn[1](x2)

        y1 = self.classifier[0](f1)
        y2 = self.classifier[1](f2)

        if self.loss == {'xent'}:
            return [y1, y2]
        elif self.loss == {'xent', 'htri'}:
            return [y1, y2], [f1, f2]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class TSE(nn.Module):
    def __init__(self, layer4, branch, use_gpu=False):
        super(TSE, self).__init__()
        self.layer4 = layer4
        self.branch = branch
        self.block_size = 3
        self.in_channels = 1024
        self.use_gpu = use_gpu

        self.conv_reduce = nn.Conv2d(2048, self.in_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_erase = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )

        # init
        for m in [self.conv_reduce]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.conv_erase:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0)
                m.bias.data.zero_()

    def correlation(self, x1, x2):
        """calculate the correlation map of x1 to x2
        """
        b, c, t, h, w = x2.size()

        x1 = F.avg_pool2d(x1, x1.size()[2:])
        x1 = self.conv_reduce(x1)
        x1 = x1.view(b, -1)

        x2 = x2.view(b, c, -1)
        import pdb; pdb.set_trace()

        f = torch.matmul(x1.view(b, 1, c), x2)
        f = f / np.sqrt(c)

        f = f.view(b, t, h, w)
        return f

    def block_binarization(self, f):
        """
        generate the binary masks
        """
        soft_masks = f
        bs, t, h, w = f.size()
        f = torch.mean(f, 3)

        weight = torch.ones(1, 1, self.block_size, 1)
        if self.use_gpu: weight = weight.cuda()
        f = F.conv2d(input=f.view(-1, 1, h, 1),
                     weight=weight,
                     padding=(self.block_size // 2, 0))

        if self.block_size % 2 == 0:
            f = f[:, :, :-1]

        index = torch.argmax(f.view(bs * t, h), dim=1)

        # generate the masks
        masks = torch.zeros(bs * t, h)
        if self.use_gpu: masks = masks.cuda()
        index_b = torch.arange(0, bs * t, dtype=torch.long)
        masks[index_b, index] = 1

        block_masks = F.max_pool2d(input=masks[:, None, :, None],
                                   kernel_size=(self.block_size, 1),
                                   stride=(1, 1),
                                   padding=(self.block_size // 2, 0))
        if self.block_size % 2 == 0:
            block_masks = block_masks[:, :, 1:]

        block_masks = 1 - block_masks.view(bs, t, h, 1)
        return block_masks, soft_masks

    def erase_feature(self, x, masks, soft_masks):
        """erasing the x with the masks, the softmasks for gradient back-propagation.
        masks: bg is 1 while fg is 0
        """
        b, c, h, w = x.size()

        soft_masks = soft_masks - (1 - masks) * 1e8 # bg is positive while fg is negative
        soft_masks = F.softmax(soft_masks.view(b, h * w), 1)

        inputs = x * masks.unsqueeze(1)
        res = torch.bmm(x.view(b, c, h * w), soft_masks.unsqueeze(-1))
        outputs = inputs + self.conv_erase(res.unsqueeze(-1))
        return outputs

    def forward(self, x):
        b, c, t, h, w = x.size()
        m = torch.ones(b, h, w)
        if self.use_gpu: m = m.cuda()

        # forward the first frame with no erasing pixels
        x0 = x[:, :, 0]
        x0 = self.erase_feature(x0, m, m)  # [b, c, h, w]
        y0 = self.layer4(x0)
        y0 = self.branch[0](y0)

        # generate the erased attention maps for second frames.
        f = self.correlation(y0.detach(), x[:, :, 1:])
        masks, soft_masks = self.block_binarization(f)
        masks, soft_masks = masks[:, 0], soft_masks[:, 0]

        # forward the second frame with saliency erasing
        x1 = x[:, :, 1]
        x1 = self.erase_feature(x1, masks, soft_masks)
        y1 = self.layer4(x1)
        y1 = self.branch[1](y1)

        return [y0, y1]

class TSB(nn.Module):
    def __init__(self, in_channels, use_gpu=False, **kwargs):
        super(TSB, self).__init__()
        self.in_channels = in_channels
        self.use_gpu = use_gpu
        self.patch_size = 2

        self.W = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )

        self.pool = nn.AvgPool3d(kernel_size=(1, self.patch_size, self.patch_size),
                                 stride=(1, 1, 1), padding=(0, self.patch_size // 2, self.patch_size // 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W[1].weight.data, 0.0)
        nn.init.constant_(self.W[1].bias.data, 0.0)

    def forward(self, x):
        b, c, t, h, w = x.size()
        inputs = x

        query = x.view(b, c, t, -1).mean(-1)
        query = query.permute(0, 2, 1)

        memory = self.pool(x)
        if self.patch_size % 2 == 0:
            memory = memory[:, :, :, :-1, :-1]

        memory = memory.contiguous().view(b, 1, c, t * h * w)

        query = F.normalize(query, p=2, dim=2, eps=1e-12)
        memory = F.normalize(memory, p=2, dim=2, eps=1e-12)
        f = torch.matmul(query.unsqueeze(2), memory) * 5
        f = f.view(b, t, t, h * w)

        # mask the self-enhance
        mask = torch.eye(t).type(x.dtype)
        if self.use_gpu: mask = mask.cuda()
        mask = mask.view(1, t, t, 1)

        f = (f - mask * 1e8).view(b, t, t * h * w)
        f = F.softmax(f, dim=-1)

        y = x.view(b, c, t * h * w)
        y = torch.matmul(f, y.permute(0, 2, 1))
        y = self.W(y.view(b * t, c, 1, 1))
        y = y.view(b, t, c, 1, 1)
        y = y.permute(0, 2, 1, 3, 4)
        z = y + inputs

        return z

if __name__ == '__main__':
    layer4 = nn.Conv2d(1024, 2048, 1)
    branch = nn.Conv2d(2048, 2048, 1)
    x = torch.rand(2, 1024, 2, 16, 8)
    model = TSE(layer4, [branch, branch], use_gpu=False)
    res = model.forward(x)
    # model = TCLNet(num_classes=2, use_gpu=False)
    # print(model)
