import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet50_AM']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiScaleAttnBlock(nn.Module):
    """
    ICCV19 Discriminative Feature Learning with Consistent Attention Regularization for Person Re-identiﬁcation
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

    def forward(self, x):
        x = self.conv_agg(self.conv3(x) + self.conv5(x) + self.conv7(x))
        x = x.sigmoid()  # (bt, 1, h, w)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead

            # -----------------------------
            # modified
            replace_stride_with_dilation = [False, False, False]
            # -----------------------------

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.am1 = MultiScaleAttnBlock(64, 2, stride=1)
        self.am2 = MultiScaleAttnBlock(256, 2, stride=2)
        self.am3 = MultiScaleAttnBlock(512, 2, stride=2)
        self.am4 = MultiScaleAttnBlock(1024, 2, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        attn1 = self.am1(x)
        x1 = self.layer1(x) * attn1

        attn2 = self.am2(x1)
        x2 = self.layer2(x1) * attn2

        attn3 = self.am3(x2)
        x3 = self.layer3(x2) * attn3

        attn4 = self.am4(x3)
        x4 = self.layer4(x3) * attn4

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x4, [attn1, attn2, attn3, attn4]
        # return [x2, x3, x4]


def remove_fc(state_dict):
    """ Remove the fc layer parameter from state_dict. """
    return {key: value for key, value in state_dict.items() if not key.startswith('fc.') and not key.startswith('am.')}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # model.load_state_dict(remove_fc(state_dict))
        # model.load_state_dict(state_dict)
        model_params = model.state_dict()
        for key, value in state_dict.items():
            if not key.startswith('fc.') and not key.startswith('am.'):
                model_params[key] = value

        model.load_state_dict(model_params)

    return model


def resnet50_AM(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class CAR(nn.Module):
    r""" Discriminative Feature Learning with Consistent Attention Regularization.
    """

    def __init__(self, nattr, c_in, bn=False, pool='max', scale=1):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(c_in, nattr, bias=False)
        self.align_pool = nn.MaxPool2d(kernel_size=2)

    def fresh_params(self):
        return self.parameters()

    def forward(self, x, gt):
        feat, heatmap = x
        hm1, hm2, hm3, hm4 = heatmap
        bt, c, h, w = feat.shape
        feat = self.pool(feat).reshape(bt, c)
        logits = self.linear(feat)

        sparse_loss = torch.norm(hm1.reshape(bt, -1), dim=1, p=1) + torch.norm(hm2.reshape(bt, -1), dim=1, p=1) + \
                      torch.norm(hm4.reshape(bt, -1), dim=1, p=1) + torch.norm(hm3.reshape(bt, -1), dim=1, p=1)

        sparse_loss = sparse_loss / 4  # (bt)

        align_loss = torch.norm((self.align_pool(hm1) - hm2).reshape(bt, -1), p=2, dim=1) + \
                     torch.norm((self.align_pool(hm2) - hm3).reshape(bt, -1), p=2, dim=1) + \
                     torch.norm((self.align_pool(hm3) - hm4).reshape(bt, -1), p=2, dim=1)
        align_loss = align_loss / 3

        return [logits, sparse_loss.mean(), align_loss.mean()], [feat, feat]

if __name__ == '__main__':
    # print(resnet50())
    model = resnet50_AM().cuda()
    x = torch.rand((1, 3, 256, 128)).cuda()
    model(x)

    # print('receptive_field_dict')