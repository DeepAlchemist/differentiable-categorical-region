# Non-local block using embedded gaussian
# Code from
# Last Change:  2022-08-28 15:33:32
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/
import torch
from torch import nn
from torch.nn import functional as F
from fastreid.modeling.backbones import simple_resnet

class _NonLocalBlockND(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=3,
                 sub_sample=False):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.sub_sample = sub_sample
        if sub_sample:
            # self.g = nn.Sequential(self.g, max_pool_layer)
            # self.phi = nn.Sequential(self.phi, max_pool_layer)
            self.register_buffer("prob_maps", torch.rand(48, 5, 24, 8))  # (b num_parts h w)

        self.norm_type = 'div'
        assert self.norm_type in ['div', 'soft']

    def sub_sample_forward(self, x):
        """
        Args:
            x: (b, c, h, w)
        """
        batch_size = x.size(0)
        num_keys = self.prob_maps.size(1)
        assert self.prob_maps.size(0) == batch_size
        if self.prob_maps.size(2) == x.size(2) and self.prob_maps.size(3) == x.size(3):
            prob_maps = self.prob_maps
        else:
            prob_maps = F.interpolate(self.prob_maps, size=x.size()[-2:], mode='bilinear', align_corners=True)

        # 1. g
        g_x = self.g(x)  # (b c h w)
        g_x = prob_maps.view(batch_size, num_keys, -1).matmul(  # (b num_keys hw)
            g_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # (b hw c)
        )  # (b num_keys c)
        g_x = g_x / (x.size(2) * x.size(3))

        # 2. theta
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (b c thw)
        theta_x = theta_x.permute(0, 2, 1)  # (b thw c)

        # 3. phi
        phi_x = self.phi(x)  # (b c h w)
        phi_x = prob_maps.view(batch_size, num_keys, -1).matmul(  # (b num_keys hw)
            phi_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # (b hw c)
        ).permute(0, 2, 1)  # (b num_keys c) to (b c num_keys)
        phi_x = phi_x / (x.size(2) * x.size(3))

        # 4. matrix
        f = torch.matmul(theta_x, phi_x)  # (b hw num_keys)
        f_div_C = f / f.size(-1) if self.norm_type == 'div' else F.softmax(f, dim=-1)  # (b hw num_keys)

        # 5. weighted addition
        y = torch.matmul(f_div_C, g_x)  # (b hw c)
        y = y.permute(0, 2, 1).contiguous()  # (b c hw)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (b c h w)

        # 6. residuals
        W_y = self.W(y)
        z = W_y + x
        return z

    def forward(self, x):
        """
        Args:
            x: NL3D (b, c, t, h, w) or  NL2D (b, c, h, w)
        Returns:
        """
        if self.sub_sample:
            return self.sub_sample_forward(x)

        batch_size = x.size(0)
        # g
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (b c thw)
        g_x = g_x.permute(0, 2, 1)  # (b thw c)
        # theta
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (b c thw)
        theta_x = theta_x.permute(0, 2, 1)  # (b thw c)
        # phi
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (b c thw)

        f = torch.matmul(theta_x, phi_x)  # (b theta_thw phi_thw)
        f_div_C = f / f.size(-1) if self.norm_type == 'div' else F.softmax(f, dim=-1)  # (b theta_thw phi_thw)

        y = torch.matmul(f_div_C, g_x)  # (b thw c)
        y = y.permute(0, 2, 1).contiguous()  # (b c thw)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (b c t h w)

        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample)

class NL2DWrapper(nn.Module):
    def __init__(self, block, sub_sample=False):
        super(NL2DWrapper, self).__init__()
        self.block = block
        self.nl = NONLocalBlock2D(block.bn3.num_features, sub_sample=sub_sample)

    def forward(self, x):
        x = self.block(x)
        x = self.nl(x)  # (n, c, h, w)
        return x

def make_non_local(net, sub_sample=False):
    # Non-local
    if isinstance(net, simple_resnet.ResNet):
        # opt1
        # net.layer2 = nn.Sequential(
        #     NL2DWrapper(net.layer2[0], sub_sample),
        #     net.layer2[1],
        #     NL2DWrapper(net.layer2[2], sub_sample),
        #     net.layer2[3],
        # )
        # net.layer3 = nn.Sequential(
        #     NL2DWrapper(net.layer3[0], sub_sample),
        #     net.layer3[1],
        #     NL2DWrapper(net.layer3[2], sub_sample),
        #     net.layer3[3],
        #     NL2DWrapper(net.layer3[4], sub_sample),
        #     net.layer3[5],
        # )
        # opt2
        # net.layer3 = nn.Sequential(
        #     NL2DWrapper(net.layer3[0], sub_sample),
        #     net.layer3[1],
        #     NL2DWrapper(net.layer3[2], sub_sample),
        #     net.layer3[3],
        #     NL2DWrapper(net.layer3[4], sub_sample),
        #     net.layer3[5],
        # )
        # opt3
        net.layer3 = nn.Sequential(
            net.layer3[0],
            net.layer3[1],
            net.layer3[2],
            NL2DWrapper(net.layer3[3], sub_sample),
            NL2DWrapper(net.layer3[4], sub_sample),
            net.layer3[5],
        )
        net.layer4 = nn.Sequential(
            NL2DWrapper(net.layer4[0], sub_sample),
            NL2DWrapper(net.layer4[1], sub_sample),
            net.layer4[2],
        )
    else:
        raise NotImplementedError

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True

    img = Variable(torch.zeros(2, 3, 384, 128))
    net = NONLocalBlock2D(3, sub_sample=sub_sample)
    out = net(img)
    print(out.size())
