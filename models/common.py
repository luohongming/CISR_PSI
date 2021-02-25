import math

import torch
import torch.nn as nn

import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:   #Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4*n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9*n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ResidualBlock(nn.Module):
    def __init__(self, input_nc, output_nc, stride=1, dowmsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(input_nc, output_nc, 3, stride)
        self.bn1 = nn.BatchNorm2d(output_nc)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(output_nc, output_nc, 3, 1)
        self.bn2 = nn.BatchNorm2d(output_nc)
        self.downsample = dowmsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

def pixel_shuffle_downscale(input, scale_factor):
    b, c, h, w = input.size()
    scale_factor_squared = scale_factor * scale_factor

    oc = c * scale_factor_squared
    oh = h // scale_factor
    ow = w // scale_factor

    input_reshaped = input.reshape(b, c, oh, scale_factor, ow, scale_factor)
    return input_reshaped.permute(0, 1, 3, 5, 2, 4).reshape(b, oc, oh, ow)


# add NonLocalBlock2D
# reference: https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        split_num = 50
        k = g_x.size(1) // split_num
        y_all = torch.zeros(g_x.size()).cuda()
        for i in range(split_num):
            if i == split_num - 1:
                f = torch.matmul(theta_x[:, k*i:, :], phi_x)
            else:
                f = torch.matmul(theta_x[:, k*i:k*(i+1), :], phi_x)

        # f = torch.matmul(theta_x, phi_x)

            f_div_C = F.softmax(f, dim=1)

            y = torch.matmul(f_div_C, g_x)
            if i == split_num - 1:
                y_all[:, k*i:, :] = y
            else:
                y_all[:, k*i:k*(i+1), :] = y


        y = y_all.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


## define trunk branch
class TrunkBranch(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        tx = self.body(x)

        return tx


## define mask branch
class MaskBranchDownUp(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(MaskBranchDownUp, self).__init__()

        MB_RB1 = []
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat, n_feat, 6, stride=2, padding=2))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        if x_RB1.size() != x_Up.size():
            x_Up = F.interpolate(x_Up, size=[x_RB1.size(2), x_RB1.size(3)])
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


## define nonlocal mask branch
class NLMaskBranchDownUp(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLMaskBranchDownUp, self).__init__()

        MB_RB1 = []
        MB_RB1.append(NonLocalBlock2D(n_feat, n_feat // 2))
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat, n_feat, 6, stride=2, padding=2))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        if x_RB1.size() != x_Up.size():
            x_Up = F.interpolate(x_Up, size=[x_RB1.size(2), x_RB1.size(3)])
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


## define residual attention module
class ResAttModuleDownUpPlus(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


## define nonlocal residual attention module
class NLResAttModuleDownUpPlus(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(NLMaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    a = torch.ones(1, 1, 4, 4)
    b = torch.cat([a*1, a*2, a*3], dim=1)
    # b = a.reshape(1, 1, 4, 4)
    b_np = b.numpy()
    c = pixel_shuffle_downscale(b, 2)
    k = c.size(1) // b.size(1)
    kernel = torch.ones(b.size(1), k, 1, 1) * (1 / k)
    c_np = c.numpy()
    d = F.conv2d(c, kernel, stride=1, groups=b.size(1))
    d_np = d.numpy()

    print(c.size())
