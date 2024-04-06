import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
import cv2
import numpy as np
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Initialization model
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    temp = torch.nn.PixelShuffle(32)(temp)
    return temp

class Basic_Block(nn.Module):
    def __init__(self, dim):
        super(Basic_Block, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.cab = ResidualGroup(default_conv, n_feat=dim, reduction=2, kernel_size=3, n_resblocks=10)
        self.conv1 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, padding=0, bias=True)
        self.cat = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x,r,z,Phi,PhiT):
        #range space
        r = self.alpha*r
        #null space
        x = self.conv1(x)
        x = self.cat(torch.cat((x,z),dim=1))
        x = self.cab(x)
        z = x
        x = self.conv2(x)
        n = x - self.alpha*PhiTPhi_fun(x, Phi, PhiT)
        x = r+n
        return x,z

class Net(nn.Module):
    def __init__(self, sensing_rate, LayerNo, channel_number):
        super(Net, self).__init__()
        act = nn.ReLU(True)
        self.LayerNo = LayerNo
        self.measurement = int(sensing_rate * 1024)
        self.base = channel_number
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)
        self.cab = ResidualGroup(default_conv,n_feat=self.base,reduction=2,kernel_size=3,n_resblocks=10)
        self.alpha = nn.Parameter(torch.Tensor([1.0]))
        onelayer = []
        for i in range(self.LayerNo):
            onelayer.append(Basic_Block(self.base))
        self.RND = nn.ModuleList(onelayer)
    def forward(self, x): 
        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)
        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)
        x = F.conv2d(y, PhiT, padding=0, bias=None)
        x = nn.PixelShuffle(32)(x)
        r = self.alpha * x
        x = self.conv1(x)
        x = self.cab(x)
        z = x
        x = self.conv2(x)
        n = x - self.alpha*PhiTPhi_fun(x, Phi, PhiT)
        x = r + n
        for i in range(self.LayerNo):
            x,z = self.RND[i](x,r,z,Phi,PhiT)

        return x
