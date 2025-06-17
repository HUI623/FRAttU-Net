#! /usr/bin/env/python3
# -*- coding=utf-8 -*-
'''
======================模块功能描述=========================    
       @File     : FRAttU_Net.py
       @IDE      : PyCharm
       @Author   : Wanghui-BIT
       @Date     : 2024/5/22 21:24
       @Desc     : v1
=========================================================   
'''


import torch
from torch import nn
from torch.nn import functional as F
import warnings
import torchvision
from models.FRAttU_Net.otsu import *
from gcn.layers import GConv

warnings.filterwarnings(action='ignore')

# 1. High and Low Frequency Feature Fusion and Enhancement Module
class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    
class UpSize(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, out_ch):
        super(UpSize, self).__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
    def forward(self, x1, x2):
        upzise1 = F.interpolate(x1, scale_factor=2, mode='nearest')  # 输入图X的尺寸，插值后变成原来的2倍，通道不变
        upzise2 = F.interpolate(x2, scale_factor=2, mode='nearest')  # 输入图X的尺寸，插值后变成原来的2倍，通道不变
        out = torch.cat([upzise1, upzise2], dim=1)   # 32 + 32 -> 64     
        return out
    


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),  # 图像大小不变
            nn.BatchNorm2d(out_channel),
            # 防止过拟合
            # nn.Dropout2d(0.3),
            nn.ReLU(),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.3),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out


# Position Attention Module
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.in_channel = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)   # 在最后一维上进行归一化

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()      # B C H W
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)  # B C N (N=W*H)-> B N C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)   # B C N
        # 用于计算两个具有相同批次大小的三维张量的矩阵乘法  (B,n,m) * (B,m,p)=(B,n,p)
        energy = torch.bmm(proj_query, proj_key)    # B N C * B C N = B N N
        attention = self.softmax(energy)  # B N N 归一化
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)  # B C N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B C N * B N N -> B C N
        out1 = out.view(m_batchsize, C, height, width)  # B C H W

        # out2 = self.gamma*out1 + x
        return out1

# Channel Attention Module,图像大小H*W
class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.in_channel = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)    # B C N ,x.view相当于numpy的reshape,-1表示一个不确定的数，不确定几列，但是确定是C行
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)   # B C N -> B N C
        energy = torch.bmm(proj_query, proj_key)  # B C N * B N C -> B C C （索引值0 1 2或是-3 -2 -1）
        # torch.max返回两个tensor，第一个tensor（[0]）是每行的最大值；第二个tensor是每行最大值的索引。-1表示在C上最大值
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  # 取C上最大值，扩展为与energy相同的size，然后减去energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)  # B C N = proj_query

        out = torch.bmm(attention, proj_value)  # B C C * B C N = B C N
        out1 = out.view(m_batchsize, C, height, width)   # B C H W

        out2 = self.gamma*out1 + x
        return out2

class HLFHead(nn.Module):
    def __init__(self, in_channel):
        super(HLFHead, self).__init__()
        self.in_channel = in_channel
        self.conv5a = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU())

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sa = PAM_Module(in_channel)
        self.sc = CAM_Module(in_channel)
        self.conv51 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        out1 = self.gamma*sa_feat + feat1
        sa_conv = self.conv51(out1)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        return feat_sum

# 困难区域注意力模块, 包含两个3*3卷积模块，注意力模块
class HLFPath(nn.Module):
    def __init__(self, in_channel, out_channel, i):  # 输入[64,512,512] 和 输出[128,256,256]
        super(HLFPath, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, i, 1, padding_mode='reflect', bias=False),  # C=64 图像尺寸缩小i=stride=2 4 8
            nn.BatchNorm2d(in_channel),
            # nn.Dropout2d(0.3),
            nn.ReLU(),

            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),  # 通道变C=128
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.3),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),   # 图像尺寸不变,通道减半 C=256->128
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.3),
            nn.ReLU()
        )
        self.hlfnet = HLFHead(out_channel)  # 64->128

    def forward(self, x, output):
    
        out1 = self.layer1(x)  # (N, C, w, h)
        out = torch.cat((out1, output), dim=1)  # f1和fi 拼接 C = 512
        out2 = self.layer2(out)
        output1 = output + out2  # (N, C, w, h)
        output2 = self.hlfnet(output1)

        return output2

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        out = self.mpconv(x)
        return out


class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpSample, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.layer = nn.Conv2d(in_ch, in_ch // 2, 1, 1)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x11 = self.up(x1)   # 上采样，降尺寸

        x12 = self.layer(x11)   # 通道减半 1024 -> 512

        diffY = x2.size()[2] - x12.size()[2]
        diffX = x2.size()[3] - x12.size()[3]

        x13 = F.pad(x12, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x3 = torch.cat([x2, x13], dim=1)   # 512+512 -> 1024

        x4 = self.conv(x3)   # 改通道  1024 -> 512
        return x4

# 2. Otsu Density Refinement Module
def otsu(img):
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 转换为灰度图
    # otsu_img = otsu_helper(gray, upper=118, down = 45,categories=1)
    otsu_img = otsu_helper(img, categories=1)
    return otsu_img

def dootsu(img):
    # se = img
    _, c, _, _ = img.size()
    # squeezelayer = nn.Conv2d(c, c // 16, kernel_size=1)
    # squeezelayer.cuda()
    # img = squeezelayer(img)
    img = img.cpu().detach()
    channel = list(img.size())[1]
    batch = list(img.size())[0]
    imgfolder = img.chunk(batch, dim=0)  # 把一个tensor均匀分割成若干个小tensor
    chw_output = []
    for index in range(batch):
        bchw = imgfolder[index]
        chw = bchw.squeeze()
        chwfolder = chw.chunk(channel, dim=0)
        hw_output = []
        for i in range(channel):
            hw = chwfolder[i].squeeze()
            hw = np.transpose(hw.detach().numpy(), (0, 1))
            hw_otsu = otsu(hw)
            hw_otsu = torch.from_numpy(hw_otsu)
            hw_output.append(hw_otsu)
        chw_otsu = torch.stack(hw_output, dim=0)
        chw_output.append(chw_otsu)
    bchw_otsu = torch.stack(chw_output, dim=0).cuda()
    # result = torch.cat([se.float().cuda(), bchw_otsu.float().cuda()],dim=1)
    return bchw_otsu



class FRAttUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FRAttUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.c1 = DoubleConv(n_channels, 64)  # 输入通道3，输出64，[3,512,512]-> [64,512,512]

        self.d1 = Down(64, 128)      # f1，通道不变64,[64,512,512]-> [64,256,256]
        self.h1 = HLFPath(64, 128, 2)  # [64,512,512]-> [128,256,256]

        self.d2 = Down(128, 256)     # f2，通道不变128, [128,256,256]->[128,128,128]
        self.h2 = HLFPath(64, 256, 4)  # [64,512,512]-> [256,128,128]

        self.d3 = Down(256, 512)     # f3，通道不变256,[256,128,128] -> [256,64,64]
        self.h3 = HLFPath(64, 512, 8)   # [64,512,512]-> [512,64,64]

        self.d4 = Down(512, 1024)     # f4，通道不变512,[512,64,64] -> [512,32,32]

        # 四次上采样
        self.u4 = UpSample(1024, 512)  # 通道变为512, 上采样[1024,32,32] -> [512,64,64]，和h3[512,64,64]拼接后 -> [1024,64,64]

        self.u3 = UpSample(512, 256)  # 上采样[512,64,64] -> [256,128,128]，和h2[256,128,128]拼接后 -> [512,128,128]

        self.u2 = UpSample(256, 128)  # 上采样[256,128,128] -> [128,256,256]，和h1[128,256,256]拼接后 -> [256,256,256]

        self.u1 = UpSample(128, 64)



        # self-attention
        self.pred_1 = single_conv(ch_in=64, ch_out=32)     # 图像尺寸减小一半
        self.pred_2 = single_conv(ch_in=64, ch_out=32)
        
        self.pred_11 = UpSize(in_ch=32, out_ch=32)  # 图像尺寸增加一半，两个特征图拼接，通道不变, 32 + 32 -> 64
        
        
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.conv_Atten1 = PAM_Module(32)
        self.conv_Atten2 = PAM_Module(32)
        
        # fusion module
        self.conv_fusion1 = DoubleConv(128, 64)  # 两个卷积层  96  128  160
        self.conv_fusion2 = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)  # v1


    # 前向计算，输出一张与原图相同尺寸的图片矩阵（特征图部分）
    def forward(self, x):
        R1 = self.c1(x)   # [64,512,512]
        R2 = self.d1(R1)        # 下采样后个卷积 f1map [128,256,256]
        R3 = self.d2(R2)      # 下采样后个卷积 f2map []
        R4 = self.d3(R3)     # 下采样后个卷积 f3map  512
        R5 = self.d4(R4)       # 下采样后个卷积 f4map  1024
        # print("R5", R5.shape)


        H3 = self.h3(R1, R4)
        O1 = self.u4(R5, H3)

        H2 = self.h2(R1, R3)
        O2 = self.u3(O1, H2)

        H1 = self.h1(R1, R2)
        O3 = self.u2(O2, H1)

        O4 = self.u1(O3, R1)

        pred_1 = self.pred_1(O4)   # 32
        pred_2 = self.pred_2(O4)   # 32
        
        # self-attention
        attention_higher = self.conv_Atten1(pred_1)     # 32
        out1 = dootsu(attention_higher)
        out2 = self.gamma * out1 + pred_1
           

        # fusion module  32+32+64 = 128        
        y = self.pred_11(out1, out2)
        
        y1 = torch.cat((y, O4), dim=1)  # C = 64 + 64
        
        y2 = self.conv_fusion1(y1)  # 128 -> 64
        pred = self.conv_fusion2(y2)  # 64 -> 3 

        return pred
        


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 128, 128).to(device)  # B最大是1，取2 4 8 16 都会超内存，该服务器是24G,灰度图通道是1  (1, 1, 128, 128)
    net = FRAttUNet(n_channels=1, n_classes=3).to(device)
    print(net(x).shape)  # torch.Size([1, 3, 128, 128]) 取256*256和512*512都会超内存