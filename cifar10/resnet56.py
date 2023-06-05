import sys
sys.path.append(".")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from flops import print_model_param_flops,print_model_param_nums
from utils import channles_selected

__all__ = ['ResNet56']


class BasicBlock(nn.Module):

    def __init__(self, in_planes,mid_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        self.baseNet = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(),

            nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.shortcut = nn.Sequential(nn.ReLU())
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(out_planes),
                     nn.ReLU()
                )


    def forward(self, x):
        x1 = self.baseNet(x)
        x1 += self.shortcut(x)
        return x1


class ResNet56(nn.Module):
    def __init__(self, num_classes=10, cfg=None):
        super(ResNet56, self).__init__()

        if cfg is None:
            cfg = [16,

                   16,16,
                   16,16,
                   16,16,
                   16,16,
                   16,16,
                   16,16,
                   16,16,
                   16,16,
                   16,16,
                   
                   16,32,
                   32,32,
                   32,32,
                   32,32,
                   32,32,
                   32,32,
                   32,32,
                   32,32,
                   32,32,
                   
                   32,64,
                   64,64,
                   64,64,
                   64,64,
                   64,64,
                   64,64,
                   64,64,
                   64,64,
                   64,64,]


        self.arc = OrderedDict([
            ('C1',nn.Sequential(nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1,bias=False),nn.BatchNorm2d(cfg[0]),nn.ReLU(inplace=True))),#(16,16,32,32)

            ('BasicBlock_1_1',BasicBlock(cfg[0],cfg[1],cfg[2],1)),
            ('BasicBlock_1_2',BasicBlock(cfg[2],cfg[3],cfg[4],1)),
            ('BasicBlock_1_3',BasicBlock(cfg[4],cfg[5],cfg[6],1)),
            ('BasicBlock_1_4',BasicBlock(cfg[6],cfg[7],cfg[8],1)),
            ('BasicBlock_1_5',BasicBlock(cfg[8],cfg[9],cfg[10],1)),
            ('BasicBlock_1_6',BasicBlock(cfg[10],cfg[11],cfg[12],1)),
            ('BasicBlock_1_7',BasicBlock(cfg[12],cfg[13],cfg[14],1)),
            ('BasicBlock_1_8',BasicBlock(cfg[14],cfg[15],cfg[16],1)),
            ('BasicBlock_1_9',BasicBlock(cfg[16],cfg[17],cfg[18],1)),#(16,16,32,32)

            ('BasicBlock_2_1',BasicBlock(cfg[18],cfg[19],cfg[20],2)),
            ('BasicBlock_2_2',BasicBlock(cfg[20],cfg[21],cfg[22],1)),
            ('BasicBlock_2_3',BasicBlock(cfg[22],cfg[23],cfg[24],1)),
            ('BasicBlock_2_4',BasicBlock(cfg[24],cfg[25],cfg[26],1)),
            ('BasicBlock_2_5',BasicBlock(cfg[26],cfg[27],cfg[28],1)),
            ('BasicBlock_2_6',BasicBlock(cfg[28],cfg[29],cfg[30],1)),
            ('BasicBlock_2_7',BasicBlock(cfg[30],cfg[31],cfg[32],1)),
            ('BasicBlock_2_8',BasicBlock(cfg[32],cfg[33],cfg[34],1)),
            ('BasicBlock_2_9',BasicBlock(cfg[34],cfg[35],cfg[36],1)),#(16,32,16,16)

            ('BasicBlock_3_1',BasicBlock(cfg[36],cfg[37],cfg[38],2)),
            ('BasicBlock_3_2',BasicBlock(cfg[38],cfg[39],cfg[40],1)),
            ('BasicBlock_3_3',BasicBlock(cfg[40],cfg[41],cfg[42],1)),
            ('BasicBlock_3_4',BasicBlock(cfg[42],cfg[43],cfg[44],1)),
            ('BasicBlock_3_5',BasicBlock(cfg[44],cfg[45],cfg[46],1)),
            ('BasicBlock_3_6',BasicBlock(cfg[46],cfg[47],cfg[48],1)),
            ('BasicBlock_3_7',BasicBlock(cfg[48],cfg[49],cfg[50],1)),
            ('BasicBlock_3_8',BasicBlock(cfg[50],cfg[51],cfg[52],1)),
            ('BasicBlock_3_9',BasicBlock(cfg[52],cfg[53],cfg[54],1)),#(16,64,8,8)


        ])

        self.features=nn.Sequential(self.arc)

        self.fc = nn.Linear(cfg[54], num_classes)


    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def channel_selected(self,x,t=2,s=0):
        masks = []
        cfg = []

        for name,conv in self.arc.items():
            if isinstance(conv,BasicBlock):
                x1 = x
                block = conv.baseNet
                shortcut = conv.shortcut
                hasshortcut = len(shortcut) == 3
                x = block[0](x)
                x = block[1](x)
                x = block[2](x)
                K = channles_selected(FM=x.cpu(),t=t,s=s)
                masks.append(K)
                cfg.append(len(K))
                if hasshortcut:
                    x = block[3](x)
                    x = block[4](x)
                    # K = channles_selected(FM=x.cpu(),n=n,s=s)
                    K = [k for k in range(block[3].out_channels)]
                    masks.append(K)
                    cfg.append(len(K))
                    x += shortcut(x1)
                else:
                    x = block[3](x)
                    x = block[4](x)
                    x += shortcut[0](x1)
                    K = [k for k in range(block[3].out_channels)]
                    masks.append(K)
                    cfg.append(len(K))
            else:
                x = conv[0](x)
                x = conv[1](x)
                x = conv[2](x)
                K = channles_selected(FM=x.cpu(),t=t,s=s)
                masks.append(K)
                cfg.append(len(K))

        print(cfg,len(cfg))
        return masks,cfg

