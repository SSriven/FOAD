# code reference from https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/l1-norm-pruning/vgg.py
import math
import sys 
sys.path.append(".")
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import channles_selected
from flops import print_model_param_flops,print_model_param_nums


defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
defaultfc = [4096,4096]

class vgg_16(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, cfg=None):
        super(vgg_16, self).__init__()
        if cfg is None:
            cfg = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        self.cfg = cfg


        self.conv_block1 = nn.Sequential(nn.Conv2d(3, cfg[0], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[0]), 
                                        nn.ReLU(inplace=True))
        self.conv_block2 = nn.Sequential(nn.Conv2d(cfg[0], cfg[1], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[1]), 
                                        nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2, stride=2))    

        self.conv_block3 = nn.Sequential(nn.Conv2d(cfg[1], cfg[2], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[2]), 
                                        nn.ReLU(inplace=True))
        self.conv_block4 = nn.Sequential(nn.Conv2d(cfg[2], cfg[3], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[3]), 
                                        nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block5 = nn.Sequential(nn.Conv2d(cfg[3], cfg[4], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[4]), 
                                        nn.ReLU(inplace=True))
        self.conv_block6 = nn.Sequential(nn.Conv2d(cfg[4], cfg[5], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[5]), 
                                        nn.ReLU(inplace=True))                                                             
        self.conv_block7 = nn.Sequential(nn.Conv2d(cfg[5], cfg[6], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[6]), 
                                        nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block8 = nn.Sequential(nn.Conv2d(cfg[6], cfg[7], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[7]), 
                                        nn.ReLU(inplace=True))
        self.conv_block9 = nn.Sequential(nn.Conv2d(cfg[7], cfg[8], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[8]), 
                                        nn.ReLU(inplace=True))                                                             
        self.conv_block10 = nn.Sequential(nn.Conv2d(cfg[8], cfg[9], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[9]), 
                                        nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2, stride=2))


        self.conv_block11 = nn.Sequential(nn.Conv2d(cfg[9], cfg[10], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[10]), 
                                        nn.ReLU(inplace=True))
        self.conv_block12 = nn.Sequential(nn.Conv2d(cfg[10], cfg[11], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[11]), 
                                        nn.ReLU(inplace=True))
        self.conv_block13 = nn.Sequential(nn.Conv2d(cfg[11], cfg[12], kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(cfg[12]), 
                                        nn.ReLU(inplace=True))

        self.conv_block = [self.conv_block1,self.conv_block2,self.conv_block3,self.conv_block4,
                            self.conv_block5,self.conv_block6,self.conv_block7,self.conv_block8,
                            self.conv_block9,self.conv_block10,self.conv_block11,self.conv_block12,
                            self.conv_block13]

        self.classifier = nn.Linear(cfg[12], num_classes)


        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        for conv in self.conv_block:
            x = conv(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def channel_selected(self,x,t=2,s = 0.5):
        masks = []
        cfg = []

        for conv in self.conv_block:
            x = conv(x)
            K = channles_selected(FM=x.cpu(),t=t,s=s)
            masks.append(K)
            cfg.append(len(K))
            # break
        return masks,cfg
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
