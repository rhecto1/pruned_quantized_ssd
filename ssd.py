"""
Copyright (c) 2017 Max deGroot, Ellis Brown
Released under the MIT license
https://github.com/amdegroot/ssd.pytorch
Updated by: Takuya Mouri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super().__init__()                                  # Avoid referring to the base class explicitly
        self.phase = phase                                  # This SSDs phase is either "test" or "train"
        self.num_classes = num_classes                      # Self-explanatory, set the number of classes to the parameter given
        self.cfg = (coco, voc)[num_classes == 21]           # Set the configuration as seen in config.py
        self.priorbox = PriorBox(self.cfg)                  #  I believe these are the default boxes 
        self.priors = self.priorbox.forward() 
        self.size = size                                    # Size of the input images, e.g. 300

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        # Network list of offsets and confidences
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # When running demo
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()

    # Direct broadcast
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        x = self.quant(x)
        for k in range(23):
            x = self.vgg[k](x)
        # Apply L2Norm to the calculation result of Conv4-3>Relu and add it to sources
        s = self.dequant(x)
        s = self.L2Norm(s)
        s = self.quant(s)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        # Added calculation result of Conv7>Relu to sources
        #s2 = self.dequant(x)
        sources.append(x)

        # Forward propagation by adding relu function to additional network
        # Add calculation result of odd-numbered layer to sources
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=False)
            if k % 2 == 1:
                sources.append(x)
        #x = self.dequant(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # (Batch size, C, W, H) → (Batch size, W, H, C) Transpose
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([self.dequant(o).view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([self.dequant(o).view(o.size(0), -1) for o in conf], 1)
        # When running demo
        if self.phase == "test":
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(self.dequant(x).data))                  # default boxes
            )
        else:
        # When train is executed
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        #print("output = ", output)

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
            
            
            


def extract_nbr(input_str):
    if input_str is None or input_str == '':
        return 0
    out_number = ''
    for ele in input_str:
        if ele.isdigit():
            out_number += ele
    return float(out_number) 
    
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# Creating a list of base networks
def vgg(cfg, i, batch_norm=False):
    layers = []
    inPlaceRelu = True
    in_channels = i
    for v in cfg:
        # Pooling layer 300 x 300 → 150 x 150
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # Round up the decimal point in the pooling layer 75×75 → 38×38
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=inPlaceRelu)]
            else:
                layers += [conv2d, nn.ReLU(inplace=inPlaceRelu)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
  
    layers += [pool5, conv6,
       nn.ReLU(inplace=inPlaceRelu), conv7, nn.ReLU(inplace=inPlaceRelu)]

    return layers


# List additional networks
def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    removeLayers=False
    if not removeLayers:
        in_channels = i
        flag = False
        for k, v in enumerate(cfg): # k is the kth element of cfg (e.g. 0..9), and v is the value of that element (e.g. 256 or S)
            #print("k =", k,"   v =", v)
            if in_channels != 'S':
                if v == 'S':
                    # Stride is 2
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                               kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
    return layers

# Create a list of offset and confidence networks
def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    # Added 21 Conv4-3 and -2 (second to last) Conv7 in the base to the list of feature maps
    for k, v in enumerate(vgg_source): # I.e. k = 0 v = 21,   k = 1 v = -2
        # Number of output layers is number of aspect ratio x number of coordinates
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # Number of output layers is number of aspect ratio x number of classes
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    # Add odd numbered layers to the list of feature maps
    for k, v in enumerate(extra_layers[1::2], 2):
        # Number of output layers is number of aspect ratio x number of coordinates
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        # Number of output layers is number of aspect ratio x number of classes
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

# Numbers are input channels, M, C are pooling, S is stride=2
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
# Number of aspect ratios for each feature map
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

# Creating a list of networks
def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # Network list of bases, additions, offsets, and confidences are arguments of class SSD
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)