import os
import argparse
import numpy as np
import logging

from pysot.core.config import cfg
from pysot.models.backbone.resnet_atrous import ResNet, Bottleneck
from pysot.models.backbone.mobile_v2 import MobileNetV2
from pysot.models.head.rpn import RPN

import torch
from torch import nn

# argparse check function
def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"Config file:{path} is not a valid file")

# argument parsing
parser = argparse.ArgumentParser(description='Script for converting the pytorch model (.pth) into onnx format.')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--snapshot', required=True, type=file_path, help="Path to the model file (.pth)")
required.add_argument('--config', required=True, type=file_path, help='path to config file')
args = parser.parse_args()


class AdjustLayer_1(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer_1, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        l = 4
        r = 11
        x = x[:, :, l:r, l:r]
        return x

class AdjustAllLayer_1(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer_1, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer_1(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer_1(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out

class AdjustLayer_2(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer_2, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        return x

class AdjustAllLayer_2(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer_2, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer_2(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer_2(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out


def xcorr_depthwise(x, kernel):
    """
    Deptwise convolution for input and weights with different shapes
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    conv = nn.Conv2d(batch*channel, batch*channel, kernel_size=(kernel.size(2), kernel.size(3)), bias=False, groups=batch*channel)
    conv.weight = nn.Parameter(kernel)
    out = conv(x) 
    out = out.view(batch, channel, out.size(2), out.size(3))
    out = out.detach()
    return out

class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels):
        super(MultiRPN, self).__init__()
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        self.weight_cls = nn.Parameter(torch.Tensor([0.38156851768108546, 0.4364767608115956,  0.18195472150731892]))
        self.weight_loc = nn.Parameter(torch.Tensor([0.17644893463361863, 0.16564198028417967, 0.6579090850822015]))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        
        rpn2 = self.rpn2
        z_f2 = z_fs[0]
        x_f2 = x_fs[0]
        c2,l2 = rpn2(z_f2, x_f2)
        
        cls.append(c2)
        loc.append(l2)
        
        rpn3 = self.rpn3
        z_f3 = z_fs[1]
        x_f3 = x_fs[1]
        c3,l3 = rpn3(z_f3, x_f3)
        
        cls.append(c3)
        loc.append(l3)
        
        rpn4 = self.rpn4
        z_f4 = z_fs[2]
        x_f4 = x_fs[2]
        c4,l4 = rpn4(z_f4, x_f4)
        
        cls.append(c4)
        loc.append(l4)
        
        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            fixed_len = 3
            for i in range(3):
                s += lst[i] * weight[i]
            return s

        return weighted_avg(cls, self.weight_cls), weighted_avg(loc, self.weight_loc)

class ConvCorrPrep(nn.Module):
    def __init__(self, in_channels, hidden, kernel_size=3):
        super(ConvCorrPrep, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
    
    def forward(self, z):
        return self.conv(z)

class XCorrPrepTarget(nn.Module):
    def __init__(self, anchor_num, in_channels):
        super(XCorrPrepTarget, self).__init__()
        out_channels = in_channels
        for i in range(len(in_channels)):
            self.add_module('xcorr_prep_target_cls'+str(i+2), ConvCorrPrep(in_channels[i], out_channels[i]))
            self.add_module('xcorr_prep_target_loc'+str(i+2), ConvCorrPrep(in_channels[i], out_channels[i]))
    
    def forward(self, z_fs):
        kernel_cls_2 = self.xcorr_prep_target_cls2(z_fs[0])
        kernel_loc_2 = self.xcorr_prep_target_loc2(z_fs[0])
        kernel_cls_3 = self.xcorr_prep_target_cls3(z_fs[1])
        kernel_loc_3 = self.xcorr_prep_target_loc3(z_fs[1])
        kernel_cls_4 = self.xcorr_prep_target_cls4(z_fs[2])
        kernel_loc_4 = self.xcorr_prep_target_loc4(z_fs[2])
        return [kernel_cls_2, kernel_cls_3, kernel_cls_4, kernel_loc_2, kernel_loc_3, kernel_loc_4]

class XCorrPrepSearch(nn.Module):
    def __init__(self, anchor_num, in_channels):
        super(XCorrPrepSearch, self).__init__()
        out_channels = in_channels
        for i in range(len(in_channels)):
            self.add_module('xcorr_prep_search_cls'+str(i+2), ConvCorrPrep(in_channels[i], out_channels[i]))
            self.add_module('xcorr_prep_search_loc'+str(i+2), ConvCorrPrep(in_channels[i], out_channels[i]))
    
    def forward(self, x_fs):
        search_cls_2 = self.xcorr_prep_search_cls2(x_fs[0])
        search_loc_2 = self.xcorr_prep_search_loc2(x_fs[0])
        search_cls_3 = self.xcorr_prep_search_cls3(x_fs[1])
        search_loc_3 = self.xcorr_prep_search_loc3(x_fs[1])
        search_cls_4 = self.xcorr_prep_search_cls4(x_fs[2])
        search_loc_4 = self.xcorr_prep_search_loc4(x_fs[2])
        return [search_cls_2, search_cls_3, search_cls_4, search_loc_2, search_loc_3, search_loc_4]


from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

def to_numpy(tensor):
    return np.ascontiguousarray(tensor.detach().cpu().numpy()) if tensor.requires_grad else np.ascontiguousarray(tensor.cpu().numpy())

class TargetNetBuilder(nn.Module):
    def __init__(self, cfg):
        super(TargetNetBuilder, self).__init__()
        # Build Backbone Model
        if 'resnet50' in cfg.BACKBONE.TYPE:
            self.backbone = ResNet(Bottleneck, [3,4,6,3], [2,3,4])
        elif 'mobilenetv2' in cfg.BACKBONE.TYPE:
            self.backbone = MobileNetV2(**cfg.BACKBONE.KWARGS)
        else:
            return None
        # Build Neck Model
        self.neck = AdjustAllLayer_1(**cfg.ADJUST.KWARGS)
        # Build Corr_prep Target Model
        self.xcorr_prep = XCorrPrepTarget(anchor_num=cfg.RPN.KWARGS.anchor_num, in_channels=cfg.RPN.KWARGS.in_channels)
    
    def forward(self, frame):
        features = self.backbone(frame)
        features = self.neck(features)
        output = self.xcorr_prep(features)
        return output

class SearchNetBuilder(nn.Module):
    def __init__(self, cfg):
        super(SearchNetBuilder, self).__init__()
        # Build Backbone Model
        if 'resnet50' in cfg.BACKBONE.TYPE:
            self.backbone = ResNet(Bottleneck, [3,4,6,3], [2,3,4])
        elif 'mobilenetv2' in cfg.BACKBONE.TYPE:
            self.backbone = MobileNetV2(**cfg.BACKBONE.KWARGS)
        else:
            return None
        # Build Neck Model
        self.neck = AdjustAllLayer_2(**cfg.ADJUST.KWARGS)
        # Build Corr_prep Search Model
        self.xcorr_prep = XCorrPrepSearch(anchor_num=cfg.RPN.KWARGS.anchor_num, in_channels=cfg.RPN.KWARGS.in_channels)
        
    def forward(self, frame):
        features = self.backbone(frame)
        features = self.neck(features)
        output = self.xcorr_prep(features)
        return output

class DepthwiseXCorr(nn.Module):
    "Depthwise Correlation Layer"
    def __init__(self, in_channels):
        super(DepthwiseXCorr, self).__init__()
        for i in range(len(in_channels)):
            self.add_module('dw_xcorr_cls'+str(i+2), nn.Conv2d(in_channels[i], in_channels[i], kernel_size=(5,5), bias=False, groups=in_channels[i]))
            self.add_module('dw_xcorr_loc'+str(i+2), nn.Conv2d(in_channels[i], in_channels[i], kernel_size=(5,5), bias=False, groups=in_channels[i]))
    
    def init(self, kernel):
        "initialize the conv weights with the output from target_net"
        kernel[0] = kernel[0].view(256, 1, 5, 5)
        kernel[1] = kernel[1].view(256, 1, 5, 5)
        kernel[2] = kernel[2].view(256, 1, 5, 5)
        kernel[3] = kernel[3].view(256, 1, 5, 5)
        kernel[4] = kernel[4].view(256, 1, 5, 5)
        kernel[5] = kernel[5].view(256, 1, 5, 5)
        self.dw_xcorr_cls2.weight = nn.Parameter(kernel[0])
        self.dw_xcorr_cls3.weight = nn.Parameter(kernel[2])
        self.dw_xcorr_cls4.weight = nn.Parameter(kernel[4])
        self.dw_xcorr_loc2.weight = nn.Parameter(kernel[1])
        self.dw_xcorr_loc3.weight = nn.Parameter(kernel[3])
        self.dw_xcorr_loc4.weight = nn.Parameter(kernel[5])
    
    def forward(self, search):
        cls = []
        loc = []
        cls_i = self.dw_xcorr_cls2(search[0])
        cls_i = cls_i.view(1, 256, cls_i.size(2), cls_i.size(3))
        cls.append(cls_i)
        loc_i = self.dw_xcorr_loc2(search[1])
        loc_i = loc_i.view(1, 256, loc_i.size(2), loc_i.size(3))
        loc.append(loc_i)
        cls_i = self.dw_xcorr_cls3(search[2])
        cls_i = cls_i.view(1, 256, cls_i.size(2), cls_i.size(3))
        cls.append(cls_i)
        loc_i = self.dw_xcorr_loc3(search[3])
        loc_i = loc_i.view(1, 256, loc_i.size(2), loc_i.size(3))
        loc.append(loc_i)
        cls_i = self.dw_xcorr_cls4(search[4])
        cls_i = cls_i.view(1, 256, cls_i.size(2), cls_i.size(3))
        cls.append(cls_i)
        loc_i = self.dw_xcorr_loc4(search[5])
        loc_i = loc_i.view(1, 256, loc_i.size(2), loc_i.size(3))
        loc.append(loc_i)
        return cls, loc


class XCorrBuilder(nn.Module):
    def __init__(self, hidden, out_channels_cls, out_channels_loc):
        super(XCorrBuilder, self).__init__()
        # depthwise xcorrelation with z as kernel
        self.dwxcorr = DepthwiseXCorr([hidden, hidden, hidden])
        # head 1x1 conv
        self.head_cls2 = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels_cls, kernel_size=1)
                )
        self.head_cls3 = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels_cls, kernel_size=1)
                )
        self.head_cls4 = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels_cls, kernel_size=1)
                )
        self.head_loc2 = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels_loc, kernel_size=1)
                )
        self.head_loc3 = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels_loc, kernel_size=1)
                )
        self.head_loc4 = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels_loc, kernel_size=1)
                )
        weight_cls2 = nn.Parameter(torch.Tensor([0.38156851768108546]))
        weight_cls3 = nn.Parameter(torch.Tensor([0.4364767608115956]))
        weight_cls4 = nn.Parameter(torch.Tensor([0.18195472150731892]))
        weight_loc2 = nn.Parameter(torch.Tensor([0.17644893463361863]))
        weight_loc3 = nn.Parameter(torch.Tensor([0.16564198028417967]))
        weight_loc4 = nn.Parameter(torch.Tensor([0.6579090850822015]))
        self.weight_cls = [weight_cls2, weight_cls3, weight_cls4]
        self.weight_loc = [weight_loc2, weight_loc3, weight_loc4]
    
    def init(self, kernel):
        # set kernel as weights for depthwise xcorrelation
        self.dwxcorr.init(kernel)
    
    def forward(self, search):
        cls, loc = self.dwxcorr(search)
        cls[0] = self.head_cls2(cls[0])
        cls[1] = self.head_cls3(cls[1])
        cls[2] = self.head_cls4(cls[2])
        loc[0] = self.head_loc2(loc[0])
        loc[1] = self.head_loc3(loc[1])
        loc[2] = self.head_loc4(loc[2])
        
        def avg(lst):
            return sum(lst) / len(lst)
        
        def weighted_avg(lst, weight):
            s = 0
            fixed_len = 3
            for i in range(3):
                s += lst[i] * weight[i]
            return s
        # return weighted_avg(cls, self.weight_cls), weighted_avg(loc, self.weight_loc)
        return cls, loc

class RPNBuilder(nn.Module):
    def __init__(self):
        super(RPNBuilder, self).__init__()

        # Build Adjusted Layer Builder
        self.rpn_head = MultiRPN(anchor_num=5,in_channels=[256, 256, 256])

    def forward(self, zf, xf):
        # Get Feature
        cls, loc = self.rpn_head(zf, xf)

        return cls, loc


def main():
    logging.info("Start of onnx conversion.")

    # load config 
    cfg.merge_from_file(args.config)

    logging.info("Load pretrained model from: " + str(args.snapshot))
    pretrained_dict = torch.load(args.snapshot, map_location=lambda storage, loc: storage.cuda(0))
    pretrained_dict_xcorr = {}#pretrained_dict
    pretrained_dict_target = {}#pretrained_dict
    pretrained_dict_search = {}#pretrained_dict

    # The shape of the inputs to the Target Network and the Search Network
    target = torch.Tensor(np.random.rand(1,3,127,127))
    search = torch.Tensor(np.random.rand(1,3,255,255))

    #===================================================================================================================
    # Build the torch backbone model
    logging.info("Build the target_net model")
    target_net = TargetNetBuilder(cfg)
    target_net.eval()
    target_net.state_dict().keys()
    target_net_dict = target_net.state_dict()

    # kernel = target_net.forward(target)

    # Load the pre-trained weight to the torch target net model
    # pretrained_dict_target = {k: v for k, v in pretrained_dict.items() if k in target_net_dict}
    for k, v in pretrained_dict.items():
        if k in target_net_dict:
            pretrained_dict_target[k] = v
        else:
            if k.startswith('rpn_head.rpn2.cls.conv_kernel'):
                tmp = k.split('rpn_head.rpn2.cls.conv_kernel')
                pretrained_dict_target['xcorr_prep.xcorr_prep_target_cls2.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn2.loc.conv_kernel'):
                tmp = k.split('rpn_head.rpn2.loc.conv_kernel')
                pretrained_dict_target['xcorr_prep.xcorr_prep_target_loc2.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn3.cls.conv_kernel'):
                tmp = k.split('rpn_head.rpn3.cls.conv_kernel')
                pretrained_dict_target['xcorr_prep.xcorr_prep_target_cls3.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn3.loc.conv_kernel'):
                tmp = k.split('rpn_head.rpn3.loc.conv_kernel')
                pretrained_dict_target['xcorr_prep.xcorr_prep_target_loc3.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn4.cls.conv_kernel'):
                tmp = k.split('rpn_head.rpn4.cls.conv_kernel')
                pretrained_dict_target['xcorr_prep.xcorr_prep_target_cls4.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn4.loc.conv_kernel'):
                tmp = k.split('rpn_head.rpn4.loc.conv_kernel')
                pretrained_dict_target['xcorr_prep.xcorr_prep_target_loc4.conv'+tmp[-1]] = v

    target_net_dict.update(pretrained_dict_target)
    target_net.load_state_dict(target_net_dict)
    # target_net.cuda()

    # tmp_target_net = list(target_net_dict)[250:]
    # tmp_pretrained = list(pretrained_dict)[250:]

    # Export the torch target net model to ONNX model
    logging.info("Export the target_net model ...")
    torch.onnx.export(target_net, torch.Tensor(target), "target_net4refit.onnx", export_params=True,input_names=['input'], output_names=['kernel_cls2', 'kernel_cls3', 'kernel_cls4', 'kernel_loc2', 'kernel_loc3', 'kernel_loc4'])
    logging.info("Export the target_net model DONE.")

    # Check whether the ONNX target net model has been successfully imported
    # onnx.checker.check_model(onnx_target)
    # print(onnx.checker.check_model(onnx_target))
    # onnx.helper.printable_graph(onnx_target.graph)
    # print(onnx.helper.printable_graph(onnx_target.graph))
    #===================================================================================================================

    #===================================================================================================================
    # Build the torch backbone model
    logging.info("Build the search_net model")
    search_net = SearchNetBuilder(cfg)
    search_net.eval()
    search_net.state_dict().keys()
    search_net_dict = search_net.state_dict()

    # Load the pre-trained weight to the torch target net model
    # pretrained_dict_search = {k: v for k, v in pretrained_dict_search.items() if k in search_net_dict}
    for k, v in pretrained_dict.items():
        if k in search_net_dict:
            pretrained_dict_search[k] = v
        else:
            if k.startswith('rpn_head.rpn2.cls.conv_search'):
                tmp = k.split('rpn_head.rpn2.cls.conv_search')
                pretrained_dict_search['xcorr_prep.xcorr_prep_search_cls2.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn2.loc.conv_search'):
                tmp = k.split('rpn_head.rpn2.loc.conv_search')
                pretrained_dict_search['xcorr_prep.xcorr_prep_search_loc2.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn3.cls.conv_search'):
                tmp = k.split('rpn_head.rpn3.cls.conv_search')
                pretrained_dict_search['xcorr_prep.xcorr_prep_search_cls3.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn3.loc.conv_search'):
                tmp = k.split('rpn_head.rpn3.loc.conv_search')
                pretrained_dict_search['xcorr_prep.xcorr_prep_search_loc3.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn4.cls.conv_search'):
                tmp = k.split('rpn_head.rpn4.cls.conv_search')
                pretrained_dict_search['xcorr_prep.xcorr_prep_search_cls4.conv'+tmp[-1]] = v
            elif k.startswith('rpn_head.rpn4.loc.conv_search'):
                tmp = k.split('rpn_head.rpn4.loc.conv_search')
                pretrained_dict_search['xcorr_prep.xcorr_prep_search_loc4.conv'+tmp[-1]] = v

    search_net_dict.update(pretrained_dict_search)
    search_net.load_state_dict(search_net_dict)
    # search_net.cuda()

    # Export the torch search net model to ONNX model
    logging.info("Export the search_net model ...")
    torch.onnx.export(search_net, torch.Tensor(search), "search_net4refit.onnx", export_params=True, 
                  input_names=['input'], output_names=['search_cls2', 'search_cls3', 'search_cls4', 'search_loc2', 'search_loc3', 'search_loc4'])
    logging.info("Export the search_net model DONE.")

    search = search_net.forward(search)

    # Check whether the ONNX target net model has been successfully imported
    # onnx.checker.check_model(onnx_search)
    # print(onnx.checker.check_model(onnx_search))
    # onnx.helper.printable_graph(onnx_search.graph)
    # print(onnx.helper.printable_graph(onnx_search.graph))
    #===================================================================================================================

    #===================================================================================================================
    # Build the torch xcorr model
    logging.info("Build the xcorr model")
    xcorr = XCorrBuilder(256, 10, 20)
    xcorr.eval()
    xcorr.state_dict().keys()
    xcorr_dict = xcorr.state_dict()

    # Load the pre-trained weights to the rpn_head model
    # pretrained_dict_head = {k: v for k, v in pretrained_dict_head.items() if k in rpn_head_dict}
    for k, v in pretrained_dict.items():
        if k.startswith('rpn_head.rpn2.cls.head'):
            tmp = k.split('rpn_head.rpn2.cls.head')
            pretrained_dict_xcorr['head_cls2'+tmp[-1]] = v
        if k.startswith('rpn_head.rpn2.loc.head'):
            tmp = k.split('rpn_head.rpn2.loc.head')
            pretrained_dict_xcorr['head_loc2'+tmp[-1]] = v
        if k.startswith('rpn_head.rpn3.cls.head'):
            tmp = k.split('rpn_head.rpn3.cls.head')
            pretrained_dict_xcorr['head_cls3'+tmp[-1]] = v
        if k.startswith('rpn_head.rpn3.loc.head'):
            tmp = k.split('rpn_head.rpn3.loc.head')
            pretrained_dict_xcorr['head_loc3'+tmp[-1]] = v
        if k.startswith('rpn_head.rpn4.cls.head'):
            tmp = k.split('rpn_head.rpn4.cls.head')
            pretrained_dict_xcorr['head_cls4'+tmp[-1]] = v
        if k.startswith('rpn_head.rpn4.loc.head'):
            tmp = k.split('rpn_head.rpn4.loc.head')
            pretrained_dict_xcorr['head_loc4'+tmp[-1]] = v
    # pretrained_dict_head.keys()
    xcorr_dict.update(pretrained_dict_xcorr)
    xcorr.load_state_dict(xcorr_dict)
    # xcorr.cuda()

    search_s = np.stack([search[0].detach().numpy(), search[1].detach().numpy(), search[2].detach().numpy(), search[3].detach().numpy(), search[4].detach().numpy(), search[5].detach().numpy()])
    # # Export the torch rpn_head model to ONNX model
    logging.info("Export the xcorr model ...")
    torch.onnx.export(xcorr, (torch.Tensor(np.random.rand(*search_s.shape))), "xcorr4refit.onnx", export_params=True, input_names = ['input'], output_names = ['cls2', 'cls3', 'cls4', 'loc2', 'loc3', 'loc4'])
    logging.info("Export the xcorr model DONE.")

    # Check whether the rpn_head model has been successfully imported
    # onnx.checker.check_model(onnx_rpn_head_model)
    # print(onnx.checker.check_model(onnx_rpn_head_model))    
    # onnx.helper.printable_graph(onnx_rpn_head_model.graph)
    # print(onnx.helper.printable_graph(onnx_rpn_head_model.graph))
    #===================================================================================================================

    logging.info("DONE.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()