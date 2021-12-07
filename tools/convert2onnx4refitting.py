import sys
import os
import argparse
import numpy as np
import onnx
import cv2

from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from pysot.models.backbone.resnet_atrous import ResNet, Bottleneck
from pysot.models.head.rpn import RPN
from toolkit.utils.region import vot_overlap, vot_float2str

from toolkit.datasets import DatasetFactory
from pysot.utils.bbox import get_axis_aligned_bbox

import torch
from torch import nn
import torch.nn.functional as F


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
optional.add_argument('--vis', action='store_true', help='whether visualzie result')
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

# class DepthwiseXCorr(nn.Module):
#     "Depthwise Correlation Layer"
#     def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
#         super(DepthwiseXCorr, self).__init__()
#         self.conv_kernel = nn.Sequential(
#                 nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
#                 nn.BatchNorm2d(hidden),
#                 nn.ReLU(inplace=True),
#                 )
#         self.conv_search = nn.Sequential(
#                 nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
#                 nn.BatchNorm2d(hidden),
#                 nn.ReLU(inplace=True),
#                 )
#         self.head = nn.Sequential(
#                 nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(hidden),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(hidden, out_channels, kernel_size=1)
#                 )

#     def forward(self, kernel, search):    
#         kernel = self.conv_kernel(kernel)
#         search = self.conv_search(search)
        
#         feature = xcorr_depthwise(search, kernel)
        
#         out = self.head(feature)
        
#         return out

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

class SiamRPNTracker(SiameseTracker):
    def __init__(self, target_net, search_net, xcorr):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.target_net = target_net
        self.search_net = search_net
        self.xcorr = xcorr
        self.kernels = None
        # self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        # self.model.template(z_crop)

        self.kernels = self.target_net(z_crop)
    
    def init_xcorr(self):
        self.xcorr.init(self.kernels)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # outputs = self.model.track(x_crop)
        xcorr_inputs = self.search_net(x_crop)
        outputs = self.xcorr(xcorr_inputs)
        # np.save("refit_cls.npy", outputs[0].cpu().detach().numpy())

        # score = self._convert_score(outputs['cls'])
        score = self._convert_score(outputs[0])
        # pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        pred_bbox = self._convert_bbox(outputs[1], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

class TargetNetBuilder(nn.Module):
    def __init__(self):
        super(TargetNetBuilder, self).__init__()
        # Build Backbone Model
        self.backbone = ResNet(Bottleneck, [3,4,6,3], [2,3,4])
        # Build Neck Model
        self.neck = AdjustAllLayer_1([512,1024,2048], [256,256,256])
        # Build Corr_prep Target Model
        self.xcorr_prep = XCorrPrepTarget(anchor_num=5, in_channels=[256, 256, 256])
    
    def forward(self, frame):
        features = self.backbone(frame)
        features = self.neck(features)
        output = self.xcorr_prep(features)
        # np.save("refit_kernel_loc2.npy", output[3].cpu().detach().numpy())
        # np.save("features0.npy", features[0].cpu().detach().numpy())
        # np.save("features1.npy", features[1].cpu().detach().numpy())
        # np.save("features2.npy", features[2].cpu().detach().numpy())
        return output

class SearchNetBuilder(nn.Module):
    def __init__(self):
        super(SearchNetBuilder, self).__init__()
        # Build Backbone Model
        self.backbone = ResNet(Bottleneck, [3,4,6,3], [2,3,4])
        # Build Neck Model
        self.neck = AdjustAllLayer_2([512,1024,2048], [256,256,256])
        # Build Corr_prep Search Model
        self.xcorr_prep = XCorrPrepSearch(anchor_num=5, in_channels=[256, 256, 256])
        
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
        # cls_i = cls_i.detach()?
        cls.append(cls_i)
        loc_i = self.dw_xcorr_loc2(search[1])
        loc_i = loc_i.view(1, 256, loc_i.size(2), loc_i.size(3))
        # loc_i = loc_i.detach()?
        loc.append(loc_i)
        cls_i = self.dw_xcorr_cls3(search[2])
        cls_i = cls_i.view(1, 256, cls_i.size(2), cls_i.size(3))
        # cls_i = cls_i.detach()?
        cls.append(cls_i)
        loc_i = self.dw_xcorr_loc3(search[3])
        loc_i = loc_i.view(1, 256, loc_i.size(2), loc_i.size(3))
        # loc_i = loc_i.detach()?
        loc.append(loc_i)
        cls_i = self.dw_xcorr_cls4(search[4])
        cls_i = cls_i.view(1, 256, cls_i.size(2), cls_i.size(3))
        # cls_i = cls_i.detach()?
        cls.append(cls_i)
        loc_i = self.dw_xcorr_loc4(search[5])
        loc_i = loc_i.view(1, 256, loc_i.size(2), loc_i.size(3))
        # loc_i = loc_i.detach()?
        loc.append(loc_i)
        # np.save("refit_weights_cls2", self.dw_xcorr_cls2.weight.cpu().detach().numpy())
        # np.save("refit_weights_cls3", self.dw_xcorr_cls3.weight.cpu().detach().numpy())
        # np.save("refit_weights_cls4", self.dw_xcorr_cls4.weight.cpu().detach().numpy())
        # np.save("refit_weights_loc2", self.dw_xcorr_loc2.weight.cpu().detach().numpy())
        # np.save("refit_weights_loc3", self.dw_xcorr_loc3.weight.cpu().detach().numpy())
        # np.save("refit_weights_loc4", self.dw_xcorr_loc4.weight.cpu().detach().numpy())
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
        # self.weight_cls = nn.Parameter(torch.Tensor([0.38156851768108546, 0.4364767608115956,  0.18195472150731892]))
        # self.weight_loc = nn.Parameter(torch.Tensor([0.17644893463361863, 0.16564198028417967, 0.6579090850822015]))
    
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

    dataset_name = 'VOT2018'

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', dataset_name)

    # load config
    config = '.\\experiments\\siamrpn_r50_l234_dwxcorr\\config.yaml'
    # config = '..\\experiments\\siamrpn_r50_l234_dwxcorr\\config.yaml'
    cfg.merge_from_file(config)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=dataset_name,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # pretrained_dict = torch.load(args.snapshot,map_location=torch.device('cpu') )
    pretrained_dict = torch.load(args.snapshot, map_location=lambda storage, loc: storage.cuda(0))
    pretrained_dict_xcorr = {}#pretrained_dict
    pretrained_dict_target = {}#pretrained_dict
    pretrained_dict_search = {}#pretrained_dict

    # The shape of the inputs to the Target Network and the Search Network
    target = torch.Tensor(np.random.rand(1,3,127,127))
    search = torch.Tensor(np.random.rand(1,3,255,255))

    #===================================================================================================================
    # Build the torch backbone model
    target_net = TargetNetBuilder()
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
    # torch.onnx.export(target_net, torch.Tensor(target), "target_net4refit2.onnx", export_params=True,input_names=['input'], output_names=['kernel_cls2', 'kernel_cls3', 'kernel_cls4', 'kernel_loc2', 'kernel_loc3', 'kernel_loc4'])
    
    # Load the saved torch target net model using ONNX
    # onnx_target = onnx.load("target_net4refit.onnx")

    # Check whether the ONNX target net model has been successfully imported
    # onnx.checker.check_model(onnx_target)
    # print(onnx.checker.check_model(onnx_target))
    # onnx.helper.printable_graph(onnx_target.graph)
    # print(onnx.helper.printable_graph(onnx_target.graph))
    #===================================================================================================================

    #===================================================================================================================
    # Build the torch backbone model
    search_net = SearchNetBuilder()
    search_net.eval()
    search_net.state_dict().keys()
    search_net_dict = search_net.state_dict()

    search = search_net.forward(search)
    tmp_search_net = list(search_net_dict)[250:]

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
    search_net.cuda()

    # Export the torch search net model to ONNX model
    # torch.onnx.export(search_net, torch.Tensor(search), "search_net4refit2.onnx", export_params=True, 
    #               input_names=['input'], output_names=['search_cls2', 'search_cls3', 'search_cls4', 'search_loc2', 'search_loc3', 'search_loc4'])

    # Load the saved torch search net model using ONNX
    # onnx_search = onnx.load("search_net4refit.onnx")

    # Check whether the ONNX target net model has been successfully imported
    # onnx.checker.check_model(onnx_search)
    # print(onnx.checker.check_model(onnx_search))
    # onnx.helper.printable_graph(onnx_search.graph)
    # print(onnx.helper.printable_graph(onnx_search.graph))
    #===================================================================================================================

    #===================================================================================================================
    # Outputs from the Target Net and Search Net
    # zfs_1, zfs_2, zfs_3 = target_net(torch.Tensor(target))
    # xfs_1, xfs_2, xfs_3 = search_net(torch.Tensor(search))

    # Adjustments to the outputs from each of the neck models to match to input shape of the torch rpn_head model
    # zfs = np.stack([zfs_1.detach().numpy(), zfs_2.detach().numpy(), zfs_3.detach().numpy()])
    # xfs = np.stack([xfs_1.detach().numpy(), xfs_2.detach().numpy(), xfs_3.detach().numpy()])
    #===================================================================================================================

    #===================================================================================================================
    # Build the torch xcorr model
    xcorr = XCorrBuilder(256, 10, 20)
    xcorr.eval()
    xcorr.state_dict().keys()
    xcorr_dict = xcorr.state_dict()

    # xcorr.init(kernel)
    # output = xcorr.forward(search)
    # tmp_xcorr = list(xcorr_dict)
    # tmp_pretrained = list(pretrained_dict)[250:]

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
    torch.onnx.export(xcorr, (torch.Tensor(np.random.rand(*search_s.shape))), "rpn_head4refit5.onnx", export_params=True, input_names = ['input'], output_names = ['cls2', 'cls3', 'cls4', 'loc2', 'loc3', 'loc4'])
    
    # Load the saved xcorr model using ONNX
    # onnx_rpn_head_model = onnx.load("rpn_head4refit.onnx")

    # Check whether the rpn_head model has been successfully imported
    # onnx.checker.check_model(onnx_rpn_head_model)
    # print(onnx.checker.check_model(onnx_rpn_head_model))    
    # onnx.helper.printable_graph(onnx_rpn_head_model.graph)
    # print(onnx.helper.printable_graph(onnx_rpn_head_model.graph))
    #===================================================================================================================

    tracker = SiamRPNTracker(target_net, search_net, xcorr)

    videoname = 'ants1'

    for v_idx, video in enumerate(dataset):
        if videoname != '':
                # test one special video
                if video.name != videoname:
                    continue
        frame_counter = 0
        lost_number = 0
        toc = 0
        pred_bboxes = []
        for idx, (img, gt_bbox) in enumerate(video):
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
            tic = cv2.getTickCount()
            if idx == frame_counter:
                # template
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                tracker.init_xcorr()
                pred_bbox = gt_bbox_
                pred_bboxes.append(1)
            elif idx > frame_counter:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                if cfg.MASK.MASK:
                    pred_bbox = outputs['polygon']
                overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                if overlap > 0:
                    # not lost
                    pred_bboxes.append(pred_bbox)
                else:
                    # lost object
                    pred_bboxes.append(2)
                    frame_counter = idx + 5 # skip 5 frames
                    lost_number += 1
            else:
                pred_bboxes.append(0)
            toc += cv2.getTickCount() - tic

            if args.vis and idx > frame_counter:
                cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                        True, (0, 255, 0), 3)
                bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (bbox[0], bbox[1]),
                                (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results
        video_path = os.path.join('results', dataset_name, 'onnx_model',
                'baseline', video.name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, '{}_000.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                if isinstance(x, int):
                    f.write("{:d}\n".format(x))
                else:
                    f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
                    # f.write('test\n')
        print('({:3d}) Video: {:12s} Time: {:2.4f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))

    print("\n\nDONE.\n")

if __name__ == '__main__':
    main()