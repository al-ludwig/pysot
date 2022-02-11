import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch
# /usr/lib/python3.6/dist-packages/tensorrt/
# sys.path.insert(0, '/usr/lib/python3.6/dist-packages/tensorrt')
# sys.path.insert(0, '/usr/lib/python3.6/dist-packages/tensorrt-7.1.3.0.dist-info/')
import tensorrt as trt
import logging
import glob
import shutil
import random

from PIL import Image
import common
import cv2

from pysot.core.config import cfg
from toolkit.datasets import DatasetFactory
from pysot.utils.anchor import Anchors
from pysot.utils.bbox import get_axis_aligned_bbox
from toolkit.utils.region import vot_overlap, vot_float2str


# argparse check function
def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")

# argument parsing
parser = argparse.ArgumentParser(description='Script for running tracker with onnx runtime.')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--search_net', required=True, type=file_path, help="Path to the search net file (.onnx)")
required.add_argument('--dataset', required=True, help="Name of the testing dataset")
required.add_argument('--config', default='', type=file_path, help='path to the config file')
required.add_argument('--precision', default='TF32', help='Specify the precision: TF32, fp16, int8 are supported.')
required.add_argument('--num', required=True, help="Number of pictures")
args = parser.parse_args()

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

runtime = trt.Runtime(TRT_LOGGER)

def GiB(val):
    return val * 1 << 30

def to_numpy(tensor):
    return np.ascontiguousarray(tensor.detach().cpu().numpy()) if tensor.requires_grad else np.ascontiguousarray(tensor.cpu().numpy())

def get_engine(model_file, precision, refittable: bool = False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    file_path = str(model_file).rsplit('.', 1)[0] + ".engine"
    def build_engine():
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        config.max_workspace_size = GiB(1)
            
        config.set_flag(precision)
        if(refittable):
            config.set_flag(trt.BuilderFlag.REFIT)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        
        engine = builder.build_engine(network, config)

        with open(file_path, "wb") as f:
	        f.write(engine.serialize())
        return engine

    if os.path.exists(file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(file_path))
        with open(file_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

class TrtModel:
    def __init__(self, search_net, precision):
        if precision == 'fp32' or precision == 'TF32':
            self.precision = trt.BuilderFlag.TF32
            warmup_type = np.float32
        elif precision == 'fp16':
            self.precision = trt.BuilderFlag.FP16
            warmup_type = np.float16
        elif precision == 'int8':
            self.precision = trt.BuilderFlag.INT8
            warmup_type = np.uint8
        else:
            self.precision = ''
            raise ValueError(str(precision) + " is not supported!")
        
        self.engine_search = get_engine(search_net, self.precision, refittable=False)

        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()

        self.engine_search_context =self.engine_search.create_execution_context()

        # placeholder for search features
        self.searchf = None

        self.search_inputs, self.search_outputs, self.search_bindings, self.search_stream = common.allocate_buffers(self.engine_search)

        # warm up engines:
        x_crop_ini = np.zeros((1, 3, 255, 255), dtype=warmup_type)
        self.warmup_engine('search', x_crop_ini)
    
    def cuda_cleanup(self):
        self.ctx.pop()
    
    def warmup_engine(self, name, ini_crop):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        if name == 'search':
            np.copyto(self.search_inputs[0].host, np.ascontiguousarray(ini_crop).ravel())
            self.searchf = common.do_inference(self.engine_search_context, bindings=self.search_bindings, inputs=self.search_inputs, outputs=self.search_outputs, stream=self.search_stream)

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
    
    def track(self, x_crop):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # np.copyto(self.search_inputs[0].host, np.ascontiguousarray(to_numpy(x_crop)).ravel())
        np.copyto(self.search_inputs[0].host, x_crop)

        tic = cv2.getTickCount()
        self.searchf = common.do_inference(self.engine_search_context, bindings=self.search_bindings, inputs=self.search_inputs, outputs=self.search_outputs, stream=self.search_stream)
        t_infer_searchf = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for inference of search features (s): " + str(t_infer_searchf))

        # somehow:
        # searchf[0] <-> cls2
        # searchf[1] <-> loc2
        # searchf[2] <-> cls3
        # searchf[3] <-> loc3
        # searchf[4] <-> cls4
        # searchf[5] <-> loc4

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        output_shape = (1, 256, 29, 29)
        self.searchf = [output.reshape(output_shape) for output in self.searchf]


class SiamRPNTracker:
    def __init__(self, trtmodel):
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.trtmodel = trtmodel # why???
    
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        # im_patch = im_patch.astype(np.float32)
        # im_patch = torch.from_numpy(im_patch)
        # if cfg.CUDA:
            # im_patch = im_patch.cuda()
        im_patch = im_patch.astype(trt.nptype(trt.DataType.HALF)).ravel()
        # return im_patch.astype(trt.nptype(trt.DataType.HALF)).ravel()
        return im_patch
    
    def compute_features(self, img, bbox):
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        # calculate channel average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crops
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        self.trtmodel.track(x_crop)
        return self.trtmodel.searchf


def main():
    logging.info("START OF SCRIPT")
    
    if not args.precision == 'fp32':
        logging.error('Only int8 is supported here.')
        sys.exit(1)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # load config
    cfg.merge_from_file(args.config)

    # create dataset
    tic = cv2.getTickCount()
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    t_load_dataset = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
    logging.debug("Time for loading dataset informations (s): " + str(t_load_dataset))

    tic = cv2.getTickCount()
    trtmodel = TrtModel(args.search_net, args.precision)
    t_load_engines = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
    logging.debug("Time for loading/creating trt engines (s): " + str(t_load_engines))

    tracker = SiamRPNTracker(trtmodel)

    outputDirectory = dataset_root + "_int8cal_xcorr"
    if os.path.exists(outputDirectory):
        shutil.rmtree(outputDirectory)
        logging.error("Directory '" + outputDirectory + " already exists so it get's removed!")
    logging.debug("Making directory: " + str(outputDirectory))
    os.mkdir(outputDirectory)

    numVideosOverall = 0
    for v_idx, video in enumerate(dataset):
        numVideosOverall += len(video.img_names)
    logging.debug("Overall images = " + str(numVideosOverall))

    # just divide, dont mind if one video has more images than other videos
    # and round up
    numImgPerVideo = int(numVideosOverall / int(args.num)) + (numVideosOverall % int(args.num) > 0)
    logging.debug("Number of images taken from each video = " + str(numImgPerVideo))

    searchf = None

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        for v_idx, video in enumerate(dataset):
            logging.debug("@" + str(video.name))
            randomIndexes = random.sample(range(len(video.img_names)), numImgPerVideo)
            for idx, (img, gt_bbox) in enumerate(video):
                if idx in randomIndexes:
                    if len(gt_bbox) == 4:
                        gt_bbox = [gt_bbox[0], gt_bbox[1],
                        gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                        gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                        gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                    
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    searchf = tracker.compute_features(img, gt_bbox_)                    
                    img_name = video.name + "_" + str(idx+1).zfill(8) + ".npy"
                    np.save(os.path.join(outputDirectory, img_name), searchf)

    trtmodel.cuda_cleanup()
    print("\nDone.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
