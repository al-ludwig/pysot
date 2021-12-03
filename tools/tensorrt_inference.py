import argparse
import os
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch
# /usr/lib/python3.6/dist-packages/tensorrt/
# sys.path.insert(0, '/usr/lib/python3.6/dist-packages/tensorrt')
# sys.path.insert(0, '/usr/lib/python3.6/dist-packages/tensorrt-7.1.3.0.dist-info/')
import tensorrt as trt
import logging

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
required.add_argument('--target_net', required=True, type=file_path, help="Path to the target net file (.onnx)")
required.add_argument('--search_net', required=True, type=file_path, help="Path to the search net file (.onnx)")
required.add_argument('--xcorr', required=True, type=file_path, help="Path to the xcorr net file (.onnx)")
required.add_argument('--dataset', required=True, help="Name of the testing dataset")
optional.add_argument('--video', default='', help="test one special video")
required.add_argument('--config', default='', type=file_path, help='path to the config file')
required.add_argument('--build', default='n', help='(y) ... build and save the engines; (n) ... load saved engines')
args = parser.parse_args()

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

runtime = trt.Runtime(TRT_LOGGER)

def GiB(val):
    return val * 1 << 30

def to_numpy(tensor):
    return np.ascontiguousarray(tensor.detach().cpu().numpy()) if tensor.requires_grad else np.ascontiguousarray(tensor.cpu().numpy())

def get_engine(model_file, refittable: bool = False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    file_path = str(model_file).rsplit('.', 1)[0] + ".engine"
    def build_engine():
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        config.max_workspace_size = GiB(1)
        if(refittable):
            config.set_flag(trt.BuilderFlag.REFIT)
            config.set_flag(trt.BuilderFlag.FP16)
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
    def __init__(self, target_net, search_net, xcorr):
        self.engine_target = get_engine(target_net, refittable=False)
        self.engine_search = get_engine(search_net, refittable=False)
        self.engine_xcorr = get_engine(xcorr, refittable=True)

        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()

        self.engine_target_context =self.engine_target.create_execution_context()
        self.engine_search_context =self.engine_search.create_execution_context()
        self.engine_xcorr_context = self.engine_xcorr.create_execution_context()

        # placeholder for kernels (output of target_net) = weights of engine_xcorr
        self.kernels = None
        # placeholder for search features
        self.searchf = None
        # placeholder for xcorr results
        self.xcorr = None

        self.target_inputs, self.target_outputs, self.target_bindings, self.target_stream = common.allocate_buffers(self.engine_target)
        self.search_inputs, self.search_outputs, self.search_bindings, self.search_stream = common.allocate_buffers(self.engine_search)
        self.xcorr_inputs, self.xcorr_outputs, self.xcorr_bindings, self.xcorr_stream = common.allocate_buffers(self.engine_xcorr)

        # warm up engines:
        z_crop_ini = np.zeros((1, 3, 127, 127), dtype=np.float16)
        self.warmup_engine('target', z_crop_ini)
        x_crop_ini = np.zeros((1, 3, 255, 255), dtype=np.float16)
        self.warmup_engine('search', x_crop_ini)
        y_crop_ini = np.zeros((6, 256, 29, 29), dtype=np.float16)
    
    def cuda_cleanup(self):
        self.ctx.pop()
    
    def warmup_engine(self, name, ini_crop):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        if name == 'target':
            np.copyto(self.target_inputs[0].host, np.ascontiguousarray(ini_crop).ravel())
            self.kernels = common.do_inference(self.engine_target_context, bindings=self.target_bindings, inputs=self.target_inputs, outputs=self.target_outputs, stream=self.target_stream)
        elif name == 'search':
            np.copyto(self.search_inputs[0].host, np.ascontiguousarray(ini_crop).ravel())
            self.searchf = common.do_inference(self.engine_search_context, bindings=self.search_bindings, inputs=self.search_inputs, outputs=self.search_outputs, stream=self.search_stream)
        elif name == 'xcorr':
            np.copyto(self.xcorr_inputs[0].host, np.ascontiguousarray(ini_crop).ravel())
            self.xcorr = common.do_inference(self.engine_xcorr_context, bindings=self.xcorr_bindings, inputs=self.xcorr_inputs, outputs=self.xcorr_outputs, stream=self.xcorr_stream)

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
    
    def template(self, z_crop):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # np.copyto(self.target_inputs[0].host, np.ascontiguousarray(to_numpy(z_crop)).ravel())
        np.copyto(self.target_inputs[0].host, z_crop)

        tic = cv2.getTickCount()
        self.kernels = common.do_inference(self.engine_target_context, bindings=self.target_bindings, inputs=self.target_inputs, outputs=self.target_outputs, stream=self.target_stream)
        t_infer_kernels = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for inference of template (creating kernels) (s): " + str(t_infer_kernels))

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        output_shape = (1, 256, 5, 5)
        xcorr_kernel_shape = (256, 1, 5, 5)
        self.kernels = [output.reshape(xcorr_kernel_shape) for output in self.kernels]

        # refit engine_xcorr weights with kernels
        logging.debug("Refitting xcorr-engine ...")
        refitter = trt.Refitter(self.engine_xcorr, TRT_LOGGER)
        # weights_names = refitter.get_all()
        # dw_xcorr_cls2 = 'Conv_2'
        # dw_xcorr_cls3 = 'Conv_28'
        # dw_xcorr_cls4 = 'Conv_54'
        # dw_xcorr_loc2 = 'Conv_15'
        # dw_xcorr_loc3 = 'Conv_41'
        # dw_xcorr_loc4 = 'Conv_67'
        conv_names = ['Conv_2', 'Conv_28', 'Conv_54', 'Conv_15', 'Conv_41', 'Conv_67']
        for name, kernel in zip(conv_names, self.kernels):
            refitter.set_weights(name, trt.WeightsRole.KERNEL, kernel)
        logging.debug("Refitting xcorr-engine done.")

    
    def track(self, x_crop):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # np.copyto(self.search_inputs[0].host, np.ascontiguousarray(to_numpy(x_crop)).ravel())
        np.copyto(self.search_inputs[0].host, x_crop)

        tic = cv2.getTickCount()
        self.searchf = common.do_inference(self.engine_search_context, bindings=self.search_bindings, inputs=self.search_inputs, outputs=self.search_outputs, stream=self.search_stream)
        t_infer_searchf = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for inference of search features (s): " + str(t_infer_searchf))

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # output_shape = (1, 256, 29, 29)
        # self.searchf = [output.reshape(output_shape) for output in self.searchf]

        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # print(np.stack(self.searchf, axis=0).flatten().shape)
        np.copyto(self.xcorr_inputs[0].host, np.stack(self.searchf, axis=0).flatten())

        tic = cv2.getTickCount()
        self.xcorr = common.do_inference(self.engine_xcorr_context, bindings=self.xcorr_bindings, inputs=self.xcorr_inputs, outputs=self.xcorr_outputs, stream=self.xcorr_stream)
        t_infer_xcorr = (cv2.getTickCount() - tic)/(cv2.getTickFrequency())
        logging.debug("Time for inference of xcorr (s): " + str(t_infer_xcorr))

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

        cls_shape = (1, 10, 25, 25)
        loc_shape = (1, 20, 25, 25)
        self.xcorr[0] = self.xcorr[0].reshape(cls_shape)
        self.xcorr[1] = self.xcorr[1].reshape(loc_shape)


class SiamRPNTracker:
    def __init__(self, trtmodel):
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
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
        delta = np.transpose(delta, (1, 2, 3, 0))
        delta = np.reshape(delta, (4, -1))
        # delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        # delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta
    
    def softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def _convert_score(self, score):
        score = score.transpose(1, 2, 3, 0)
        score = np.reshape(score, (2, -1))
        score = np.transpose(score, (1, 0))
        # score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = self.softmax(score, axis=1)[:, 1]
        # score = torch.nn.functional.softmax(score, dim=1).data[:, 1].cpu().numpy()
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

        # calculate channel average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        tic = cv2.getTickCount()
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        t_get_zcrop = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for get z_crop: " + str(t_get_zcrop))
        
        # run inference of template image (z_crop)
        # self.model.template(z_crop)
        self.trtmodel.template(z_crop)

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
        self.trtmodel.track(x_crop)

        # self.xcorr[0] = cls
        # self.xcorr[1] = loc
        score = self._convert_score(self.trtmodel.xcorr[0])
        pred_bbox = self._convert_bbox(self.trtmodel.xcorr[1], self.anchors)

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



def main():
    logging.info("START OF SCRIPT")
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
    trtmodel = TrtModel(args.target_net, args.search_net, args.xcorr)
    t_load_engines = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
    logging.debug("Time for loading/creating trt engines (s): " + str(t_load_engines))

    tracker = SiamRPNTracker(trtmodel)

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
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
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
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
            toc /= cv2.getTickFrequency()

            # save results
            video_path = os.path.join('results', args.dataset, 'trt_model',
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
            print('({:3d}) Video: {:12s} Time: {:2.4f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))

    logging.info("FPS: " + str(175/toc))
    trtmodel.cuda_cleanup()
    print("\nDone.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
