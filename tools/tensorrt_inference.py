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

from PIL import Image
import common
import cv2

from pysot.core.config import cfg
from toolkit.datasets import DatasetFactory
from pysot.tensorrt.TrtModel import TrtModel
from pysot.tensorrt.TrtSiamRPNTracker import TrtSiamRPNTracker
from pysot.tensorrt.TrtSiamRPNLTTracker import TrtSiamRPNLTTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from toolkit.utils.region import vot_overlap, vot_float2str

debug = False
valid_log_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

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
required.add_argument('--target_net', required=True, type=file_path, help="Path to the target net ().onnx or -engine)")
required.add_argument('--search_net', required=True, type=file_path, help="Path to the search net (.onnx or .engine)")
required.add_argument('--xcorr', required=True, type=file_path, help="Path to the xcorr net (.onnx or .engine)")
optional.add_argument('--target_net_pr', default='fp32', choices=['fp32', 'fp16', 'int8'], help="Set the precision of the target_net engine. Will be ignored when loading an engine directly.")
optional.add_argument('--search_net_pr', default='fp32', choices=['fp32', 'fp16', 'int8'], help="Set the precision of the search_net engine. Will be ignored when loading an engine directly.")
optional.add_argument('--xcorr_pr', default='fp32', choices=['fp32', 'fp16', 'int8'], help="Set the precision of the xcorr engine. Will be ignored when loading an engine directly.")
required.add_argument('--dataset', required=True, help="Name of the testing dataset")
optional.add_argument('--video', default='', help="test one special video")
required.add_argument('--config', default='', type=file_path, help='path to the config file')
optional.add_argument('--warmUp', default=5, type=int, help='Specify the number of warm-up runs per engine (default=5)')
optional.add_argument('--log', default='info', help='Set the logging level (' + str(valid_log_levels) + ')')
args = parser.parse_args()

def main(results_path):
    logging.info("START OF SCRIPT")
    logging.info("Printing version info:")
    versions = {}
    for module in sys.modules:
        try:
            versions[module] = sys.modules[module].__version__
        except:
            try:
                if type(sys.modules[module].version) is str:
                    versions[module] = sys.modules[module].version
                else:
                    versions[module] = sys.modules[module].version()
            except:
                try:
                    versions[module] = sys.modules[module].VERSION
                except:
                    pass
    logging.info(versions)
    logging.info("Given arguments: " + str(args))
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
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)
    try:
        trtmodel = TrtModel(TRT_LOGGER, runtime, args.target_net, args.target_net_pr, args.search_net, args.search_net_pr, args.xcorr, args.xcorr_pr, args.warmUp)
    except Exception as e:
        logging.error(e)
        return
    t_load_engines = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
    logging.debug("Time for loading/creating trt engines (s): " + str(t_load_engines))

    if 'SiamRPNTracker' in cfg.TRACK.TYPE:
        tracker = TrtSiamRPNTracker(trtmodel)
    elif 'SiamRPNLTTracker' in cfg.TRACK.TYPE:
        tracker = TrtSiamRPNLTTracker(trtmodel)

    report_lines = []
    speed = []

    import subprocess
    try:
        tegrastats_loggingpath = os.path.join(results_path, 'tegrastats.log')
        subprocess.Popen(["/usr/bin/tegrastats", "--interval", "100", "--logfile", os.path.abspath(tegrastats_loggingpath), "--start"])
        logging.info("Started tegrastats logging @ {}".format(tegrastats_loggingpath))
    except Exception as e:
        logging.info("Tegrastats is not available on this setup, so no log about device utilization will be made.")
        logging.info(e)

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
            overlaps = []
            for idx, (img, gt_bbox) in enumerate(video):
                logging.debug("")
                logging.debug("@ img "+ str(idx))
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
                    overlaps.append('1\n')
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                        overlaps.append(str(overlap) + '\n')
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        overlaps.append('2\n')
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                    overlaps.append('0\n')
                toc += cv2.getTickCount() - tic
            toc /= cv2.getTickFrequency()

            # save results
            video_path = os.path.join(results_path, video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_000.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            
            overlaps_path = os.path.join(video_path, '{}_overlaps.txt'.format(video.name))
            with open(overlaps_path, 'w') as f:
                for o in overlaps:
                    f.write(o)
            
            report_text = '({:3d}) Video: {:12s} Time: {:2.4f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number)
            logging.info(report_text)
            report_lines.append(report_text)
            speed.append(idx / toc)
    else:
        # OPE tracking (OPE ... one pass evaluation -> no re-init)
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                logging.debug("at img nr " + str(idx))
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()

            # save results
            if 'VOT2019-LT' == args.dataset:
                video_path = os.path.join(results_path, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            
            report_text = '({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc)
            logging.info(report_text)
            report_lines.append(report_text)
            speed.append(idx / toc)

    try:
        subprocess.Popen(["tegrastats", "--stop"])
        logging.info("Stopped tegrastats logging")
    except Exception as e:
        logging.info("Could not stop tegratstas logging.")
        logging.info(e)

    average_speed = sum(speed) / len(speed)
    report_path = os.path.join(results_path, 'inference_report.txt')
    with open(report_path, 'w') as f:
        for line in report_lines:
            f.write(line + '\n')
        f.write("\n\nAverage Speed: {:3.1f}fps".format(average_speed))

    logging.info("End of script.")
    logging.shutdown()


if __name__ == '__main__':
    if args.log.upper() not in valid_log_levels:
        logging.error("Given log level '" + str(args.log) + "' is not valid. Exiting.")
        sys.exit(-1)
    import shutil
     # create results directory
    now = datetime.now()
    now = now.strftime("%d_%m_%Y_%H_%M")
    model_name = "trt_model_" + now    
    results_path = os.path.join('results', args.dataset, model_name)
    if os.path.isdir(results_path):
        shutil.rmtree(results_path)    
    os.makedirs(results_path)
    logging_path = os.path.join(results_path, 'log.txt')
    logging_level = getattr(logging, args.log.upper())
    log_handlers = [logging.StreamHandler(), logging.FileHandler(logging_path)]
    logging.basicConfig(level=logging_level, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', handlers=log_handlers)
    main(results_path)
