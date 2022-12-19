from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

import sys
sys.path.insert(0, "C:\\Workspace\\pysot")

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
from pathlib import Path
from datetime import datetime
import logging

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--logtofile', action='store_true', default=False, help="set if logging output to file")
parser.add_argument('--initbbox', nargs='+', type=int, help='specify the initial bounding box')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    longterm_state = False

    # create model
    model = ModelBuilder()

    # load model
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    logger.info(f"Tracker running on {device}")
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = Path(args.video_name).stem
    else:
        video_name = 'webcam'
    logger.info(f"Running demo on video '{video_name}'")
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    idx = 0
    logger.info("bbox format: x (left corner), y (left corner), width, height")
    for frame in get_frames(args.video_name):
        if first_frame:
            if args.initbbox:
                init_rect = tuple(args.initbbox)
            else:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
            tracker.init(frame, init_rect)
            logger.info(f"Initial bounding box: {init_rect}")
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                            (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                            (0, 255, 0), 3)
            score = outputs['best_score']
            log_message = f"@ frame {idx}: confidence = {score}, bbox = {bbox}"
            cv2.putText(frame, "conficence: " + str(score), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            if cfg.TRACK.TYPE == "SiamRPNLTTracker":
                if score < cfg.TRACK.CONFIDENCE_LOW:
                    longterm_state = True
                elif score > cfg.TRACK.CONFIDENCE_HIGH:
                    longterm_state = False
                cv2.putText(frame, "longterm state: " + str(longterm_state), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                log_message = log_message + f", longterm_state = {longterm_state}"
            cv2.imshow(video_name, frame)
            logger.info(log_message)
            cv2.waitKey(40)
        idx += 1


if __name__ == '__main__':
    now = datetime.now()
    now = now.strftime("%d_%m_%Y_%H_%M")

    logging_dir_path = Path("../demo/logs")
    if not os.path.isdir(logging_dir_path):
        os.makedirs(logging_dir_path)

    logger = logging.getLogger("global")
    logger.setLevel(logging.INFO)

    logging_path = Path(logging_dir_path, f"demo_log_{now}.log")
    streamhandler = logging.StreamHandler()
    FileFormatter = logging.Formatter(fmt=f"%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    streamhandler.setFormatter(FileFormatter)
    
    streamhandler.setLevel(logging.INFO)
    if args.logtofile:
        filehandler = logging.FileHandler(logging_path)
        filehandler.setFormatter(FileFormatter)
        filehandler.setLevel(logging.INFO)
        logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info("Running demo")
    logger.info("Given arguments: " + str(args))
    main()
