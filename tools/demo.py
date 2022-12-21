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
from pathlib import Path
import time

import sys
sys.path.insert(0, "C:\\Workspace\\pysot")

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
from pysot.utils.bbox import corner2center, cxy_wh_2_rect, IoU, rect_2_cxy_wh,center2corner
from pathlib import Path
from datetime import datetime
import logging

torch.set_num_threads(1)

# argparse check function
def is_txt(path):
    if os.path.isfile(path) and Path(path).suffix == ".txt":
        return path
    else:
        raise argparse.ArgumentTypeError(f"gt file:{path} is not a valid file")

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', required=True, type=str, help='config file')
parser.add_argument('--snapshot', required=True, type=str, help='model name')
parser.add_argument('--video_name', required=True, default='', type=str,
                    help='videos or image files')
parser.add_argument('--logtofile', action='store_true', default=False, help="set if logging output to file")
parser.add_argument('--initbbox', nargs='+', type=int, help='specify the initial bounding box')
parser.add_argument('--gt', type=is_txt, help="path to ground truth (gt) file (.txt)")
parser.add_argument('--vis',  action='store_false', help='whether visualize video (default true)')
args = parser.parse_args()


def get_frames(video_name):
    if video_name.endswith('avi') or \
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
    logger.info(f"Given config: {cfg}")
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

    video_name = Path(args.video_name).stem
    
    logger.info(f"Running demo on video '{video_name}'")
    logger.info("bbox format: x (left corner), y (left corner), width, height")

    first_frame = True
    gt_bboxes = []
    if args.gt:
        with open(Path(args.gt), 'r') as f:
            gt = f.read().splitlines()
        for i in range(len(gt)): gt_bboxes.append(tuple(map(float,gt[i][1:-1].split(', '))))

    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    idx = 0
    confidence = []
    pred_bboxes = []
    iou = []
    fps = []

    try:
        for frame in get_frames(args.video_name):
            log_message = ""
            start_time = time.time()
            if first_frame:
                # if path to ground truth file is given, the init_rect is taken from the first ground truth bounding box
                # if no ground truth file is given, but argument initbbox is set, take init_rect from there
                # if no ground truth file and no initbox argument is given, let the user select the bbox
                if args.gt:
                    # gt comes in format: (x1, y1, x2, y2) but (x1, y1, w, h) is needed for init_rect
                    # (x1, y1, x2, y2) -> corner2center() -> (cx, cy, w, h) -> cxy_wh_2_rect() -> (x1, y1, w, h)
                    cx, cy, w, h = corner2center(gt_bboxes[0])
                    init_rect = tuple(map(int, cxy_wh_2_rect([cx, cy], [w, h])))
                    iou.append(1.0)
                elif args.initbbox:
                    init_rect = tuple(args.initbbox)
                else:
                    try:
                        init_rect = cv2.selectROI(video_name, frame, False, False)
                    except:
                        exit()
                tracker.init(frame, init_rect)
                log_message += f"Initial bounding box: {init_rect}"
                first_frame = False
                pred_bboxes.append(init_rect)
                confidence.append(1.0)
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
                confidence.append(score)
                log_message += f"@ frame {idx:5d}: confidence = {confidence[idx]:.5f}, bbox = {bbox}"
                cv2.putText(frame, f"confidence: {confidence[idx]:.5f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if cfg.TRACK.TYPE == "SiamRPNLTTracker":
                    if score < cfg.TRACK.CONFIDENCE_LOW:
                        longterm_state = True
                    elif score > cfg.TRACK.CONFIDENCE_HIGH:
                        longterm_state = False
                    cv2.putText(frame, "longterm state: " + str(longterm_state), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    log_message += f", longterm_state = {longterm_state}"
                if args.vis:
                    cv2.imshow(video_name, frame)
                    cv2.waitKey(40)
                if args.gt:
                    # IoU needs (x1, y1, x2, y2)
                    # predicted bbox is in format (x1, y1, w, h)
                    # (x1, y1, w, h) -> rect_2_cxy_wh() -> (cx, cy, w, h) -> center2corner() -> (x1, y1, x2, y2)
                    # gt bbox is in format (x1, y1, x2, y2)
                    pred_bboxes.append(bbox)
                    pred_pos, pred_size = rect_2_cxy_wh(pred_bboxes[idx])
                    pred_rect = center2corner([*pred_pos, *pred_size])
                    iou.append(IoU(gt_bboxes[idx], pred_rect))
                    log_message += f", IoU = {iou[idx]:.3f}"
            fps.append(1/(time.time()-start_time))
            log_message += f", fps = {fps[idx]:.1f}"
            logger.info(log_message)
            idx += 1
    except KeyboardInterrupt:
        logger.info(f"Tracking cancelled by user.")

    if args.gt and confidence and iou and fps:
        logger.info(f"Stats: mean confidence = {np.mean(confidence):.3f}, mean IoU = {np.mean(iou):.3f}, mean fps = {np.mean(fps):.1f}")


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
