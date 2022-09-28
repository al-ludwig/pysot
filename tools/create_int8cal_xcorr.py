import argparse
import logging
import sys
import os
from datetime import datetime
import cv2
import tensorrt as trt
import numpy as np

valid_log_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
file_dir = os.path.dirname(os.path.realpath(__file__))

# argparse check function
def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")

# argument parsing
parser = argparse.ArgumentParser(description='Script for creating xcorr calibration data out of search engine.')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument('--log', default='info', help='Set the logging level (' + str(valid_log_levels) + ')')
optional.add_argument('--warmUp', default=5, type=int, help='Specify the number of warm-up runs per engine (default=5)')
required.add_argument('--target_net', required=True, type=file_path, help="Path to the target net ().onnx or -engine)")
required.add_argument('--search_net', required=True, type=file_path, help="Path to the search net (.onnx or .engine)")
required.add_argument('--xcorr', required=True, type=file_path, help="Path to the xcorr net (.onnx or .engine)")
required.add_argument('--num', required=True, help="Number of pictures")
required.add_argument('--dataset', required=True, help="Name of the testing dataset")
required.add_argument('--config', default='', type=file_path, help='path to the config file')
args = parser.parse_args()

from pysot.core.config import cfg
from toolkit.datasets import DatasetFactory
from pysot.tensorrt.TrtModel import TrtModel
from pysot.tensorrt.TrtSiamRPNTracker import TrtSiamRPNTracker
from pysot.tensorrt.TrtSiamRPNLTTracker import TrtSiamRPNLTTracker
from pysot.utils.bbox import get_axis_aligned_bbox

output_name = args.dataset + "_int8cal_xcorr"
output_directory = os.path.join(file_dir, '../testing_dataset', output_name)

def main():
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

    dataset_root = os.path.join(file_dir, '../testing_dataset', args.dataset)

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
        trtmodel = TrtModel(TRT_LOGGER, runtime, args.target_net, 'fp32', args.search_net, 'fp32', args.xcorr, 'fp32', args.warmUp, calibration_path="")
    except Exception as e:
        logging.error(e)
        return
    t_load_engines = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
    logging.debug("Time for loading/creating trt engines (s): " + str(t_load_engines))

    if 'SiamRPNTracker' in cfg.TRACK.TYPE:
        tracker = TrtSiamRPNTracker(trtmodel)
    elif 'SiamRPNLTTracker' in cfg.TRACK.TYPE:
        tracker = TrtSiamRPNLTTracker(trtmodel)
    
    numImagesOverall = 0
    for v_idx, video in enumerate(dataset):
        numImagesOverall += len(video.img_names)
    logging.debug("Overall images = " + str(numImagesOverall))

    # just divide, dont mind if one video has more images than other videos
    # and round up
    numImgPerVideo = int(numImagesOverall / int(args.num)) + (numImagesOverall % int(args.num) > 0)
    logging.debug("Number of images taken from each video = " + str(numImgPerVideo))

    for v_idx, video in enumerate(dataset):
        logging.info("@" + str(video.name))
        frame_counter = 0
        save_counter = 0
        for idx, (img, gt_bbox) in enumerate(video):
            if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
            if idx == frame_counter:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
            elif idx > frame_counter:
                outputs = tracker.track(img)
                # searchf[0] <-> cls2
                # searchf[1] <-> loc2
                # searchf[2] <-> cls3
                # searchf[3] <-> loc3
                # searchf[4] <-> cls4
                # searchf[5] <-> loc4
                # xcorr sequence:
                # cls2, cls3, cls4, loc2, loc3, loc4
                img_name = video.name + "_" + str(idx+1).zfill(8)
                np.save(os.path.join(output_directory, img_name + "_cls2"), trtmodel.searchf[0])
                np.save(os.path.join(output_directory, img_name + "_cls3"), trtmodel.searchf[2])
                np.save(os.path.join(output_directory, img_name + "_cls4"), trtmodel.searchf[4])
                np.save(os.path.join(output_directory, img_name + "_loc2"), trtmodel.searchf[1])
                np.save(os.path.join(output_directory, img_name + "_loc3"), trtmodel.searchf[3])
                np.save(os.path.join(output_directory, img_name + "_loc4"), trtmodel.searchf[5])
                save_counter += 1
            if save_counter == numImgPerVideo:
                break
    
    logging.info("Finished.")
    logging.shutdown()


if __name__ == '__main__':
    if args.log.upper() not in valid_log_levels:
        logging.error("Given log level '" + str(args.log) + "' is not valid. Exiting.")
        sys.exit(-1)
    import shutil
    if os.path.isdir(output_directory):
        logging.info("Directory '" + output_directory + " already exists so it get's removed!")
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    now = datetime.now()
    now = now.strftime("%d_%m_%Y_%H_%M")
    logging_path = os.path.join(file_dir, '../testing_dataset', 'create_int8cal_xcorr_LOG_' + now + '.txt')
    logging_level = getattr(logging, args.log.upper())
    log_handlers = [logging.StreamHandler(), logging.FileHandler(logging_path)]
    logging.basicConfig(level=logging_level, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', handlers=log_handlers)
    logging.root.setLevel(args.log.upper())
    main()


