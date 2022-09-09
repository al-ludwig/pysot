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
from pysot.utils.anchor import Anchors
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
optional.add_argument('--calibration_path', default='', help='Set the path to the calibration images')
optional.add_argument('--logtofile', action='store_true', default=False, help='save log to file (Bool)')
optional.add_argument('--warmUp', default=5, type=int, help='Specify the number of warm-up runs per engine (default=5)')
optional.add_argument('--log', default='info', help='Set the logging level (' + str(valid_log_levels) + ')')
args = parser.parse_args()

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

runtime = trt.Runtime(TRT_LOGGER)

def GiB(val):
    return val * 1 << 30

def to_numpy(tensor):
    return np.ascontiguousarray(tensor.detach().cpu().numpy()) if tensor.requires_grad else np.ascontiguousarray(tensor.cpu().numpy())

def get_precision_builderflag(arg):
    precision_builderflag = None
    if 'fp32' in arg:
        precision_builderflag = trt.BuilderFlag.TF32
    elif 'fp16' in arg:
        precision_builderflag = trt.BuilderFlag.FP16
    elif 'int8' in arg:
        precision_builderflag = trt.BuilderFlag.INT8
    return precision_builderflag

def get_engine(model_file, precision, refittable: bool = False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        precision_builderflag = get_precision_builderflag(precision)

        config.max_workspace_size = GiB(1)
        if precision_builderflag == trt.BuilderFlag.INT8:
            if not args.calibration_path:
                raise Exception("No calibration path given!.")
            NUM_IMAGES_PER_BATCH = 1
            if 'target' in model_file:
                dir_name = args.dataset + '_int8cal_target'
            elif 'search' in model_file:
                dir_name = args.dataset + '_int8cal_search'
            elif 'xcorr' in model_file:
                dir_name = args.dataset + '_int8cal_xcorr'
            else:
                logging.error("Unsupported name of onnx/enginge models.")
                return 
            calibration_files = get_calibration_files(os.path.join(args.calibration_path, dir_name))
            config.int8_calibrator = ImagenetCalibrator(calibration_files, NUM_IMAGES_PER_BATCH)
            
        config.set_flag(precision_builderflag)
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

        with open(str(model_file).rsplit('.', 1)[0] + "_" + precision + ".engine", "wb") as f:
	        f.write(engine.serialize())
        return engine

    if model_file.endswith('.onnx'):
        # if a engine with the name already exists -> delete it
        # if not: create trt engine
        if os.path.exists(str(model_file).rsplit('.', 1)[0] + ".engine"):
            logging.info("Found an already existing engine: " + str(model_file).rsplit('.', 1)[0] + ".engine")
            logging.info("Deleting this engine ...")
            os.remove(str(model_file).rsplit('.', 1)[0] + ".engine")
            logging.info(str(model_file).rsplit('.', 1)[0] + ".engine" + " has been deleted.")
    
        try:
            logging.info("Creating trt engine " + str(model_file).rsplit('.', 1)[0] + "_" + precision + ".engine")
            return build_engine()
        except Exception as e:
            logging.error("Something wrent wrong while creating the "+ str(model_file).rsplit('.', 1)[0] + ".engine")
            logging.error("Details: " + str(e))
    elif model_file.endswith('.engine'):
        # try to load the engine
        logging.info("Reading engine from file {}:".format(model_file))
        try:
            with open(model_file, "rb") as f:
                return runtime.deserialize_cuda_engine(f.read())
        except:
            logging.error("Something wrent wrong while reading the engine from file {}".format(model_file))
            sys.exit()
    else:
        # no valid file ending (onnx or engine)
        raise ValueError("No valid file ending! Supported file types are .onnx and .engine")

def get_calibration_files(calibration_data, allowed_extensions=(".jpeg", ".jpg", ".png", ".npy")):
    """Returns a list of all filenames ending with `allowed_extensions` found in the `calibration_data` directory.
    Parameters
    ----------
    calibration_data: str
        Path to directory containing desired files.
    Returns
    -------
    calibration_files: List[str]
         List of filenames contained in the `calibration_data` directory ending with `allowed_extensions`.
    """

    logging.info("Collecting calibration files from: {:}".format(calibration_data))
    calibration_files = [path for path in glob.iglob(os.path.join(calibration_data, "**"), recursive=True)
                         if os.path.isfile(path) and path.lower().endswith(allowed_extensions)]
    logging.info("Number of Calibration Files found: {:}".format(len(calibration_files)))

    if len(calibration_files) == 0:
        raise Exception("Calibration data path [{:}] contains no files!".format(calibration_data))

    return calibration_files

# https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html
class ImagenetCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator Class for Imagenet-based Image Classification Models.
    Parameters
    ----------
    calibration_files: List[str]
        List of image filenames to use for INT8 Calibration
    batch_size: int
        Number of images to pass through in one batch during calibration
    input_shape: Tuple[int]
        Tuple of integers defining the shape of input to the model (Default: (3, 224, 224))
    cache_file: str
        Name of file to read/write calibration cache from/to.
    preprocess_func: function -> numpy.ndarray
        Pre-processing function to run on calibration data. This should match the pre-processing
        done at inference time. In general, this function should return a numpy array of
        shape `input_shape`.
    """

    def __init__(self, calibration_files=[], batch_size=32, input_shape=(1, 3, 127, 127),
                 cache_file="calibration.cache", preprocess_func=None):
        # super().__init__()
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.batch = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.batch.nbytes)

        self.files = calibration_files
        # Pad the list so it is a multiple of batch_size
        if len(self.files) % self.batch_size != 0:
            print("Padding # calibration files to be a multiple of batch_size {:}".format(self.batch_size))
            self.files += calibration_files[(len(calibration_files) % self.batch_size):self.batch_size]

        self.batches = self.load_batches()

        if preprocess_func is None:
            print("No preprocess_func defined! Please provide one to the constructor.")
        else:
            self.preprocess_func = preprocess_func

    def load_batches(self):
        # Populates a persistent self.batch buffer with images.
        for index in range(0, len(self.files), self.batch_size):
            for offset in range(self.batch_size):
                ext = os.path.splitext(self.files[index + offset])[-1].lower()
                if ext == '.jpg' or ext == '.jpeg':
                    image = Image.open(self.files[index + offset])
                    image = np.array(image)
                elif ext == '.npy':
                    image = np.load(self.files[index + offset])
                else:
                    logging.error("File type of calibration images is not supported! Given filetype: " + str(ext))
                    return None
                
                image = image.transpose(2, 0, 1)
                image = image[np.newaxis, :, :, :]
                image = image.astype(np.float32)
                self.batch[offset] = image
                # self.batch[offset] = self.preprocess_func(image, *self.input_shape)
            logging.info("Calibration images pre-processed: {:}/{:}".format(index+self.batch_size, len(self.files)))
            yield self.batch

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            batch = next(self.batches)
            # Assume that self.device_input is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            print("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)

class TrtModel:
    def __init__(self, target_net, search_net, xcorr):      
        logging.info("Creating tensorrt engines ...")
        try:
            self.engine_target = get_engine(target_net, args.target_net_pr, refittable=False)
            self.engine_search = get_engine(search_net, args.search_net_pr, refittable=False)
            self.engine_xcorr = get_engine(xcorr, args.xcorr_pr, refittable=True)
        except Exception as e:
            raise Exception("Something went wrong when getting the engines.\nDetails: " + str(e))
        
        if self.engine_target == None:
            raise Exception("Something went wrong when getting the target-engine.\nPlease look in the trt-log for more details.")
        if self.engine_search == None:
            raise Exception("Something went wrong when getting the search-engine.\nPlease look in the trt-log for more details.")
        if self.engine_xcorr == None:
            raise Exception("Something went wrong when getting the xcorr_engine.\nPlease look in the trt-log for more details.")
        logging.info("Creating tensorrt engines successfully completed.")


        self.engine_target_context =self.engine_target.create_execution_context()
        self.engine_search_context =self.engine_search.create_execution_context()
        self.engine_xcorr_context = self.engine_xcorr.create_execution_context()

        # placeholder for kernels (output of target_net) = weights of engine_xcorr
        self.kernels = None
        # placeholder for search features
        self.searchf = None
        # placeholder for xcorr results
        self.xcorr = None

        self.target_host_inputs, self.target_cuda_inputs, self.target_host_outputs, self.target_cuda_outputs, self.target_bindings, self.target_stream = common.allocate_buffers(self.engine_target)
        self.search_host_inputs, self.search_cuda_inputs, self.search_host_outputs, self.search_cuda_outputs, self.search_bindings, self.search_stream = common.allocate_buffers(self.engine_search)
        self.xcorr_host_inputs, self.xcorr_cuda_inputs, self.xcorr_host_outputs, self.xcorr_cuda_outputs, self.xcorr_bindings, self.xcorr_stream = common.allocate_buffers(self.engine_xcorr)

        # warm up engines:
        logging.info("Warming up target engine ...")
        z_crop_ini = np.zeros((1, 3, 127, 127), dtype=np.float32)
        self.warmup_engine('target', z_crop_ini)

        logging.info("Warming up search engine ...")
        x_crop_ini = np.zeros((1, 3, 255, 255), dtype=np.float32)
        self.warmup_engine('search', x_crop_ini)

        logging.info("Warming up xcorr engine ...")
        # xcorr sequence is:
        # cls2, cls3, cls4, loc2, loc3, loc4
        y_crop_ini = [np.zeros((1, int(cfg.RPN.KWARGS.in_channels[0]), 29, 29), dtype=np.float32), np.zeros((1, int(cfg.RPN.KWARGS.in_channels[1]), 29, 29), dtype=np.float32), np.zeros((1, int(cfg.RPN.KWARGS.in_channels[2]), 29, 29), dtype=np.float32)]
        self.warmup_engine('xcorr', y_crop_ini)
        logging.info("warmup engines (" + str(args.warmUp) + " runs each) finished")
    
    def warmup_engine(self, name, ini_crop):
        if name == 'target':
            np.copyto(self.target_host_inputs[0], ini_crop.ravel())
            for i in range(args.warmUp):
                self.warmup_output = common.do_inference(self.engine_target_context, self.target_bindings, self.target_host_inputs, self.target_cuda_inputs, self.target_host_outputs, self.target_cuda_outputs, self.target_stream)
        elif name == 'search':
            np.copyto(self.search_host_inputs[0], ini_crop.ravel())
            for i in range(args.warmUp):
                self.searchf = common.do_inference(self.engine_search_context, self.search_bindings, self.search_host_inputs, self.search_cuda_inputs, self.search_host_outputs, self.search_cuda_outputs, self.search_stream)
        elif name == 'xcorr':
            for i in range(len(self.xcorr_host_inputs)):
                np.copyto(self.xcorr_host_inputs[i], ini_crop[i%len(ini_crop)].ravel())
                for i in range(args.warmUp):
                    self.xcorr = common.do_inference(self.engine_xcorr_context, self.xcorr_bindings, self.xcorr_host_inputs, self.xcorr_cuda_inputs, self.xcorr_host_outputs, self.xcorr_cuda_outputs, self.xcorr_stream)
    
    def template(self, z_crop):
        np.copyto(self.target_host_inputs[0], z_crop.ravel())

        tic = cv2.getTickCount()
        self.kernels = common.do_inference(self.engine_target_context, self.target_bindings, self.target_host_inputs, self.target_cuda_inputs, self.target_host_outputs, self.target_cuda_outputs, self.target_stream)
        t_infer_kernels = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for inference of template (creating kernels) (s): " + str(t_infer_kernels))

        # somehow:
        # kernels[0] <-> cls2
        # kernels[1] <-> loc2
        # kernels[2] <-> cls3
        # kernels[3] <-> loc3
        # kernels[4] <-> cls4
        # kernels[5] <-> loc4

        for i in range(len(self.kernels)):
            self.kernels[i] = self.kernels[i].reshape(self.engine_target.get_binding_shape(i+1))

        if debug:
            np.save("z_cls2.npy", self.kernels[0])
            np.save("z_loc2.npy", self.kernels[1])
            np.save("z_cls3.npy", self.kernels[2])
            np.save("z_loc3.npy", self.kernels[3])
            np.save("z_cls4.npy", self.kernels[4])
            np.save("z_loc4.npy", self.kernels[5])
        
        # refit engine_xcorr weights with kernels
        logging.debug("Refitting xcorr-engine ...")
        refitter = trt.Refitter(self.engine_xcorr, TRT_LOGGER)

        logging.debug("Refittable weights:")
        refittable_weights = refitter.get_all_weights()
        logging.debug(refittable_weights)
        conv_names = ['dwxcorr.dw_xcorr_cls2.weight', 'dwxcorr.dw_xcorr_loc2.weight', 'dwxcorr.dw_xcorr_cls3.weight', 'dwxcorr.dw_xcorr_loc3.weight', 'dwxcorr.dw_xcorr_cls4.weight', 'dwxcorr.dw_xcorr_loc4.weight']
        tmp = [i for i in refittable_weights if i in conv_names]

        if len(tmp) != len(conv_names):
            # older opset version may not use 'names' for the layers
            # dw_xcorr_cls2 = '1', dw_xcorr_cls3 = '3', dw_xcorr_cls4 = '5'
            # dw_xcorr_loc2 = '2', dw_xcorr_loc3 = '4', dw_xcorr_loc4 = '6'
            conv_names = ['1', '2', '3', '4', '5', '6']
        
        for name, kernel in zip(conv_names, self.kernels):
            refitter.set_named_weights(name, kernel)
        missing_weights = refitter.get_missing_weights()
        assert len(missing_weights) == 0, "Refitter found missing weights. Call set_named_weights() for all missing weights"
        refitter.refit_cuda_engine()
        logging.debug("Refitting xcorr-engine done.")
    
    def track(self, x_crop):
        tic = cv2.getTickCount()
        logging.debug("shape of x_crop: " + str(x_crop.shape))
        np.copyto(self.search_host_inputs[0], x_crop.ravel())
        t_copyinput = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for loading input (s): " + str(t_copyinput))

        tic = cv2.getTickCount()
        self.searchf = common.do_inference(self.engine_search_context, self.search_bindings, self.search_host_inputs, self.search_cuda_inputs, self.search_host_outputs, self.search_cuda_outputs, self.search_stream)
        t_infer_searchf = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for inference of search features (s): " + str(t_infer_searchf))

        for i in range(len(self.searchf)):
            self.searchf[i] = self.searchf[i].reshape(self.engine_search.get_binding_shape(i+1))

        # somehow:
        # searchf[0] <-> cls2
        # searchf[1] <-> loc2
        # searchf[2] <-> cls3
        # searchf[3] <-> loc3
        # searchf[4] <-> cls4
        # searchf[5] <-> loc4

        if debug:
            output_shape = (1, 256, 29, 29)
            xcorr_search_shape = [(1, int(cfg.RPN.KWARGS.in_channels[0]), 29, 29), (1, int(cfg.RPN.KWARGS.in_channels[1]), 29, 29), (1, int(cfg.RPN.KWARGS.in_channels[2]), 29, 29)]
            xcorr_search_indexes = [0, 0, 1, 1, 2, 2]
            searchf = []
            for i in range(len(self.searchf)):
                searchf.append(self.searchf[i].reshape(xcorr_search_shape[xcorr_search_indexes[i]]))
            # searchf = [output.reshape(output_shape) for output in self.searchf]
            np.save("x_cls2.npy", searchf[0])
            np.save("x_loc2.npy", searchf[1])
            np.save("x_cls3.npy", searchf[2])
            np.save("x_loc3.npy", searchf[3])
            np.save("x_cls4.npy", searchf[4])
            np.save("x_loc4.npy", searchf[5])

        # xcorr sequence is:
        # cls2, cls3, cls4, loc2, loc3, loc4

        tic = cv2.getTickCount()
        # flatten vs ravel vs reshape(-1)??
        # https://stackoverflow.com/questions/28930465/what-is-the-difference-between-flatten-and-ravel-functions-in-numpy
        np.copyto(self.xcorr_host_inputs[0], self.searchf[0].ravel())
        np.copyto(self.xcorr_host_inputs[1], self.searchf[2].ravel())
        np.copyto(self.xcorr_host_inputs[2], self.searchf[4].ravel())
        np.copyto(self.xcorr_host_inputs[3], self.searchf[1].ravel())
        np.copyto(self.xcorr_host_inputs[4], self.searchf[3].ravel())
        np.copyto(self.xcorr_host_inputs[5], self.searchf[5].ravel())
        t_copysearchf = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for copying searchf (s): " + str(t_copysearchf))

        tic = cv2.getTickCount()
        self.xcorr = common.do_inference(self.engine_xcorr_context, self.xcorr_bindings, self.xcorr_host_inputs, self.xcorr_cuda_inputs, self.xcorr_host_outputs, self.xcorr_cuda_outputs, self.xcorr_stream)
        t_infer_xcorr = (cv2.getTickCount() - tic)/(cv2.getTickFrequency())
        logging.debug("Time for inference of xcorr (s): " + str(t_infer_xcorr))

        if debug:
            np.save("xcorr_cls2.npy", self.xcorr[0])
            np.save("xcorr_cls3.npy", self.xcorr[1])
            np.save("xcorr_cls4.npy", self.xcorr[2])
            np.save("xcorr_loc2.npy", self.xcorr[3])
            np.save("xcorr_loc3.npy", self.xcorr[4])
            np.save("xcorr_loc4.npy", self.xcorr[5])

        tic = cv2.getTickCount()
        # weights:
        # cls2: 0.3816
        # cls3: 0.4365
        # cls4: 0.1820
        # loc2: 0.1764
        # loc3: 0.1656
        # loc4: 0.6579

        # self.xcorr[0] = self.xcorr[0] * 0.381
        # self.xcorr[1] = self.xcorr[1] * 0.436
        # self.xcorr[2] = self.xcorr[2] * 0.183
        # self.xcorr[3] = self.xcorr[3] * 0.176
        # self.xcorr[4] = self.xcorr[4] * 0.165
        # self.xcorr[5] = self.xcorr[5] * 0.659

        for i in range(len(self.xcorr)):
            self.xcorr[i] = self.xcorr[i].reshape(self.engine_xcorr.get_binding_shape(i+6))

        # LT
        self.xcorr[0] = self.xcorr[0] * 0.3492
        self.xcorr[1] = self.xcorr[1] * 1.4502
        self.xcorr[2] = self.xcorr[2] * 0.8729
        self.xcorr[3] = self.xcorr[3] * 0.7019
        self.xcorr[4] = self.xcorr[4] * 0.6217
        self.xcorr[5] = self.xcorr[5] * 1.3487

        self.cls = self.xcorr[0] + self.xcorr[1] + self.xcorr[2]
        self.loc = self.xcorr[3] + self.xcorr[4] + self.xcorr[5]

        t_multiplyreshape = (cv2.getTickCount() - tic)/(cv2.getTickFrequency())
        logging.debug("Time for multiplying and reshaping (s): " + str(t_multiplyreshape))


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
        logging.debug("went into normal tracking")
        tic = cv2.getTickCount()
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        t_getwindow = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for get subwindow (s): " + str(t_getwindow))
        # outputs = self.model.track(x_crop)
        self.trtmodel.track(x_crop)
        t_afterinf = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time after trt inference (s): " + str(t_afterinf))

        # self.xcorr[0] = cls
        # self.xcorr[1] = loc
        score = self._convert_score(self.trtmodel.cls)
        pred_bbox = self._convert_bbox(self.trtmodel.loc, self.anchors)

        t_afterconv = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time after score and box conversion (s): " + str(t_afterconv))

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
        t_track = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for tracker.track() (s): " + str(t_track))
        return {
                'bbox': bbox,
                'best_score': best_score
               }

class SiamRPNLTTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamRPNLTTracker, self).__init__(model)
        self.longterm_state = False

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        logging.debug("went into lt tracking ...")
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        score_size = (instance_size - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)

        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)
        
        self.trtmodel.track(x_crop)

        # self.xcorr[0] = cls
        # self.xcorr[1] = loc
        score = self._convert_score(self.trtmodel.cls)
        pred_bbox = self._convert_bbox(self.trtmodel.loc, anchors)

        # outputs = self.model.track(x_crop)
        # score = self._convert_score(outputs['cls'])
        # pred_bbox = self._convert_bbox(outputs['loc'], anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                    window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]

            width = self.size[0]
            height = self.size[1]

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if best_score < cfg.TRACK.CONFIDENCE_LOW:
            self.longterm_state = True
        elif best_score > cfg.TRACK.CONFIDENCE_HIGH:
            self.longterm_state = False
        logging.debug("longterm_state = " + str(self.longterm_state))
        return {
                'bbox': bbox,
                'best_score': best_score
               }


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
    try:
        trtmodel = TrtModel(args.target_net, args.search_net, args.xcorr)
    except Exception as e:
        logging.error(e)
        return
    t_load_engines = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
    logging.debug("Time for loading/creating trt engines (s): " + str(t_load_engines))

    if 'SiamRPNTracker' in cfg.TRACK.TYPE:
        tracker = SiamRPNTracker(trtmodel)
    elif 'SiamRPNLTTracker' in cfg.TRACK.TYPE:
        tracker = SiamRPNLTTracker(trtmodel)

    report_lines = []
    speed = []

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
            
            overlaps_path = os.path.join(video_path, '{}_overlaps.txt'.format(video.name))
            with open(overlaps_path, 'w') as f:
                for o in overlaps:
                    f.write(o)
            
            report_text = '({:3d}) Video: {:12s} Time: {:2.4f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number)
            # print('({:3d}) Video: {:12s} Time: {:2.4f}s Speed: {:3.1f}fps Lost: {:d}'.format(
            #         v_idx+1, video.name, toc, idx / toc, lost_number))
            logging.info(report_text)
            report_lines.append(report_text)
            speed.append(idx / toc)
        
        average_speed = sum(speed) / len(speed)
        report_path = os.path.join('results', args.dataset, 'trt_model', 'baseline', 'inference_report.txt')
        with open(report_path, 'w') as f:
            for line in report_lines:
                f.write(line + '\n')
            f.write("\n\nAverage Speed: {:3.1f}fps".format(average_speed))

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
                video_path = os.path.join('results', args.dataset, 'trt_model',
                        'longterm', video.name)
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
        
        average_speed = sum(speed) / len(speed)
        report_path = os.path.join('results', args.dataset, 'trt_model', 'longterm', 'inference_report.txt')
        with open(report_path, 'w') as f:
            for line in report_lines:
                f.write(line + '\n')
            f.write("\n\nAverage Speed: {:3.1f}fps".format(average_speed))

    logging.shutdown()
    
    now = datetime.now()
    now = now.strftime("%d_%m_%Y_%H_%M")
    os.rename(os.path.join('results', args.dataset, 'trt_model'), os.path.join('results', args.dataset, 'trt_model_'+ "_" + now))

    # print("\nDone.")


if __name__ == '__main__':
    if args.log.upper() not in valid_log_levels:
        logging.error("Given log level '" + str(args.log) + "' is not valid. Exiting.")
        sys.exit(-1)
    import shutil
    if os.path.isdir(os.path.join('results', args.dataset, 'trt_model')):
        shutil.rmtree(os.path.join('results', args.dataset, 'trt_model'))
    os.makedirs(os.path.join('results', args.dataset, 'trt_model'))
    logging_path = os.path.join('results', args.dataset, 'trt_model', 'log.txt')
    logging_level = getattr(logging, args.log.upper())
    log_handlers = [logging.StreamHandler()]
    if args.logtofile:
        log_handlers.append(logging.FileHandler(logging_path))
    logging.basicConfig(level=logging_level, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', handlers=log_handlers)
    main()
