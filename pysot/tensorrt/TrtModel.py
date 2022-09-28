import logging
import numpy as np
import cv2
import tensorrt as trt

from pysot.utils.get_trtengine import get_engine
from pysot.utils.trt_utils import allocate_buffers, do_inference
from pysot.core.config import cfg

class TrtModel:
    def __init__(self, TRT_LOGGER, runtime, target_net, target_net_pr, search_net, search_net_pr, xcorr, xcorr_pr, warmUp):
        self.TRT_LOGGER = TRT_LOGGER
        logging.info("Creating tensorrt engines ...")
        try:
            self.engine_target = get_engine('target_net', TRT_LOGGER, runtime, target_net, target_net_pr, cfg.INFERENCE.INT8_CALIBRATION.TARGET_NET, refittable=False)
            self.engine_search = get_engine('search_net', TRT_LOGGER, runtime, search_net, search_net_pr, cfg.INFERENCE.INT8_CALIBRATION.SEARCH_NET, refittable=False)
            self.engine_xcorr = get_engine('xcorr', TRT_LOGGER, runtime, xcorr, xcorr_pr, cfg.INFERENCE.INT8_CALIBRATION.XCORR, refittable=True)
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

        self.target_host_inputs, self.target_cuda_inputs, self.target_host_outputs, self.target_cuda_outputs, self.target_bindings, self.target_stream = allocate_buffers(self.engine_target)
        self.search_host_inputs, self.search_cuda_inputs, self.search_host_outputs, self.search_cuda_outputs, self.search_bindings, self.search_stream = allocate_buffers(self.engine_search)
        self.xcorr_host_inputs, self.xcorr_cuda_inputs, self.xcorr_host_outputs, self.xcorr_cuda_outputs, self.xcorr_bindings, self.xcorr_stream = allocate_buffers(self.engine_xcorr)

        # warm up engines:
        logging.info("Warming up target engine ...")
        z_crop_ini = np.zeros((1, 3, 127, 127), dtype=np.float32)
        self.warmup_engine('target', z_crop_ini, warmUp)

        logging.info("Warming up search engine ...")
        x_crop_ini = np.zeros((1, 3, 255, 255), dtype=np.float32)
        self.warmup_engine('search', x_crop_ini, warmUp)

        logging.info("Warming up xcorr engine ...")
        # xcorr sequence is:
        # cls2, cls3, cls4, loc2, loc3, loc4
        y_crop_ini = [np.zeros((1, int(cfg.RPN.KWARGS.in_channels[0]), 29, 29), dtype=np.float32), np.zeros((1, int(cfg.RPN.KWARGS.in_channels[1]), 29, 29), dtype=np.float32), np.zeros((1, int(cfg.RPN.KWARGS.in_channels[2]), 29, 29), dtype=np.float32)]
        self.warmup_engine('xcorr', y_crop_ini, warmUp)
        logging.info("warmup engines (" + str(warmUp) + " runs each) finished")
    
    def warmup_engine(self, name, ini_crop, warmUP_runs):
        if name == 'target':
            np.copyto(self.target_host_inputs[0], ini_crop.ravel())
            for i in range(warmUP_runs):
                self.warmup_output = do_inference(self.engine_target_context, self.target_bindings, self.target_host_inputs, self.target_cuda_inputs, self.target_host_outputs, self.target_cuda_outputs, self.target_stream)
        elif name == 'search':
            np.copyto(self.search_host_inputs[0], ini_crop.ravel())
            for i in range(warmUP_runs):
                self.searchf = do_inference(self.engine_search_context, self.search_bindings, self.search_host_inputs, self.search_cuda_inputs, self.search_host_outputs, self.search_cuda_outputs, self.search_stream)
        elif name == 'xcorr':
            for i in range(len(self.xcorr_host_inputs)):
                np.copyto(self.xcorr_host_inputs[i], ini_crop[i%len(ini_crop)].ravel())
                for i in range(warmUP_runs):
                    self.xcorr = do_inference(self.engine_xcorr_context, self.xcorr_bindings, self.xcorr_host_inputs, self.xcorr_cuda_inputs, self.xcorr_host_outputs, self.xcorr_cuda_outputs, self.xcorr_stream)
    
    def template(self, z_crop):
        """Calculate the features of the template image and refit the xcorr engine with these features. Therefore the are called kernels (of the convs in xcorr).
        Assignment:
        kernels[0] <-> cls2
        kernels[1] <-> loc2
        kernels[2] <-> cls3
        kernels[3] <-> loc3
        kernels[4] <-> cls4
        kernels[5] <-> loc4

        Args:
            z_crop (_type_): crop of the template image
        """
        np.copyto(self.target_host_inputs[0], z_crop.ravel())

        tic = cv2.getTickCount()
        self.kernels = do_inference(self.engine_target_context, self.target_bindings, self.target_host_inputs, self.target_cuda_inputs, self.target_host_outputs, self.target_cuda_outputs, self.target_stream)
        t_infer_kernels = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for inference of template (creating kernels) (s): " + str(t_infer_kernels))

        for i in range(len(self.kernels)):
            self.kernels[i] = self.kernels[i].reshape(self.engine_target.get_binding_shape(i+1))
        
        # refit engine_xcorr weights with kernels
        logging.debug("Refitting xcorr-engine ...")
        refitter = trt.Refitter(self.engine_xcorr, self.TRT_LOGGER)

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
        """Calculate the search features which are the input to the xcorr engine. Subsequently calculate the cls and loc feature maps with the xcorr engine.
        Assignment:
        searchf[0] <-> cls2
        searchf[1] <-> loc2
        searchf[2] <-> cls3
        searchf[3] <-> loc3
        searchf[4] <-> cls4
        searchf[5] <-> loc4

        xcorr sequence:
        cls2, cls3, cls4, loc2, loc3, loc4

        Args:
            x_crop (_type_): crop of the search image
        """
        tic = cv2.getTickCount()
        logging.debug("shape of x_crop: " + str(x_crop.shape))
        np.copyto(self.search_host_inputs[0], x_crop.ravel())
        t_copyinput = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for loading input (s): " + str(t_copyinput))

        tic = cv2.getTickCount()
        self.searchf = do_inference(self.engine_search_context, self.search_bindings, self.search_host_inputs, self.search_cuda_inputs, self.search_host_outputs, self.search_cuda_outputs, self.search_stream)
        t_infer_searchf = (cv2.getTickCount() - tic)/cv2.getTickFrequency()
        logging.debug("Time for inference of search features (s): " + str(t_infer_searchf))

        # TODO: why reshaping when the next step is .ravel()???
        for i in range(len(self.searchf)):
            self.searchf[i] = self.searchf[i].reshape(self.engine_search.get_binding_shape(i+1))

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
        self.xcorr = do_inference(self.engine_xcorr_context, self.xcorr_bindings, self.xcorr_host_inputs, self.xcorr_cuda_inputs, self.xcorr_host_outputs, self.xcorr_cuda_outputs, self.xcorr_stream)
        t_infer_xcorr = (cv2.getTickCount() - tic)/(cv2.getTickFrequency())
        logging.debug("Time for inference of xcorr (s): " + str(t_infer_xcorr))

        tic = cv2.getTickCount()
        # TODO: calculate add weights like in actual pysot tracking
        # ST add weights
        # self.xcorr[0] = self.xcorr[0] * 0.3816
        # self.xcorr[1] = self.xcorr[1] * 0.4365
        # self.xcorr[2] = self.xcorr[2] * 0.1820
        # self.xcorr[3] = self.xcorr[3] * 0.1764
        # self.xcorr[4] = self.xcorr[4] * 0.1656
        # self.xcorr[5] = self.xcorr[5] * 0.6579

        for i in range(len(self.xcorr)):
            self.xcorr[i] = self.xcorr[i].reshape(self.engine_xcorr.get_binding_shape(i+6))

        # LT add weights
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
        