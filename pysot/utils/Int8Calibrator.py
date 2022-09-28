import tensorrt as trt
import logging
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import os
from PIL import Image
import glob

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
class Int8Calibrator(trt.IInt8EntropyCalibrator2):
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

    def __init__(self, input_shapes, calibration_files=[], batch_size=32,
                 cache_file="calibration.cache", preprocess_func=None):
        # super().__init__()
        trt.IInt8EntropyCalibrator2.__init__(self)
        # self.input_shape = input_shape
        self.input_shapes = input_shapes
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.batch = []
        self.device_inputs = []
        for input_shape in self.input_shapes:
            batch = np.ascontiguousarray(np.zeros((self.batch_size, *input_shape), dtype=np.float32))
            self.batch.append(batch)
            self.device_inputs.append(cuda.mem_alloc(batch.nbytes))

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
        # for index in range(0, len(self.files), self.batch_size):
        for index in range(0, len(self.files), len(self.input_shapes)):
            # for offset in range(self.batch_size):
            for offset in range(len(self.input_shapes)):
                ext = os.path.splitext(self.files[index + offset])[-1].lower()
                if ext == '.jpg' or ext == '.jpeg':
                    image = Image.open(self.files[index + offset])
                    image = np.array(image)
                    image = image.transpose(2, 0, 1)
                    image = image[np.newaxis, :, :, :]
                elif ext == '.npy':
                    image = np.load(self.files[index + offset])
                else:
                    logging.error("File type of calibration images is not supported! Given filetype: " + str(ext))
                    return None
                
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
            # cuda.memcpy_htod(self.device_input, batch)
            for i in range(len(self.device_inputs)):
                cuda.memcpy_htod(self.device_inputs[i], np.ascontiguousarray(batch[i]))
            # return [int(self.device_input)]
            return [int(input) for input in self.device_inputs]
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