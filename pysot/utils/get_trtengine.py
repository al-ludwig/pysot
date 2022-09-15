import logging
import tensorrt as trt
import os
import sys

from pysot.utils.Int8Calibrator import Int8Calibrator, get_calibration_files

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def get_precision_builderflag(arg):
    precision_builderflag = None
    if 'fp32' in arg:
        precision_builderflag = trt.BuilderFlag.TF32
    elif 'fp16' in arg:
        precision_builderflag = trt.BuilderFlag.FP16
    elif 'int8' in arg:
        precision_builderflag = trt.BuilderFlag.INT8
    return precision_builderflag

def GiB(val):
    return val * 1 << 30

def get_engine(TRT_LOGGER, runtime, model_file, precision, calibration_path: None, refittable: bool = False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        precision_builderflag = get_precision_builderflag(precision)

        config.max_workspace_size = GiB(1)
        if precision_builderflag == trt.BuilderFlag.INT8:
            if not calibration_path:
                raise Exception("No calibration path given!.")
            NUM_IMAGES_PER_BATCH = 1
            calibration_files = get_calibration_files(calibration_path)
            config.int8_calibrator = Int8Calibrator(calibration_files, NUM_IMAGES_PER_BATCH)
            
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