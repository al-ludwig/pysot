# This dockerfile is for creating the 22.03 TensorRT NGC container
# This container is supposed to be run on a PC (not xavier)

FROM nvcr.io/nvidia/tensorrt:22.03-py3
# RUN chmod 777 tmp
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip && pip3 install --upgrade appdirs==1.4.4 colorama==0.4.4 Cython==0.29.27 \
dataclasses==0.6 decorator==5.1.1 Mako==1.1.6 MarkupSafe==2.0.1 numpy==1.19.4 onnx==1.11.0 Pillow==8.4.0 \
protobuf==3.19.4 pytools==2021.2.9 torchvision==0.10.0 PyYAML==6.0 setuptools==59.6.0 tqdm==4.62.2 \
typing_extensions==4.0.1 yacs==0.1.8 opencv-python
RUN apt-get install build-essential && apt-get install python3-dev && apt-get install -y libopenblas-dev \
&& pip3 install pycuda==2020.1 && DEBIAN_FRONTEND="noninteractive" apt-get install --yes python-opencv
# RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-opencv 
# RUN wget https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl && pip3 install torch-1.9.0-cp36-cp36m-linux_aarch64.whl
# RUN wget https://download.pytorch.org/whl/cu113/torch-1.10.0%2Bcu113-cp36-cp36m-win_amd64.whl -O torch-1.10.0cu113-cp36-cp36m-win_amd64.whl
# RUN pip3 install torch-1.10.0cu113-cp36-cp36m-win_amd64.whl
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113
RUN apt-get install -y git
RUN git clone --single-branch --branch develop https://github.com/al-ludwig/pysot.git /home/pysot
# ADD model.pth /home/pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth
ENV PYTHONPATH="/home/pysot:${PYTHONPATH}"
WORKDIR /home/pysot
RUN python3 /home/pysot/setup.py build_ext --inplace