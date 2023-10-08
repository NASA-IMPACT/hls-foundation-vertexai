FROM nvidia/cuda:11.5.2-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.9-dev

RUN apt-get update && apt-get install -y libgl1 python3-pip git

RUN rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install -U pip

RUN pip3 install --no-cache-dir Cython

RUN pip3 install --upgrade pip

RUN pip3 install imagecodecs

RUN pip3 install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115

RUN git clone https://github.com/nasa-impact/hls-foundation-os.git

RUN cd hls-foundation-os && git checkout 9cdb612 && pip3 install -e .

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip3 install -U openmim

RUN mim install mmengine==0.7.4

RUN mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html

ENV CUDA_VISIBLE_DEVICES=0,1,2

ENV CUDA_HOME=/usr/local/cuda

ENV FORCE_CUDA="1"

COPY ./code/ /app/

CMD uvicorn app.main:app --host 0.0.0.0 --port $AIP_HTTP_PORT
