FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qqy && \
    apt-get install -y \
      build-essential \
      cmake \
      g++ \
      git \
      ffmpeg \
      libsm6 \
      libxext6 \
      supervisor \
      libxrender1 \
      libssl-dev \
      pkg-config \
      poppler-utils \
      python-dev \
      software-properties-common \
      && \
    apt-get clean && \
    apt-get autoremove

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3-pip python3.9-venv && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip setuptools wheel
RUN python -m pip install --upgrade pip

WORKDIR /opt/code/nlp_1st_sem
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt --no-cache-dir --timeout=10000
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
RUN rm -rf /var/lib/apt/lists/* /root/.cache/pip
RUN apt-get autoremove -y && apt-get clean -y

COPY . /opt/code/nlp_1st_sem
ENV PYTHONPATH=/opt/code/nlp_1st_sem
ENV TRANSFORMERS_CACHE=/opt/workdir/cache/huggingface
CMD ["python", "sentiment/train.py"]
