FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

WORKDIR /workspace

# Install system dependencies and python3.9 dev headers
RUN apt-get clean && apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.9 python3.9-dev python3-pip git wget unzip nano \
    libglib2.0-0 libsm6 libxext6 libxrender-dev build-essential && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch (CUDA 11.6 build)
RUN python3 -m pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# Install mmcv and mmdet
RUN python3 -m pip install openmim && \
    python3 -m pip install mmcv==2.1.0 && \
    python3 -m pip install mmdet==3.3.0

# Install other deps
RUN python3 -m pip install SharedArray scikit-image pyquaternion
RUN apt install -y libgl1

CMD ["/bin/bash"]
