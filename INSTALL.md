# Installation

## Requirements

We use an evironment with the following specifications, packages and dependencies:

- Ubuntu 20.04.3 LTS
- Python 3.8.13
- conda 4.12.0
- [PyTorch v1.10.1](https://pytorch.org/get-started/previous-versions/)
- [Torchvision v0.11.2](https://pytorch.org/get-started/previous-versions/)
- [Detectron2 v0.6](https://github.com/facebookresearch/detectron2/releases/tag/v0.6)
- [NATTEN v0.14.4](https://github.com/SHI-Labs/NATTEN/releases/tag/v0.14.4)

## Setup Instructions

- Create a conda environment
  
  ```bash
  conda create --name env python=3.8 -y
  conda activate env
  ```

- Install packages and other dependencies.

  ```bash
  git clone https://github.com/orgs/HI-ComputerVision/uni-encoder-code.git
  cd uni-encoder-code

  # Install Pytorch
  conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

  # Install opencv (required for running the demo)
  pip3 install -U opencv-python

  # Install detectron2
  python tools/setup_detectron2.py

  # Install other dependencies
  pip3 install git+https://github.com/cocodataset/panopticapi.git
  pip3 install git+https://github.com/mcordts/cityscapesScripts.git
  pip3 install -r requirements.txt
  ```

- Setup CUDA Kernel for MSDeformAttn. `CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

  ```bash
  # Setup MSDeformAttn
  cd uni-encoder-code/modeling/pixel_decoder/ops
  sh make.sh
  cd ../../../..
  ```
  
