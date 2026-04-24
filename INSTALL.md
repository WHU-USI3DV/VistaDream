## INSTILLATION

### Requirements
We use an evironment with the following specifications, packages and dependencies:

- Ubuntu 20.04
- CUDA 12.3
- Python 3.10.12
- Pytorch 2.1.0
- GeForce RTX 4090.

### Setup Instructions
- Basic environment [we update packages in ```vdplus``` environment.]
```
conda env create -f environment.yaml
conda activate vdplus
```

- Install Depth-Pro
```
cd tools/DepthPro
pip install -e .
cd ../..
```

- Install requirements of OneFormer
```
# detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# MSDA
cd tools/OneFormer/oneformer/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../../..
```


