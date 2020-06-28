# Scaffold for Instance Discrimination with PyTorch Lightning

## Setup Instructions

1. Install a conda environment.
```
conda env create -f environment.yml
```

## Usage Instructions

1. For every fresh terminal, you must
```
conda activate contrastive
source init_env.sh
```
2. Edit the config files to path the right paths.
3. Run the main script
```
python scripts/run.py ./config/cifar10/pretrain.json --gpu-device 0
```
