#!/bin/bash -l

#SBATCH --gres gpu:4
#SBATCH -c 4

setcuda 11.3

source vev/bin/activate

python MAD-CNN/CNN.py
