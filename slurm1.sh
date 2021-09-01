#!/bin/bash -l

#SBATCH --gres gpu:1
#SBATCH -c 2
#SBATCH -p longrun

setcuda 11.3

source vev/bin/activate

python MAD-CNN/CNN.py
