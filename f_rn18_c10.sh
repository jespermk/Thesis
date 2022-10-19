#!/bin/sh
#BSUB -q gpuv100
#BSUB -L /bin/bash
#BSUB -J FRn18C10
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=50GB]"
#BSUB -W 24:00
#BSUB -u s202796@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err


python3 fac_resnet18_CIFAR10_1.py
