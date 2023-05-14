#!/bin/bash 
#SBATCH --time=200:00:00 
#SBATCH --cpus-per-task=12 
#SBATCH --gres=gpu:1 
#SBATCH --mem=72G 
#SBATCH --output=/Users/sjia/Documents/Research/MyPapers/contrastivePFL/repo/code/plato/examples/pfl/slurm_loggings/CF10noniid03_100/byol_CIFAR10_resnet18_500.out 

/home/sijia/envs/miniconda3/envs/INFOCOM23/bin/python /Users/sjia/Documents/Research/MyPapers/contrastivePFL/repo/code/plato/examples/pfl/approaches/SSL/byol/byol.py -c /Users/sjia/Documents/Research/MyPapers/contrastivePFL/repo/code/plato/examples/pfl/approaches/configs/CF10noniid03_100/byol_CIFAR10_resnet18_500.yml -b /data/sijia/NIPS23/experiments/CF10noniid03_100