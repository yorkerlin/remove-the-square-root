#!/bin/bash
#SBATCH --qos=m2
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1

#source  ~/mambaforge/etc/profile.d/conda.sh
source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
conda activate pytorch2-latest


torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--nnodes=1 \
--nproc-per-node=1 \
main_search.py \
--cfg=configs/swin/swin_tiny_patch4_window7_224.yaml --data-path=./datasets/imagewoof2/ --batch-size=128 --opt_name=shampoo --T=2 --damping=2.1489494820632203e-09 --lr=0.043214047452640146 --lr_cov=0.013201750018884498 --momentum=0.7697798662758283 --wt=0.001380849506134062
