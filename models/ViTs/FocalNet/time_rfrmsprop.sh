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
train.py --data-dir=datasets/imagewoof2 --epochs=300 --num-classes=10 --sched=cosine --warmup-epochs=5 \
--batch-size=128 --reprob=0.25 --remode=pixel --smoothing=0.1 --layer-decay=0.6 --drop-path=0.1 \
--model=focalnet_tiny_srf --experiment=focalnet-exp --log-wandb \
--opt=rfrmsprop --amp --amp-dtype=bfloat16 --damping=0.0002503568662915299 --lr=8.30081414725448e-05 --lr_cov=0.031994374719137825 --momentum=0.6184064986620386 --weight-decay=0.0006059899032851918 "-j 4"


