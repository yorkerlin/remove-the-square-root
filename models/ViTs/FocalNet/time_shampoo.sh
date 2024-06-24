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
--opt=shampoo --T=2 --amp --amp-dtype=bfloat16 --damping=1.1108828154927186e-09 --lr=0.04412730828668628 --lr_cov=0.0020148269551425375 --momentum=0.9676809527901786 --weight-decay=7.772708658324724e-05 "-j 4"

