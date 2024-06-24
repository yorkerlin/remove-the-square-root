#!/bin/bash
#SBATCH --qos=m2
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1

source  ~/mambaforge/etc/profile.d/conda.sh
#source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
conda activate pytorch2-latest


torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--nnodes=1 \
--nproc-per-node=1 \
train_search.py \
--config=./configs/gc_vit_xxtiny_noaug.yml \
--data_dir=datasets/imagewoof2 \
--num-classes=10 \
--experiment=gc_vit-better-exp \
--log-wandb \
--opt=shampoo \
--native-amp \
--T=2 --damping=5.458191895017763e-09 --lr=0.03167045832978714 --lr_cov=0.03212838546649561 --momentum=0.5918724700924196 --weight_decay=0.014047051201507368

