#source  ~/mambaforge/etc/profile.d/conda.sh
source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
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
--opt=rfrmsprop \
--native-amp \
--damping=0.0008239984850091628 --lr=8.734947461418809e-05 --lr_cov=0.01548699057283096 --momentum=0.9265807226857388 --weight_decay=0.010519005280385896


