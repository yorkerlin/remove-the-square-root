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
--opt=ifshampoo_dense \
--native-amp \
--T=2 --beta2=0.5533682171451965 --damping=4.9681007785982885e-05 --lr=0.00022083805410345403 --lr_cov=0.16827076696097876 --momentum=0.16820648279346878 --weight_decay=0.0024023896445608027

