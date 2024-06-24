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
--opt=adamw \
--native-amp \
--damping=3.0058691942443522e-09 --lr=0.0004430575620569308 --lr_cov=0.0016193662851592095 --momentum=0.7362815144116387 --weight_decay=0.03439151042746387

