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
--opt=sgd \
--native-amp \
--lr=0.25759171779409035 --momentum=0.9056105459161732 --weight_decay=2.891402831287728e-05

