#source  ~/mambaforge/etc/profile.d/conda.sh
source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
conda activate pytorch2-latest

torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--nnodes=1 \
--nproc-per-node=1 \
main_search.py --cfg=configs/swin/swin_tiny_patch4_window7_224.yaml --data-path=./datasets/imagewoof2/ --batch-size=128 \
--opt_name=rfrmsprop --damping=0.00032101260998353837 --lr=6.515923439086859e-05 --lr_cov=0.010260974799841373 --momentum=0.8597665157730524 --wt=0.0005900771997088468

