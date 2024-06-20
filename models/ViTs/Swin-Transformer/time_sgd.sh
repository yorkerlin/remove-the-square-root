#source  ~/mambaforge/etc/profile.d/conda.sh
source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
conda activate pytorch2-latest

torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--nnodes=1 \
--nproc-per-node=1 \
main_search.py --cfg=configs/swin/swin_tiny_patch4_window7_224.yaml --data-path=./datasets/imagewoof2/ --batch-size=128 \
--opt_name=sgd --lr=0.31930884041600577 --momentum=0.10974379221957035 --wt=8.233012799067309e-08


