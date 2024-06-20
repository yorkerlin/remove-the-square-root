#source  ~/mambaforge/etc/profile.d/conda.sh
source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
conda activate pytorch2-latest

torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--nnodes=1 \
--nproc-per-node=1 \
main_search.py --cfg=configs/swin/swin_tiny_patch4_window7_224.yaml --data-path=./datasets/imagewoof2/ --batch-size=128 \
--opt_name=adamw --damping=2.372158719505584e-09 --lr=0.001067435163546135 --lr_cov=0.0022282143851627433 --momentum=0.9097739854378516 --wt=0.006355091220648278

