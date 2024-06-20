#source  ~/mambaforge/etc/profile.d/conda.sh
source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
conda activate pytorch2-latest


torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:0 \
--nnodes=1 \
--nproc-per-node=1 \
main_search.py \
--cfg=configs/swin/swin_tiny_patch4_window7_224.yaml --data-path=./datasets/imagewoof2/ --batch-size=128 --opt_name=ifshampoo --T=2  --beta2=0.642773584616785 --damping=8.877795017532643e-05 --lr=0.00043015011049685176 --lr_cov=0.4417508425641541 --momentum=0.6519359875635321 --wt=0.005780409991008009


