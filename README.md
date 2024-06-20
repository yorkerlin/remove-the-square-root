Experiment code to reproduce results on [Can We Remove the Square-Root in Adaptive Gradient Methods? A Second-Order Perspective (ICML 2024)](https://arxiv.org/abs/2402.03496)

We provide a prototype implementation of the [root-free RMSProp](https://github.com/yorkerlin/remove-the-square-root/blob/main/myoptim/rfrmsprop.py) and [inverse-free Shampoo](https://github.com/yorkerlin/remove-the-square-root/blob/main/myoptim/ifshampoo.py).

For a clean implementation of the inverse-free Shampoo, please check out this [repository](https://github.com/f-dangel/sirfshampoo).


## Baseline Methods (Square-root-based Methods)
We use PyTorch’s built-in SGD, AdamW, and RMSProp. For Shampoo, we rely on the
state-of-the-art [PyTorch implementation](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) from Meta [(Shi et al., 2023)](https://arxiv.org/abs/2309.06497). We tune the hyperparameters (HPs) for each optimizer (see the following HP search space)

## Hyperparameter Tuning 
For matrix adaptive methods (Shampoo and inverse-free Shampoo), we update their matrix preconditioners at each two iterations. By updating the preconditioners less frequently, we can further reduce their wall clock time.
We employ a two-stage HP tuning protocol for all tasks and optimizers based on random search [(Choi et al., 2019)](https://arxiv.org/abs/1910.05446). 
Unlike  [Choi et al., 2019](https://arxiv.org/abs/1910.05446), we only consider a `small damping` term (e.g., 0 < λ < 5e−4) for all methods in our HP search space since a large damping term (e.g., λ >1 )  can turn Adam into SGD.
In the first stage, we use larger search regimes for all HPs. Based on this stage, we select a narrower HP range and re-run the search, reporting the best run for each method. We use 100 runs in each stage.

HP search space: [CNNs](https://github.com/yorkerlin/remove-the-square-root/tree/main/models/CNNs/wandb-sweep), [SwinViT](https://github.com/yorkerlin/remove-the-square-root/tree/main/models/ViTs/Swin-Transformer/wandb-sweep), VMamba, GCViT, FocalNet, LSTM, GNN

## Mixed-precision Training 
For all optimizers, only the forward pass is executed in mixed precision with `BFP-16` (as
recommended by the [official PyTorch guide](https://pytorch.org/docs/stable/amp.html#torch.autocast)). The gradients are automatically cast back to FP-32 by PyTorch. Shampoo uses
these `FP-32` gradients for its preconditioner and is unstable when converting them to BFP-16 [(Shi et al., 2023)](https://arxiv.org/abs/2309.06497). Instead, our
IF-Shampoo converts the gradients into `BFP-16`, updates the preconditioner, and even takes preconditioned gradient steps (including momentum) in
half-precision. Our method works well in half-precision `without` using `matrix decomposition` and `matrix solve/inversion`.

Note: These matrix operations (e.g., eigen, Cholesky, SVD) in `half-precision` are not supported in PyTorch and JAX because they are [numerically unstable](https://github.com/pytorch/pytorch/issues/40427).

# Todo
* add NN models and training scripts considered in our paper
* add our HP search space for each method (in the second stage) and the optimal HPs used in our paper
