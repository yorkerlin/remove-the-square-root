PyTorch implementation of our square-root-free adaptive methods (root-free RMSProp and Inverse-free Shampoo without the root) based on [Can We Remove the Square-Root in Adaptive Gradient Methods? A Second-Order Perspective (ICML 2024)](https://arxiv.org/abs/2402.03496)


## Baseline Adaptive Methods
We use PyTorch’s built-in SGD, AdamW, and RMSProp. For Shampoo, we rely on the
state-of-the-art [PyTorch implementation](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) from Meta [(Shi et al., 2023)](https://arxiv.org/abs/2309.06497). We tune the hyperparameters (HPs) for each optimizer (see the HP search space)

## Hyperparameter Tuning 
For matrix adaptive methods (Shampoo and IF-Shampoo), we update their matrix preconditioners at each two iterations.
We employ a two-stage HP tuning protocol for all tasks and optimizers based on random search [(Choi et al., 2019)](https://arxiv.org/abs/1910.05446). 
Unlike  [(Choi et al., 2019)](https://arxiv.org/abs/1910.05446), we only consider a `small damping` term (e.g., 0 < λ < 10−4) for all methods in our HP search space. 
In the first stage, we use larger search regimes for all HPs. Based on this stage, we select a narrower HP range and re-run the search, reporting the best run for each method. We use 100 runs in each stage.

## Mixed-precision Training 
For all optimizers, only the forward pass is executed in mixed precision with `BFP-16` (as
recommended by the official PyTorch guide). The gradients are automatically cast back to FP-32 by PyTorch. Shampoo uses
these `FP-32` gradients for its preconditioner and is unstable when converting them to BFP-16 [(Shi et al., 2023)](https://arxiv.org/abs/2309.06497). Instead, our
IF-Shampoo converts the gradients into `BFP-16`, updates the preconditioner, and even takes preconditioned gradient steps (including momentum) in
half precision. Our method works well in half-precision `without` using `matrix decomposition` and `matrix solve/inversion`.

Note: These matrix operations in half-precision are not supported in PyTorch and JAX because they are numerically unstable.

# Todo
* add the root-free RMSProp and inverse-free Shampoo
* add NN models and training scripts considered in our paper
* add our HP search space for each method (in the second stage) and the optimal HPs used in our paper
