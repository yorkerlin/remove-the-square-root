"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from distributed_shampoo.utils.shampoo_fsdp_utils import (
    CommunicatedSplitShampooPreconditioner,
    CommunicationType,
    SplitShampooPreconditioner,
    TensorBlockRecoveryMethod,
)

from distributed_shampoo.utils.shampoo_utils import GraftingType, LargeDimMethod

logger = logging.getLogger(__name__)

BETAS = "betas"
EXP_AVG = "exp_avg"
EPSILON = "epsilon"
GRAFTING_BETA2 = "grafting_beta2"
GRAFTING_EPSILON = "grafting_epsilon"
GRAFTING_MOMENTUM = "grafting_momentum"
LR = "lr"
MOMENTUM = "momentum"
PARAMS = "params"
PRECONDITIONERS = "preconditioners"
STEP = "step"
WEIGHT_DECAY = "weight_decay"


class FSDPShampoo(torch.optim.Optimizer):
    """Prototype distributed Shampoo algorithm for Fully Sharded Data Parallel training.

    Under development. Built on previous prototype of DDP Distributed Shampoo.

    Developers:
        Anna Cai (Meta Platforms, Inc.)
        Hao-Jun Michael Shi (Meta Platforms, Inc.)
        Tsung-Hsien Lee
        Shintaro Iwasaki (Meta Platforms, Inc.)

    with support from Andrew Gu (Meta), Wanchao Liang (Meta), and Mike Rabbat (Meta).

    Uses infinity norm to evaluate residuals and errors. By default, grafts from Adagrad.

    ------------
    Requirements
    ------------

    1. PyTorch >= 2.0
    2. Python >= 3.8
    3. CUDA 11.3, 11.4, 12.2+

    If one wants to use DTensor which leads to memory savings, please set use_dtensor = True. Requires PyTorch 2 nightly build.

    Note: We have observed known instabilities with the torch.linalg.eigh operator on CUDA 11.6-12.1, specifically for low-rank
    matrices, which may appear with using a small start_preconditioning_step. Please avoid these versions of CUDA if possible.
    See: https://github.com/pytorch/pytorch/issues/94772.

    --------
    Features
    --------

    1. Layerwise Grafting: In order to tune Shampoo, we can "graft" a layer-wise learning rate schedule from a previous method
        and apply it to Shampoo. This is performed by taking the norm of the layer-wise step of the grafted method, normalizing
        the Shampoo step, and re-scaling the normalized Shampoo step by the product of the norm of the grafted step + learning rate.

        This may be interpreted as an additional block re-scaling of the entire Shampoo preconditioner.
        This is the key ingredient to making Shampoo work in practice.

        We support the following methods:
            - GraftingType.NONE: Performs no grafting.
            - GraftingType.SGD: Grafts the stochastic gradient method.
            - GraftingType.ADAGRAD: Grafts the Adagrad method.
            - GraftingType.RMSPROP: Grafts the RMSProp method.
            - GraftingType.ADAM: Grafts the Adam method.
            - GraftingType.ADAGRAD_NORMALIZED: Grafts the Adagrad method with normalized gradients.
            - GraftingType.RMSPROP_NORMALIZED: Grafts the RMSProp method with normalized gradients.
            - GraftingType.ADAM_NORMALIZED: Grafts the Adam method with normalized gradients.

        NOTE: These methods do not graft the first-moment component - it is entirely based upon grafting using the
        diagonal preconditioner. If using an exponential moving average of the gradient (or gradient filtering), we
        can set beta1 as the same value from before, and both Shampoo and the grafted method will use the filtered
        gradient.

    2. Large-Dimensional Tensors: Supports multiple approaches for scaling Shampoo to tensors with large dimensions.
        For simplicity, we explain using a linear layer/matrix parameter, although this is generalizable to higher-order
        tensors.

        Suppose that W is a m x n matrix, i.e.,

            [[w_11 w_12 ... w_1n]
             [w_21 w_22 ... w_2n]
        W =           :
             [w_m1 w_m2 ... w_mn]]

        - LargeDimMethod.BLOCKING (Default): Given a max_preconditioner_dim tau > 0, blocks W and applies Shampoo to
            each block, i.e., if tau divides both m, n, then:

                [[W_11 W_12 ... W_1k]
                 [W_21 W_22 ... W_2k]
            W =           :
                 [W_l1 W_l2 ... W_lk]]

            and apply Shampoo to W_ij which is a tau x tau matrix. This can be viewed as further blocking each block of the
            block-diagonal preconditioner.

            Computational cost = O(tau^3)
            Memory cost = 4mn (including root inverse preconditioners)

        - LargeDimMethod.ADAGRAD: Given a max_preconditioner_dim tau > 0, checks if any dimensions of the tensor is greater
            than tau. If so, uses Adagrad preconditioner in place of Shampoo. Corresponds to a diagonal preconditioner.

            Computational cost = O(mn)
            Memory cost = mn

        - LargeDimMethod.DIAGONAL: Given a max_preconditioner_dim tau > 0, uses a diagonal Shampoo preconditioner in place of
            the full Shampoo preconditioner. Corresponds to a (crude) diagonal preconditioner.

            Computational cost = O(mn)
            Memory cost = m + n

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        param_metadata (Dict[torch.nn.Parameter, Tuple]): FSDP shard metadata for each parameter consisting of fqn,
            original shape, original numels, and shard param info. See FSDP class FlatParameter for more details.
            https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/flat_param.py#L190
        lr (float): learning rate (Default: 1e-2)
        betas (Tuple[float, float]): coefficients used for computing running averages
            of gradient and its square (Default: (0.9, 1.0))
        epsilon (float): term added to the denominator to improve numerical stability (Default: 1e-12)
        momentum (float): momentum parameter (default: 0.)
        weight_decay (float): weight decay (L2 penalty) (Default: 0)
        max_preconditioner_dim (int): maximum preconditioner dimension (Default: 1024)
        precondition_frequency (int): frequency for computing root inverse preconditioner (Default: 1)
        start_preconditioning_step (int): iteration to start computing inverse preconditioner. If -1, uses
            the same value as precondition_frequency. (Default: -1)
        exponent_override (int): exponent to use in Shampoo. (Default: 0)
        exponent_multiplier (float): number to be multiplied to the numerator of the inverse root. (Default: 1.0)
        use_nesterov (bool): uses Nesterov momentum (default: False)
        use_bias_correction (bool): flag for using bias correction (Default: True)
        use_decoupled_weight_decay (bool): Flag for using AdamW-style decoupled weight decay (Default: True)
        preconditioner_dtype (torch.dtype): data type for preconditioner (Default: torch.float)
        large_dim_method (LargeDimMethod): method for handling large scale tensors. (Default: LargeDimMethod.BLOCKING)
        num_trainers_per_group (int): number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the LOCAL_WORLD_SIZE. (Default: -1)
        use_merge_dims (bool): merge dimensions if possible while respecting max_preconditioner_dim. (Default: True)
        grafting_type (GraftingType): selects grafting method. (Default: GraftingType.ADAGRAD)
        grafting_epsilon (float): epsilon for grafting method. (Default: 1e-3)
        grafting_beta2 (float): exponential moving average factor for grafting method. (Default: 1.0)
        use_protected_eigh (bool): Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.
        use_dtensor (bool): use DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        debug_mode (bool): debugging mode. Uses more memory to compute error to fp64 case. Must enable logging level to
            DEBUG. (Default: False)
        tensor_block_recovery (TensorBlockRecoveryMethod): method for tensor block recovery.
            (Default: TensorBlockRecoveryMethod.COMM) NOTE: change default in the future after experimenting
        dist_group (Optional[dist.ProcessGroup]): Process group for distributed computation. Only relevant if HSDP is
            used. (Default: None)

    """

    def __init__(
        self,
        params,
        param_metadata: Dict[torch.nn.Parameter, Tuple],
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 1.0),
        epsilon: float = 1e-12,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        max_preconditioner_dim: int = 1024,
        precondition_frequency: int = 1,
        start_preconditioning_step: int = -1,
        exponent_override: int = 0,
        exponent_multiplier: float = 1.0,
        use_nesterov: bool = False,
        use_bias_correction: bool = True,
        use_decoupled_weight_decay: bool = True,
        preconditioner_dtype: torch.dtype = torch.float,
        large_dim_method: LargeDimMethod = LargeDimMethod.BLOCKING,
        num_trainers_per_group: int = -1,
        use_merge_dims: bool = True,
        grafting_type: GraftingType = GraftingType.ADAGRAD,
        grafting_epsilon: float = 1e-3,
        grafting_beta2: float = 1.0,
        use_protected_eigh: bool = True,
        use_dtensor: bool = False,
        debug_mode: bool = False,
        tensor_block_recovery: TensorBlockRecoveryMethod = TensorBlockRecoveryMethod.COMM,
        dist_group: Optional[dist.ProcessGroup] = None,
    ):
        logger.info(
            "FSDP Shampoo is experimental and still under development! Checkpointing is not currently supported."
        )

        # Hyperparameter checks.
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be >= 0.0.")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 0: {betas[0]}. Must be in [0.0, 1.0)."
            )
        if not 0.0 < betas[1] <= 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 1: {betas[1]}. Must be in (0.0, 1.0]."
            )
        if not epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}. Must be > 0.0.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"Invalid momentum parameter: {momentum}. Must be [0.0, 1.0)."
            )
        if not weight_decay >= 0.0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}. Must be > 0.0."
            )
        if not max_preconditioner_dim >= 1:
            raise ValueError(
                f"Invalid max preconditioner dimension: {max_preconditioner_dim}. Must be >= 1."
            )
        if not precondition_frequency >= 1:
            raise ValueError(
                f"Invalid precondition frequency: {precondition_frequency}. Must be >= 1."
            )
        if not start_preconditioning_step >= -1:
            raise ValueError(
                f"Invalid start preconditioning step: {start_preconditioning_step}"
            )
        if not num_trainers_per_group >= -1:
            raise ValueError(
                f"Invalid number of GPUs per group: {num_trainers_per_group}. Must be >= -1."
            )
        if not exponent_override >= 0:
            raise ValueError(
                f"Invalid exponent override: {exponent_override}. Must be >= 0."
            )
        if not 0.0 < grafting_beta2 <= 1.0:
            raise ValueError(
                f"Invalid grafting beta parameter: {grafting_beta2}. Must be in (0.0, 1.0]."
            )
        if not grafting_epsilon > 0.0:
            raise ValueError(
                f"Invalid epsilon value: {grafting_epsilon}. Must be > 0.0."
            )

        # Distributed checks.
        if num_trainers_per_group > 1 or num_trainers_per_group == -1:
            if not torch.cuda.is_available():
                raise ValueError("Using distributed version of Shampoo without GPUs!")
            if not dist.is_initialized():
                raise ValueError(
                    "Using distributed version of Shampoo without initializing distributed process group!"
                )

            # Defaults to number of GPUs per node if using -1.
            if num_trainers_per_group == -1:
                num_trainers_per_group = int(
                    os.environ.get("LOCAL_WORLD_SIZE", dist.get_world_size())
                )

            if not dist.get_world_size() >= num_trainers_per_group:
                num_trainers_per_group = dist.get_world_size()
                logger.warning(
                    f"Number of GPUs per group {num_trainers_per_group} is specified larger than global world size {dist.get_world_size()}. Setting to default world size."
                )
            if not dist.get_world_size() % num_trainers_per_group == 0:
                raise ValueError(
                    f"Invalid number of GPUs per group: {num_trainers_per_group}. Must divide global world size {dist.get_world_size()}."
                )
        else:
            num_trainers_per_group = 1

        super(FSDPShampoo, self).__init__(
            params,
            {
                LR: lr,
                BETAS: betas,
                MOMENTUM: momentum,
                WEIGHT_DECAY: weight_decay,
                EPSILON: epsilon,
                GRAFTING_EPSILON: grafting_epsilon,
                GRAFTING_BETA2: grafting_beta2,
            },
        )

        # Initialize algorithm-related fields.
        self._param_metadata = param_metadata
        self._max_preconditioner_dim = max_preconditioner_dim
        self._precondition_frequency = precondition_frequency
        self._exponent_override = exponent_override
        self._exponent_multiplier = exponent_multiplier
        self._num_trainers_per_group = num_trainers_per_group
        self._use_merge_dims = use_merge_dims
        self._large_dim_method = large_dim_method
        self._use_decoupled_weight_decay = use_decoupled_weight_decay
        self._preconditioner_dtype = preconditioner_dtype
        self._use_bias_correction = use_bias_correction
        self._grafting_type = grafting_type
        self._grafting_epsilon = grafting_epsilon
        self._grafting_beta2 = grafting_beta2
        self._parameter_count = 0
        self._use_nesterov = use_nesterov
        self._use_protected_eigh = use_protected_eigh
        self._use_dtensor = use_dtensor
        self._debug_mode = debug_mode
        self._tensor_block_recovery = tensor_block_recovery
        self._dist_group = dist_group
        if self._use_nesterov and momentum == 0.0:
            logger.warning(
                "Nesterov flag is enabled but momentum parameter is zero! Continuing without using momentum or Nesterov acceleration..."
            )

        if start_preconditioning_step == -1:
            self._start_preconditioning_step = precondition_frequency
            logger.warning(
                f"start_preconditioning_step set to -1. Setting start_preconditioning_step equal to precondition frequency {precondition_frequency} by default."
            )
        elif start_preconditioning_step < precondition_frequency:
            raise ValueError(
                f"Invalid start_preconditioning_step value: {start_preconditioning_step}. Must be >= {precondition_frequency = }."
            )
        else:
            self._start_preconditioning_step = start_preconditioning_step

        self._initialize_preconditioners_and_steps()

    @torch.no_grad()
    def _initialize_preconditioners_and_steps(self):
        """Initialize Shampoo preconditioners and inverse preconditioners."""
        group_rank = dist.get_rank(group=self._dist_group)
        group_size = dist.get_world_size(group=self._dist_group)

        for group in self.param_groups:
            for idx, p in enumerate(group[PARAMS]):
                # skip parameters not on worker
                if p.numel() == 0:
                    continue

                state = self.state[p]
                state[STEP] = torch.tensor(0)

                if p not in self._param_metadata:
                    raise RuntimeError(
                        f"Parameter {p} not found in metadata. Please make sure that the module containing this parameter has been wrapped with FSDP."
                    )

                if self._tensor_block_recovery == TensorBlockRecoveryMethod.SPLIT:
                    state[PRECONDITIONERS] = SplitShampooPreconditioner(
                        p,
                        self._param_metadata[p],
                        large_dim_method=self._large_dim_method,
                        beta1=group[BETAS][0],
                        beta2=group[BETAS][1],
                        epsilon=group[EPSILON],
                        exponent_override=self._exponent_override,
                        exponent_multiplier=self._exponent_multiplier,
                        use_bias_correction=self._use_bias_correction,
                        max_preconditioner_dim=self._max_preconditioner_dim,
                        dtype=self._preconditioner_dtype,
                        idx=idx,
                        use_merge_dims=self._use_merge_dims,
                        start_preconditioning_step=self._start_preconditioning_step,
                        grafting_type=self._grafting_type,
                        grafting_beta2=self._grafting_beta2,
                        grafting_epsilon=self._grafting_epsilon,
                        use_protected_eigh=self._use_protected_eigh,
                        use_dtensor=self._use_dtensor,
                    )
                elif self._tensor_block_recovery == TensorBlockRecoveryMethod.COMM:
                    # TODO: can do more complex assignment here
                    left_comm = (
                        CommunicationType.NONE
                        if group_rank == 0
                        else CommunicationType.RECV
                    )
                    right_comm = (
                        CommunicationType.NONE
                        if group_rank == group_size - 1
                        else CommunicationType.SEND
                    )

                    state[PRECONDITIONERS] = CommunicatedSplitShampooPreconditioner(
                        p,
                        self._param_metadata[p],
                        large_dim_method=self._large_dim_method,
                        beta1=group[BETAS][0],
                        beta2=group[BETAS][1],
                        epsilon=group[EPSILON],
                        exponent_override=self._exponent_override,
                        exponent_multiplier=self._exponent_multiplier,
                        use_bias_correction=self._use_bias_correction,
                        max_preconditioner_dim=self._max_preconditioner_dim,
                        dtype=self._preconditioner_dtype,
                        idx=idx,
                        use_merge_dims=self._use_merge_dims,
                        start_preconditioning_step=self._start_preconditioning_step,
                        grafting_type=self._grafting_type,
                        grafting_beta2=self._grafting_beta2,
                        grafting_epsilon=self._grafting_epsilon,
                        use_protected_eigh=self._use_protected_eigh,
                        use_dtensor=self._use_dtensor,
                        left_comm=left_comm,
                        right_comm=right_comm,
                    )
                else:
                    raise NotImplementedError(
                        f"Invalid tensor block recovery method {self._tensor_block_recovery}!"
                    )

                # Count parameters from preconditioners for logging purposes.
                self._parameter_count += state[PRECONDITIONERS].parameter_count

        # Logs total number of parameters for optimizer.
        logger.info(f"Total Parameter Count: {self._parameter_count}")

    @torch.no_grad()
    def _send_grad(self, forward_direction: bool):
        # Get communication ops from all of the preconditioners.
        ops = []
        for group in self.param_groups:
            for p in group[PARAMS]:
                # skip parameters not on worker
                if p.numel() == 0 or p.grad is None:
                    continue
                state = self.state[p]

                if forward_direction:
                    ops.extend(state[PRECONDITIONERS].get_forward_ops(p.grad))
                else:  # backward
                    ops.extend(state[PRECONDITIONERS].get_backward_ops())

        if len(ops) > 0:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                # Wait for communication to complete before resuming computation.
                req.wait()

    @torch.no_grad()
    def _compute_root_inverse(self):
        """Root inverse computation across all preconditioners/parameters."""
        for group in self.param_groups:
            for p in group[PARAMS]:
                # skip parameters not on worker
                if p.numel() == 0 or p.grad is None:
                    continue

                state = self.state[p]

                if isinstance(
                    state[PRECONDITIONERS],
                    (
                        SplitShampooPreconditioner,
                        CommunicatedSplitShampooPreconditioner,
                    ),
                ):
                    state[PRECONDITIONERS].compute_root_inverse()

    @torch.no_grad()
    def _compute_and_log_root_inverse_residuals(
        self,
    ):
        """Compute root inverse residuals over all preconditioners."""

        # Compute expected relative errors/residuals for debugging purposes
        if self._preconditioner_dtype == torch.float64:
            expected_relative_error = 1e-7
        elif self._preconditioner_dtype == torch.float:
            expected_relative_error = 1e-3
        else:
            logger.warning(
                "Expected relative error/residual not supported for precision lower than float32."
            )

        # Accumulate relative errors/residuals
        relative_errors = []
        relative_residuals = []

        for group in self.param_groups:
            for p in group[PARAMS]:
                # skip parameters not on worker
                if p.numel() == 0:
                    continue

                state = self.state[p]

                if isinstance(
                    state[PRECONDITIONERS],
                    (
                        SplitShampooPreconditioner,
                        CommunicatedSplitShampooPreconditioner,
                    ),
                ):
                    relative_error, relative_residual = state[
                        PRECONDITIONERS
                    ].compute_root_inverse_residuals()

                    relative_errors += relative_error
                    relative_residuals += relative_residual

        relative_errors = torch.stack(relative_errors)
        relative_residuals = torch.stack(relative_residuals)

        quantiles = torch.as_tensor(
            [0, 0.25, 0.5, 0.75, 1],
            device=relative_errors.device,
            dtype=relative_errors.dtype,
        )
        logger.debug(f"Expect Relative Error <= {expected_relative_error}")
        logger.debug(
            f"Relative Error (||X - X_hat||_inf / ||X||_inf)       Average: {torch.mean(relative_errors)}, Quantiles [0, 25, 50, 75, 100]: {torch.quantile(relative_errors, quantiles, interpolation='nearest')}"
        )
        logger.debug(
            f"Relative Residual (||X_hat^-r - A||_inf / ||A||_inf) Average: {torch.mean(relative_residuals)}, Quantiles [0, 25, 50, 75, 100]: {torch.quantile(relative_residuals, quantiles, interpolation='nearest')}"
        )

    @torch.no_grad()
    def _apply_weight_decay(self):
        """Incorporate weight decay into the gradient if we are not using decoupled weight decay.

        Equivalent to adding an L2-regularization term:
          F(w) + lambda * ||w||^2.

        """
        for group in self.param_groups:
            weight_decay = group[WEIGHT_DECAY]
            for p in group[PARAMS]:
                # skip parameters not on worker
                if p.numel() == 0 or p.grad is None:
                    continue
                grad = p.grad

                # TODO: Sparse case still not supported.
                if grad.is_sparse:
                    raise Exception(
                        "Sparse parameters are not currently supported by Shampoo."
                    )

                if weight_decay != 0:
                    grad.add_(p, alpha=weight_decay)

    @torch.no_grad()
    def _update_preconditioners(self):
        """Updates preconditioners.

        Note: If using L2-regularization/weight decay, it is NOT computed within this function and
        therefore should be computed elsewhere beforehand.

        """
        for group in self.param_groups:
            for p in group[PARAMS]:
                # skip parameters not on worker
                if p.numel() == 0:
                    continue

                grad = p.grad
                state = self.state[p]
                if grad is None:
                    continue

                # TODO: Sparse case still not supported.
                if p.grad.is_sparse:
                    raise Exception(
                        "Sparse parameters are not currently supported by Shampoo."
                    )

                else:
                    # Update each preconditioner using the gradient.
                    state[PRECONDITIONERS].update_preconditioners(grad, state[STEP])

    @torch.no_grad()
    def _init_group(
        self,
        group: Dict[str, Any],
        iteration: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        # Set momentum parameter
        momentum_param = group[MOMENTUM]

        # Instantiate lists for params, grads, and momentum.
        split_params = []
        split_preconditioned_grads = []
        split_momentum_directions = []

        for p in group[PARAMS]:
            # skip parameters not on worker
            if p.numel() == 0 or p.grad is None:
                continue

            if p.grad.is_sparse:
                raise Exception(
                    "Sparse parameters are not currently supported by Shampoo."
                )

            state = self.state[p]

            # Initialize momentum of gradient.
            if momentum_param != 0.0 and MOMENTUM not in state:
                state[MOMENTUM] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )

            # Generate split lists.
            split_params.extend(state[PRECONDITIONERS].apply_split(p))

            if self._tensor_block_recovery == TensorBlockRecoveryMethod.SPLIT:
                # Compute preconditioned gradient.
                split_preconditioned_grads.extend(
                    state[PRECONDITIONERS].precondition(p.grad, iteration)
                )
            elif self._tensor_block_recovery == TensorBlockRecoveryMethod.COMM:
                split_preconditioned_grads.extend(
                    state[PRECONDITIONERS].retrieve_preconditioned_grad()
                )
            else:
                raise NotImplementedError(
                    "invalid tensor block recovery method {self._tensor_block_recovery}"
                )

            split_momentum_directions.extend(
                state[PRECONDITIONERS].apply_split(state[MOMENTUM])
                if momentum_param != 0.0
                else []
            )

        return split_params, split_preconditioned_grads, split_momentum_directions

    @torch.no_grad()
    def _iterate_step(self) -> torch.Tensor:
        iteration = None
        for group in self.param_groups:
            for p in group[PARAMS]:
                # skip parameters not on worker
                if p.numel() == 0:
                    continue

                self.state[p][STEP] += 1
                iteration = self.state[p][STEP]
        return iteration

    @torch.no_grad()
    def reset_preconditioners(self):
        for group in self.param_groups:
            for p in group[PARAMS]:
                # skip parameters not on worker
                if p.numel() == 0:
                    continue

                self.state[p][PRECONDITIONERS].reset_preconditioners()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        iteration = self._iterate_step()

        if not self._use_decoupled_weight_decay:
            self._apply_weight_decay()

        # If applicable, send gradient slices forward.
        if self._tensor_block_recovery == TensorBlockRecoveryMethod.COMM:
            self._send_grad(forward_direction=True)

        self._update_preconditioners()

        # Computes root inverse of all preconditioners every self._precondition_frequency
        # after the self._start_preconditioning_step iteration.
        if (
            iteration % self._precondition_frequency == 0
            and iteration >= self._start_preconditioning_step
        ):
            self._compute_root_inverse()

            if self._debug_mode:
                self._compute_and_log_root_inverse_residuals()

        # Loops over all parameter groups and parameters to compute exponential moving average and, if applicable, preconditioned gradients.
        for group in self.param_groups:
            for p in group[PARAMS]:
                state = self.state[p]
                if p.numel() == 0 or p.grad is None:
                    continue

                # Incorporate first-moment or filtered gradient estimation.
                # TODO: maybe this can be moved inside the precondition function, unsure if that would be clean
                state[PRECONDITIONERS].update_exp_avg(p.grad, iteration)

                if self._tensor_block_recovery == TensorBlockRecoveryMethod.COMM:
                    # Compute preconditioned gradient and store within class/buffers.
                    state[PRECONDITIONERS].precondition_and_store(p.grad, iteration)

        # If applicable, send gradient slices backward.
        if self._tensor_block_recovery == TensorBlockRecoveryMethod.COMM:
            self._send_grad(forward_direction=False)

        # Loops over all parameter groups and parameters to perform update.
        for group in self.param_groups:
            momentum_param = group[MOMENTUM]
            weight_decay = group[WEIGHT_DECAY]
            lr = group[LR]

            # Compute/retrieve preconditioned gradients and momentum.
            (
                split_params,
                split_preconditioned_grads,
                split_momentum_directions,
            ) = self._init_group(group, iteration)

            # Set search direction as preconditioned grads.
            split_search_directions = split_preconditioned_grads

            # Incorporate decoupled weight decay.
            if self._use_decoupled_weight_decay and weight_decay != 0.0:
                # Decoupled weight decay (no momentum case)
                if momentum_param == 0.0:
                    torch._foreach_mul_(split_params, 1.0 - lr * weight_decay)

                # Decoupled weight decay (momentum case)
                else:
                    torch._foreach_add_(
                        split_search_directions, split_params, alpha=weight_decay
                    )

            # Update momentum.
            if momentum_param != 0.0:
                torch._foreach_mul_(split_momentum_directions, momentum_param)
                torch._foreach_add_(split_momentum_directions, split_search_directions)

                # Incorporates Nesterov momentum.
                if self._use_nesterov:
                    torch._foreach_add_(
                        split_search_directions,
                        split_momentum_directions,
                        alpha=momentum_param,
                    )

                else:
                    split_search_directions = split_momentum_directions

            # Updates weights.
            torch._foreach_add_(split_params, split_search_directions, alpha=-lr)

        return loss
