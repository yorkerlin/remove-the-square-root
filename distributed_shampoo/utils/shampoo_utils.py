"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from distributed_shampoo.utils.matrix_functions import (
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_inverse_root,
)
from distributed_shampoo.utils.optimizer_modules import OptimizerModule
from distributed_shampoo.utils.shampoo_dist_utils import (
    allocate_distributed_tensor,
    get_dtype_size,
    use_local_tensor,
)
from torch import Tensor
from torch.autograd import profiler

logger = logging.getLogger(__name__)

###### ENUM CLASSES ######
class PreconditionerType(enum.Enum):
    FULL = 0
    DIAGONAL = 1


class GraftingType(enum.Enum):
    NONE = 0
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    ADAM = 4
    ADAGRAD_NORMALIZED = 5
    RMSPROP_NORMALIZED = 6
    ADAM_NORMALIZED = 7
    LARS = 8
    LAMB = 9


class CommunicationDType(enum.Enum):
    DEFAULT = 0
    FP16 = 1
    BF16 = 2
    FP32 = 3


class LargeDimMethod(enum.Enum):
    DIAGONAL = 0
    ADAGRAD = 1
    BLOCKING = 2


# DType mapping for quantized communications.
dtype_mapping = {0: "DEFAULT", 1: torch.float16, 2: torch.bfloat16, 3: torch.float32}


###### MERGING AND BLOCKING HELPER FUNCTIONS ######
def merge_small_dims(tensor_shape: List[int], threshold: int) -> List[int]:
    """Reshapes tensor by merging small dimensions.

    Args:
        tensor_shape (List[int]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.

    Returns:
        new_tensor_shape (List[int]): New tensor shape.

    """

    new_tensor_shape = [tensor_shape[0]]
    for next_tensor_shape in tensor_shape[1:]:
        new_dimension = new_tensor_shape[-1] * next_tensor_shape
        if (
            new_tensor_shape[-1] == 1
            or next_tensor_shape == 1
            or new_dimension <= threshold
        ):
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(next_tensor_shape)

    return new_tensor_shape


def multi_dim_split(tensor: Tensor, splits: List[int]) -> List[Tensor]:
    """Chunks tensor across multiple dimensions based on splits.

    Args:
        tensor (Tensor): Gradient or tensor to split.
        splits (List[int]): List of sizes for each block or chunk along each dimension.

    Returns:
        split_grad (List[Tensor]): List of tensors.

    """
    split_tensors = [tensor]
    for dim, split in enumerate(splits):
        split_tensors = [
            s for t in split_tensors for s in torch.split(t, split, dim=dim)
        ]
    return split_tensors


def multi_dim_cat(split_tensors: List[Tensor], num_splits: List[int]) -> Tensor:
    """Concatenates multiple tensors to form single tensor across multiple dimensions.

    Args:
        split_tensor (List[Tensor]): List of tensor splits or blocks.
        num_splits (List[int]): Number of splits/blocks.

    Returns:
        merged_tensor (Tensor): Merged tensor.

    """
    merged_tensor = split_tensors
    for dim, split in reversed(list(enumerate(num_splits))):
        if split > 0:
            merged_tensor = [
                torch.cat(merged_tensor[i : i + split], dim=dim)
                for i in range(0, len(merged_tensor), split)
            ]
    assert len(merged_tensor) == 1
    return merged_tensor[0]


###### PRECONDITIONER CLASSES ######
class Preconditioner(OptimizerModule):
    """Preconditioner base class.

    Args:
        param (Tensor): Parameter of interest.

    """

    def __init__(
        self,
        param,
    ):
        super(Preconditioner, self).__init__()
        self._parameter_count = 0
        self._dims = list(param.size())
        self._block_count = 1
        self._num_bytes = 0

    def update_preconditioners(self, grad: Tensor, iteration: int) -> None:
        pass

    def precondition(self, grad: Tensor, iteration: int) -> Union[Tensor, List[Tensor]]:
        pass

    def compute_norm(self, grad: Tensor, iteration: int) -> Tensor:
        pass

    @property
    def block_count(self) -> int:
        return self._block_count

    @property
    def parameter_count(self) -> int:
        return self._parameter_count


class DistributedPreconditioner(Preconditioner):
    """Distributed Preconditioner class.

    Builds on top of Preconditioner base class to instantiate group and distributed buffer information.

    Args:
        param (Tensor): Parameter of interest.
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Source rank (or owner) of preconditioner data. (Default: 0)
        dist_buffer (Optional[Tensor]): Distributed buffer for distributed computation. (Default: None)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(DistributedPreconditioner, self).__init__(param)

        # Initialize distributed buffer and source rank.
        # Note that Adagrad dtype is the same as parameter/gradient dtype.
        group_size = (
            dist.get_world_size(group)
            if dist.is_initialized() and group is not None
            else 1
        )
        group_rank = (
            dist.get_rank(group) if dist.is_initialized() and group is not None else 0
        )

        # Initializes distributed buffer if dist_buffer is provided;
        # otherwise, sets to default values.
        if dist_buffer is not None:
            requested_dist_buffer_size = self.get_dist_buffer_size(
                param, communication_dtype
            )

            # We ignore the remainder chunk.
            self._dist_buffer = (
                dist_buffer.split(requested_dist_buffer_size)[0]
                .view(
                    param.dtype
                    if communication_dtype == CommunicationDType.DEFAULT
                    else dtype_mapping[communication_dtype.value]
                )
                .view(param.shape)
            )

        else:
            self._dist_buffer = None

        # Flag for determining whether or not we are on source rank.
        # If group is None and group source rank is not provided, then this will be True.
        self._group_source_rank = group_source_rank
        self._on_source_rank = group_rank == group_source_rank

        # Initialize device mesh and placements for DTensor.
        global_size = dist.get_world_size() if dist.is_initialized() else 1
        self._device_mesh_ranks = [
            *range(group_source_rank % group_size, global_size, group_size)
        ]

    def get_debug_info(self) -> Dict[str, str]:
        debug_info = {
            "name": "DistributedPreconditioner",
            "dims": str(tuple(self._dims)),
            "num_bytes": str(self._num_bytes),
        }
        if self._dist_buffer is not None:
            debug_info.update(
                {
                    "group_source_rank": str(self._group_source_rank),
                    "dist_buffer": f"({tuple(self._dist_buffer.size())}, dtype={self._dist_buffer.dtype})",
                }
            )

        return debug_info

    def combine_and_split_dims(self, p: Tensor) -> List[Tensor]:
        return [p]

    def get_split_parameters(self, param: Tensor) -> List[Tensor]:
        return self.combine_and_split_dims(param)

    @staticmethod
    def get_dist_buffer_size(param, communication_dtype: CommunicationDType) -> int:
        # Get the buffer size in bytes
        return int(
            math.prod(param.shape)
            * get_dtype_size(
                param.dtype
                if communication_dtype == CommunicationDType.DEFAULT
                else dtype_mapping[communication_dtype.value]
            )
        )

    @property
    def on_source_rank(self) -> bool:
        return self._on_source_rank

    def preconditioned_grad_to_dist_buffer(self, grad: Tensor, iteration: int) -> None:
        if self._on_source_rank and self._dist_buffer is not None:
            self._dist_buffer.copy_(self.precondition(grad, iteration))

    def get_from_dist_buffer(self) -> Optional[Tensor]:
        # _dist_buffer must be returned regardless of the owner.
        # When this function is called, this buffer should be already filled
        # by a communication operation.
        return self._dist_buffer

    def get_split_dist_buffers(self) -> List[Tensor]:
        return [self._dist_buffer] if self._dist_buffer is not None else []

    def get_num_bytes(self, group_rank: int = -1) -> int:
        return (
            self._num_bytes
            if group_rank == -1 or self._group_source_rank == group_rank
            else 0
        )


class AdagradPreconditioner(DistributedPreconditioner):
    """Adagrad / Adam / RMSProp preconditioner for a generic layer.

    Stores preconditioner using same format as parameter p. Operations are performed in-place.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSProp, set beta2 = 0.999.
    To enable Adam, set beta2 = 0.999, use_bias_correction = True.

    Other variants can also be specified.

    Args:
        param (Tensor): Parameter of interest.
        beta1 (float): Exponential moving average factor for gradient. (Default: 0.0)
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)
        idx (Union[None, str, int]): Layer index (for logging purposes). (Default: None)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner. (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        beta1: float = 0.0,
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
        idx: Union[None, str, int] = None,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(AdagradPreconditioner, self).__init__(
            param,
            group,
            group_source_rank,
            dist_buffer,
            communication_dtype,
        )
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._preconditioner = allocate_distributed_tensor(
            param.shape,
            dtype=param.dtype,
            device=param.device,
            device_mesh_ranks=self._device_mesh_ranks,
            use_dtensor=use_dtensor,
        )
        self._idx = idx
        self._use_bias_correction = use_bias_correction
        self._bias_correction2 = 1.0
        self._parameter_count += self._preconditioner.numel()
        self._filtered_grad = (
            allocate_distributed_tensor(
                param.shape,
                dtype=param.dtype,
                device=param.device,
                device_mesh_ranks=self._device_mesh_ranks,
                use_dtensor=use_dtensor,
            )
            if self._beta1 != 0.0
            else None
        )

        if self._idx is not None:
            self._preconditioner_idx = str(self._idx) + "." + str(0)
            logger.info(
                f"Diagonal Adagrad Preconditioner {self._preconditioner_idx} with Parameter {self._idx}"
            )

    def update_preconditioners(self, grad: Tensor, iteration: int) -> None:
        if not self._on_source_rank:
            return

        preconditioner = use_local_tensor(self._preconditioner)

        if self._beta2 == 1.0:
            preconditioner.addcmul_(grad, grad, value=1)
        else:
            preconditioner.mul_(self._beta2).addcmul_(grad, grad, value=1 - self._beta2)

        if self._use_bias_correction and self._beta2 < 1.0:
            self._bias_correction2 = 1.0 - self._beta2**iteration

    def precondition(self, grad: Tensor, iteration: int) -> Tensor:
        if not self._on_source_rank:
            return grad

        with profiler.record_function("## adagrad:precondition ##"):
            if self._beta1 != 0.0:
                # Compute bias corrections.
                bias_correction1 = (
                    1.0 - self._beta1**iteration if self._use_bias_correction else 1.0
                )
                # Compute exponential moving average of the gradient (with potential bias correction).
                filtered_grad = use_local_tensor(self._filtered_grad)
                filtered_grad.mul_(self._beta1).add_(grad, alpha=1 - self._beta1)
                grad.copy_(filtered_grad / bias_correction1)

            denom = (
                (use_local_tensor(self._preconditioner) / self._bias_correction2)
                .sqrt()
                .add_(self._epsilon)
            )
            grad.div_(denom)
            return grad

    def compute_norm(self, grad: Tensor, iteration: int):
        if not self._on_source_rank:
            return torch.as_tensor(1.0)  # return cheap tensor

        denom = (
            (use_local_tensor(self._preconditioner) / self._bias_correction2)
            .sqrt()
            .add_(self._epsilon)
        )
        adagrad_nrm = torch.linalg.norm(grad / denom)
        return adagrad_nrm

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            self._preconditioner = self._preconditioner.to(device=device)

    def num_preconditioners(self) -> int:
        return 1


class ShampooKroneckerFactor(OptimizerModule):
    """Shampoo Kronecker Factor Matrix / Preconditioner data class."""

    def __init__(
        self,
        preconditioner_type: PreconditionerType,
        factor_matrix: Tensor,
        inv_factor_matrix: Optional[Tensor] = None,
        index: Optional[str] = None,
        is_diagonal: bool = True,
    ):
        super(ShampooKroneckerFactor, self).__init__()
        self.preconditioner_type = preconditioner_type
        self.factor_matrix = factor_matrix
        self.inv_factor_matrix = inv_factor_matrix
        self.index = index
        self.is_diagonal = is_diagonal


class ShampooPreconditioner(DistributedPreconditioner):
    """Shampoo preconditioners for some generic layer.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        beta1 (float): Exponential moving average factor for gradient. (Default: 0.0)
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        exponent_override (int, List[int]): inverse root to use in Shampoo. If a list [l1, l2, ..., lp], then we will
            use -1 / l1 for 1-D tensor (vectors), -1 / l2 for 2-D tensors (matrices), and so on. If the order of the
            tensor exceeds the length of the list, we revert to using the default value. If 0 is used, uses the default
            inverse root -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float): number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        diagonal_threshold (int): Threshold for using diagonal preconditioners. If None, disabled. (Default: None)
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        idx (Union[None, int, str]): Layer index (for logging purposes). (Default: None)
        start_preconditioning_step (int): initial delay before starting to compute root inverse. Applies grafting method
            beforehand. (default: 0)
        grafting_type (GraftingType): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner. (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. (Default: None)
        use_protected_eigh (bool): Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        beta1: float = 0.0,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        exponent_override: Union[int, List[int]] = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        diagonal_threshold: Union[None, int] = None,
        dtype: torch.dtype = torch.float,
        idx: Union[None, int, str] = None,
        start_preconditioning_step: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        use_protected_eigh: bool = True,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(ShampooPreconditioner, self).__init__(
            param,
            group,
            group_source_rank,
            dist_buffer,
            communication_dtype,
        )

        # Initialize parameters.
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._exponent_override = exponent_override
        self._exponent_multiplier = exponent_multiplier
        self._diagonal_threshold = diagonal_threshold
        self._dtype = dtype
        self._use_bias_correction = use_bias_correction
        self._bias_correction2 = 1.0
        self._order = param.dim()
        self._idx = idx
        self._grafting_type = grafting_type
        self._start_preconditioning_step = start_preconditioning_step
        self._use_protected_eigh = use_protected_eigh
        self._communication_dtype = communication_dtype
        self._filtered_grad = (
            allocate_distributed_tensor(
                param.shape,
                dtype=dtype,
                device=param.device,
                device_mesh_ranks=self._device_mesh_ranks,
                use_dtensor=use_dtensor,
            )
            if beta1 != 0.0
            else None
        )

        # Compute root.
        self._root = self._get_root_from_exponent_override(
            self._exponent_override, self._order
        )

        # Initialize lists for preconditioners, inverse preconditioners, types, and ranks.
        self._preconditioners = []

        for k, dim in enumerate(self._dims):
            index = str(self._idx) + "." + str(k) if self._idx else None

            # Creates a diagonal Shampoo preconditioner if dimension is larger than
            # self._diagonal_threshold.
            if self._diagonal_threshold is not None and dim > self._diagonal_threshold:
                preconditioner_type = PreconditionerType.DIAGONAL
                factor_matrix = allocate_distributed_tensor(
                    dim,
                    dtype=param.dtype,
                    device=param.device,
                    device_mesh_ranks=self._device_mesh_ranks,
                    use_dtensor=use_dtensor,
                )
                inv_factor_matrix = None
                num_params = dim

            # Otherwise, generates a full Shampoo preconditioner.
            else:
                preconditioner_type = PreconditionerType.FULL
                factor_matrix = allocate_distributed_tensor(
                    (dim, dim),
                    dtype=self._dtype,
                    device=param.device,
                    device_mesh_ranks=self._device_mesh_ranks,
                    use_dtensor=use_dtensor,
                )
                inv_factor_matrix = allocate_distributed_tensor(
                    (dim, dim),
                    dtype=self._dtype,
                    device=param.device,
                    device_mesh_ranks=self._device_mesh_ranks,
                    use_dtensor=use_dtensor,
                )
                num_params = 2 * dim**2

            # Counts parameters and adds to lists.
            if preconditioner_type == PreconditionerType.FULL:
                is_diagonal = False
            self._parameter_count += num_params
            self._num_bytes += num_params * get_dtype_size(dtype)
            self._preconditioners.append(
                ShampooKroneckerFactor(
                    preconditioner_type,
                    factor_matrix,
                    inv_factor_matrix,
                    index,
                    is_diagonal = is_diagonal,
                )
            )

        # Initialize grafting method.
        if self._grafting_type == GraftingType.NONE:
            self._grafting = None
        elif self._grafting_type == GraftingType.SGD:
            self._grafting = SGDGrafting(param)
        elif self._grafting_type == GraftingType.ADAGRAD:
            self._grafting = AdagradGrafting(
                param,
                epsilon=grafting_epsilon,
                group=group,
                group_source_rank=group_source_rank,
                dist_buffer=dist_buffer,
                use_dtensor=use_dtensor,
                communication_dtype=communication_dtype,
            )
        elif self._grafting_type == GraftingType.RMSPROP:
            self._grafting = RMSPropGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
                group=group,
                group_source_rank=group_source_rank,
                dist_buffer=dist_buffer,
                use_dtensor=use_dtensor,
                communication_dtype=communication_dtype,
            )
        elif self._grafting_type == GraftingType.ADAM:
            self._grafting = AdamGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
                group=group,
                group_source_rank=group_source_rank,
                dist_buffer=dist_buffer,
                use_dtensor=use_dtensor,
                communication_dtype=communication_dtype,
            )
        elif self._grafting_type == GraftingType.ADAGRAD_NORMALIZED:
            self._grafting = AdagradNormalizedGrafting(
                param,
                epsilon=grafting_epsilon,
                group=group,
                group_source_rank=group_source_rank,
                dist_buffer=dist_buffer,
                use_dtensor=use_dtensor,
                communication_dtype=communication_dtype,
            )
        elif self._grafting_type == GraftingType.RMSPROP_NORMALIZED:
            self._grafting = RMSPropNormalizedGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
                group=group,
                group_source_rank=group_source_rank,
                dist_buffer=dist_buffer,
                use_dtensor=use_dtensor,
                communication_dtype=communication_dtype,
            )
        elif self._grafting_type == GraftingType.ADAM_NORMALIZED:
            self._grafting = AdamNormalizedGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
                group=group,
                group_source_rank=group_source_rank,
                dist_buffer=dist_buffer,
                use_dtensor=use_dtensor,
                communication_dtype=communication_dtype,
            )
        elif self._grafting_type == GraftingType.LARS:
            self._grafting = SGDGrafting(param)
        elif self._grafting_type == GraftingType.LAMB:
            self._grafting = AdamGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
                group=group,
                group_source_rank=group_source_rank,
                dist_buffer=dist_buffer,
                use_dtensor=use_dtensor,
            )
        else:
            raise ValueError(f"Invalid Grafting Type {self._grafting_type}!")

        # Counts parameters for grafted method.
        self._parameter_count += getattr(self._grafting, "parameter_count", 0)
        self._num_bytes += getattr(self._grafting, "num_bytes", 0)

    def get_debug_info(self) -> Dict[str, str]:
        debug_info = super().get_debug_info()
        preconditioner_strs = []
        for preconditioner in self._preconditioners:
            matrix_sizes = [tuple(preconditioner.factor_matrix.size())] + (
                [tuple(preconditioner.inv_factor_matrix.size())]
                if preconditioner.inv_factor_matrix is not None
                else []
            )
            preconditioner_strs.append(
                f"{'Diagonal' if preconditioner.is_diagonal else 'Full'}({preconditioner.index}, [{', '.join(str(s) for s in matrix_sizes)}], {self._dtype})"
            )

        debug_info.update(
            {
                "name": "ShampooPreconditioner",
                "idx": str(self._idx),
                "preconditioners": f"[{', '.join(preconditioner_strs)}]",
            }
        )
        return debug_info

    @staticmethod
    def _get_root_from_exponent_override(
        exponent_override: Union[int, List[int]], order: int
    ) -> int:
        """Retrieves the appropriate root from the exponent override parameter.

        Args:
            exponent_override (int, List[int]): Exponent override int or list.
            order (List[int]): Order of the tensor of interest.

        Returns:
            root (int): Root to use in Shampoo.

        """
        if isinstance(exponent_override, list):
            if order > len(exponent_override):
                return 2 * order
            else:
                return exponent_override[order - 1]
        else:
            return 2 * order if exponent_override == 0 else exponent_override

    def update_preconditioners(self, grad: Tensor, iteration: int) -> None:
        if not self._on_source_rank:
            return

        with profiler.record_function("## shampoo:update_preconditioners ##"):
            for k, (dim, preconditioner) in enumerate(
                zip(self._dims, self._preconditioners)
            ):
                factor_matrix = use_local_tensor(preconditioner.factor_matrix)

                if self._beta2 != 1.0:
                    factor_matrix.mul_(self._beta2)

                # Update diagonal Shampoo preconditioner.
                if preconditioner.preconditioner_type == PreconditionerType.DIAGONAL:
                    diagonal_or_outer_product = torch.linalg.norm(
                        grad.transpose(0, k).contiguous().view(dim, -1),
                        dim=1,
                    ).pow(2)

                # Update full Shampoo preconditioner.
                else:
                    contract_idx = [*range(k)] + [*range(k + 1, self._order)]
                    diagonal_or_outer_product = torch.tensordot(
                        grad,
                        grad,
                        dims=(contract_idx, contract_idx),
                    )
                    if diagonal_or_outer_product.dtype != self._dtype:
                        diagonal_or_outer_product = diagonal_or_outer_product.to(
                            dtype=self._dtype
                        )

                factor_matrix.add_(
                    diagonal_or_outer_product,
                    alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
                )

            # Update grafting preconditioner.
            if self._grafting_type != GraftingType.NONE:
                self._grafting.update_preconditioners(grad, iteration)

            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = 1.0 - self._beta2**iteration

    def _shampoo_precondition(self, grad: Tensor, iteration: int) -> Tensor:
        if not self._on_source_rank:
            return grad  # An invalid tensor that can be returned at the lowest cost.

        preconditioned_grad = grad.clone()
        for k, preconditioner in enumerate(self._preconditioners):
            # Use local versions of factor and inv factor matrices.
            factor_matrix = use_local_tensor(preconditioner.factor_matrix)
            inv_factor_matrix = use_local_tensor(preconditioner.inv_factor_matrix)

            # To handle diagonal case, requires not transposing the tensor.
            if self._diagonal_threshold is not None:

                # Precondition using diagonal preconditioner.
                if preconditioner.preconditioner_type == PreconditionerType.DIAGONAL:
                    denom = (factor_matrix / self._bias_correction2).add_(self._epsilon)
                    preconditioned_grad.div_(
                        denom.pow(-self._exponent_multiplier / self._root)[
                            (None,) * k + (...,) + (None,) * (self._order - k - 1)
                        ]
                    )

                # Precondition using full Shampoo preconditioner.
                # Uses einsum in order to avoid transposing.
                else:
                    gradient_idx = [*range(1, self._order + 1)]
                    matrix_product_idx = deepcopy(gradient_idx)
                    matrix_product_idx[k] = 0
                    preconditioned_grad = torch.einsum(
                        inv_factor_matrix,
                        [0, k + 1],
                        preconditioned_grad,
                        gradient_idx,
                        matrix_product_idx,
                    )

            # Handles full Shampoo preconditioner case more efficiently but
            # transposes the tensor continually.
            else:
                preconditioned_grad = torch.tensordot(
                    preconditioned_grad, inv_factor_matrix, [[0], [0]]
                )

            if preconditioned_grad.dtype != grad.dtype:
                preconditioned_grad.to(dtype=grad.dtype)

        # Apply grafting.
        if self._grafting_type != GraftingType.NONE:
            grafting_norm = self._grafting.direction_norm(grad, iteration)
            shampoo_norm = torch.linalg.norm(preconditioned_grad)
            preconditioned_grad.mul_(grafting_norm).div_(shampoo_norm + 1e-16)

        return preconditioned_grad

    def _graft_precondition(self, grad: Tensor, iteration: int) -> Tensor:
        if not self._on_source_rank:
            return grad  # An invalid tensor that can be returned at the lowest cost.
        return (
            self._grafting.precondition(grad, iteration)
            if self._grafting_type != GraftingType.NONE
            else grad
        )

    def precondition(self, grad: Tensor, iteration: int) -> Tensor:
        if not self._on_source_rank:
            return grad  # An invalid tensor that can be returned at the lowest cost.

        use_graft_precondition = iteration < self._start_preconditioning_step
        if self._beta1 != 0.0:
            # Compute bias corrections.
            bias_correction1 = (
                1.0 - self._beta1**iteration if self._use_bias_correction else 1.0
            )
            # Compute exponential moving average of the gradient (with potential bias correction).
            filtered_grad = use_local_tensor(self._filtered_grad)
            filtered_grad.mul_(self._beta1).add_(grad, alpha=1 - self._beta1)
            grad.copy_(filtered_grad / bias_correction1)

        with profiler.record_function(
            "## shampoo:graft_precondition ##"
            if use_graft_precondition
            else "## shampoo:shampoo_precondition ##"
        ):
            return (
                self._graft_precondition
                if use_graft_precondition
                else self._shampoo_precondition
            )(grad, iteration)

    def compute_root_inverse(self) -> None:
        if not self._on_source_rank:
            return

        for k, preconditioner in enumerate(self._preconditioners):
            # Use local versions of factor and inv factor matrices.
            factor_matrix = use_local_tensor(preconditioner.factor_matrix)
            inv_factor_matrix = use_local_tensor(preconditioner.inv_factor_matrix)

            # Check that this is a full Shampoo preconditioner.
            if preconditioner.preconditioner_type == PreconditionerType.FULL:

                # For tracking diagonality of the preconditioner.
                # Checks if the preconditioner is currently diagonal, then checks whether or not
                # the update matrix is diagonal.
                if preconditioner.is_diagonal and not check_diagonal(factor_matrix):
                    preconditioner.is_diagonal = False
                    logger.debug(
                        f"Preconditioner {preconditioner.index} is not diagonal."
                    )

                # Add epsilon term and incorporate bias correction.
                bias_corrected_preconditioner = factor_matrix / self._bias_correction2

                # Check for nan or inf values.
                if torch.isnan(bias_corrected_preconditioner).any():
                    raise ValueError(
                        f"Encountered nan values in bias-corrected factor matrix {self._idx}.{k}! To mitigate, check if nan inputs are being passed into the network or nan gradients are being passed to the optimizer."
                        f"For debugging purposes, factor_matrix {self._idx}.{k}: {torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, {factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
                    )
                if torch.isinf(bias_corrected_preconditioner).any():
                    raise ValueError(
                        f"Encountered inf values in bias-corrected factor matrix {self._idx}.{k}! In some cases, this may be due to divergence of the algorithm. To mitigate, try decreasing the learning rate or increasing grafting epsilon."
                        f"For debugging purposes, factor_matrix {self._idx}.{k}: {torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, {factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
                    )

                # Compute inverse preconditioner.
                # If reuse_previous_inv_factor_matrix is True, will reuse previous matrix if matrix
                # inverse root computation fails.
                try:
                    computed_inv_factor_matrix = matrix_inverse_root(
                        A=bias_corrected_preconditioner,
                        root=self._root,
                        epsilon=self._epsilon,
                        exponent_multiplier=self._exponent_multiplier,
                        is_diagonal=preconditioner.is_diagonal,
                        retry_double_precision=self._use_protected_eigh,
                    ).to(dtype=self._dtype)

                    # Check if we encounter NaN or inf values in computed inverse matrix.
                    if (
                        torch.isnan(computed_inv_factor_matrix).any()
                        or torch.isinf(computed_inv_factor_matrix).any()
                    ):
                        torch.set_printoptions(threshold=100_000)
                        raise ValueError(
                            f"Encountered nan or inf values in inverse factor matrix {self._idx}.{k}! To mitigate, check factor matrix before matrix inverse root computation: {bias_corrected_preconditioner=}"
                        )

                    inv_factor_matrix.copy_(computed_inv_factor_matrix)

                except Exception as exception:
                    if (
                        not self._use_protected_eigh
                        or "Encountered nan or inf values in inverse factor matrix"
                        in str(exception)
                    ):
                        raise exception
                    else:
                        logger.warning(
                            f"Matrix inverse root computation failed for preconditioner {self._idx}.{k} with exception {exception}. Using previous inv_factor_matrix and continuing..."
                        )

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        relative_errors = []
        relative_residuals = []

        if not self._on_source_rank:
            return relative_errors, relative_residuals

        for preconditioner in self._preconditioners:
            if preconditioner.preconditioner_type == PreconditionerType.FULL:
                # Use local versions of factor and inv factor matrices.
                factor_matrix = use_local_tensor(preconditioner.factor_matrix)
                inv_factor_matrix = use_local_tensor(preconditioner.inv_factor_matrix)

                bias_corrected_preconditioner = factor_matrix / self._bias_correction2
                (
                    relative_error,
                    relative_residual,
                ) = compute_matrix_root_inverse_residuals(
                    bias_corrected_preconditioner,
                    inv_factor_matrix,
                    self._root,
                    self._epsilon,
                    self._exponent_multiplier,
                )
                relative_errors.append(relative_error)
                relative_residuals.append(relative_residual)

        return (
            relative_errors,
            relative_residuals,
        )

    def compute_norm(self, grad: Tensor, iteration: int) -> Tensor:
        return torch.linalg.norm(self.precondition(grad, iteration))

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            for preconditioner in self._preconditioners:
                preconditioner.to(device)
            if self._grafting is not None:
                self._grafting.to(device=device)

    def num_preconditioners(self) -> int:
        return 1

    def reset_preconditioners(self) -> None:
        for preconditioner in self._preconditioners:
            preconditioner.factor_matrix.zero_()


class BlockShampooPreconditioner(DistributedPreconditioner):
    """Shampoo with blocking applied to the parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        beta1 (float): Exponential moving average factor for gradient. (Default: 0.0)
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        exponent_override (int, List[int]): inverse root to use in Shampoo. If a list [l1, l2, ..., lp], then we will
            use -1 / l1 for 1-D tensor (vectors), -1 / l2 for 2-D tensors (matrices), and so on. If the order of the
            tensor exceeds the length of the list, we revert to using the default value. If 0 is used, uses the default
            inverse root -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float): number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        block_size (int): Block size for blocking large tensors. (Default: 1024)
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        idx (Union[None, int, str]): Layer index (for logging purposes). (Default: None)
        use_merge_dims (bool): Denotes whether or not dimensions are merged. (Default: True)
        cache_split_params (bool): cache split parameters across iterations. (Default: False)
        start_preconditioning_step (int): initial delay before starting to compute root inverse. Applies grafting method beforehand. (Default: 0)
        grafting_type (LayerwiseGraftingType): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        dist_buffer_ranks (Optional[List[Tuple[Tensor, int]]]): List of distributed buffers and their group rank
            assignments. (Default: None)
        dist_buffer_index (int): Index for getting dist_buffer and rank from dist_buffer and rank list.
            (Default: 0)
        use_protected_eigh (bool): Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        beta1: float = 0.0,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        exponent_override: Union[int, List[int]] = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        block_size: int = 1024,
        dtype: torch.dtype = torch.float,
        idx: Union[None, int, str] = None,
        use_merge_dims: bool = True,
        cache_split_params: bool = False,
        start_preconditioning_step: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
        group: Optional[dist.ProcessGroup] = None,
        dist_buffer_ranks: Optional[List[Tuple[Tensor, int]]] = None,
        dist_buffer_index: int = 0,
        use_protected_eigh: bool = True,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(BlockShampooPreconditioner, self).__init__(
            param,
        )

        # Set parameters.
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._exponent_override = exponent_override
        self._exponent_multiplier = exponent_multiplier
        self._use_bias_correction = use_bias_correction
        self._block_size = block_size
        self._dtype = dtype
        self._idx = idx
        self._start_preconditioning_step = start_preconditioning_step
        self._use_merge_dims = use_merge_dims
        self._cache_split_params = cache_split_params
        self._original_dims = list(param.size())
        self._merged_dims = (
            merge_small_dims(self._original_dims, self._block_size)
            if self._block_size is not None and use_merge_dims
            else self._original_dims
        )

        # Construct splits for blocking
        self._splits = [block_size] * len(self._merged_dims)
        self._num_splits = [math.ceil(dim / block_size) for dim in self._merged_dims]

        # Construct multiple preconditioners for each block
        self._split_preconditioners = []
        self._split_sizes = []
        self._cached_split_params = []

        split_param = self.combine_and_split_dims(param)
        for i, p in enumerate(split_param):
            self._split_sizes.append(torch.as_tensor(p.shape))
            split_idx = str(idx) + "." + str(i)
            dist_buffer, group_source_rank = (
                dist_buffer_ranks[dist_buffer_index + i]
                if dist_buffer_ranks is not None
                else (None, 0)
            )
            preconditioner = ShampooPreconditioner(
                p,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                exponent_override=exponent_override,
                exponent_multiplier=exponent_multiplier,
                use_bias_correction=use_bias_correction,
                dtype=dtype,
                idx=split_idx,
                start_preconditioning_step=start_preconditioning_step,
                grafting_type=grafting_type,
                grafting_beta2=grafting_beta2,
                grafting_epsilon=grafting_epsilon,
                group=group,
                group_source_rank=group_source_rank,
                dist_buffer=dist_buffer,
                use_protected_eigh=use_protected_eigh,
                use_dtensor=use_dtensor,
                communication_dtype=communication_dtype,
            )
            self._split_preconditioners.append(preconditioner)
            self._parameter_count += preconditioner.parameter_count

        self._block_count = len(self._split_preconditioners)
        # Initialize source rank based on whether or not any preconditioner is
        # on source rank.
        self._on_source_rank = any(
            preconditioner.on_source_rank
            for preconditioner in self._split_preconditioners
        )

    def get_debug_info(self) -> Dict[str, str]:
        debug_info = super().get_debug_info()
        split_preconditioner_strs = []
        for split_preconditioner in self._split_preconditioners:
            info = split_preconditioner.get_debug_info()
            split_preconditioner_strs.append(
                f"{info['name']}({', '.join([f'{key}={val}' for key, val in info.items() if key != 'name'])})"
            )

        debug_info.update(
            {
                "name": "BlockShampooPreconditioner",
                "idx": str(self._idx),
                "preconditioners": "[\n  "
                + (",\n  ".join(split_preconditioner_strs))
                + "\n]",
            }
        )
        return debug_info

    def combine_and_split_dims(self, p: Tensor) -> List[Tensor]:
        if self._use_merge_dims:
            p = p.view(self._merged_dims)
        return multi_dim_split(p, self._splits)

    def _multi_dim_cat(self, split_grads: List[Tensor]) -> Tensor:
        preconditioned_grad = multi_dim_cat(split_grads, self._num_splits)
        return (
            preconditioned_grad.view(self._original_dims)
            if self._use_merge_dims
            else preconditioned_grad
        )

    def update_preconditioners(self, grad: Tensor, iteration: int):
        split_grad = self.combine_and_split_dims(grad)
        assert (
            len(split_grad) == self.num_preconditioners()
        ), f"BlockShampooPreconditioner {self._idx} has {self.num_preconditioners()} preconditioners but grad was split into {len(split_grad)} blocks!"
        for block_preconditioner, block_grad in zip(
            self._split_preconditioners, split_grad
        ):
            block_preconditioner.update_preconditioners(block_grad, iteration)

    def precondition(self, grad: Tensor, iteration: int) -> Tensor:
        split_grad = self.combine_and_split_dims(grad)
        assert self.num_preconditioners() == len(
            split_grad
        ), f"BlockShampooPreconditioner {self._idx} has {self.num_preconditioners()} preconditioners but grad was split into {len(split_grad)} blocks!"
        split_preconditioned_grad = [
            p.precondition(g, iteration)
            for p, g in zip(self._split_preconditioners, split_grad)
        ]
        preconditioned_grad = multi_dim_cat(split_preconditioned_grad, self._num_splits)
        return (
            preconditioned_grad.view(self._original_dims)
            if self._use_merge_dims
            else preconditioned_grad
        )

    def compute_root_inverse(self) -> None:
        for preconditioner in self._split_preconditioners:
            preconditioner.compute_root_inverse()

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        relative_errors = []
        relative_residuals = []

        for preconditioner in self._split_preconditioners:
            (
                relative_errors_temp,
                relative_residuals_temp,
            ) = preconditioner.compute_root_inverse_residuals()

            relative_errors += relative_errors_temp
            relative_residuals += relative_residuals_temp

        return (
            relative_errors,
            relative_residuals,
        )

    def compute_norm(self, grad: Tensor, iteration: int) -> Tensor:
        return torch.linalg.norm(self.precondition(grad, iteration))

    @staticmethod
    def get_dist_buffer_sizes(
        param,
        block_size: int,
        use_merge_dims: bool,
        communication_dtype: CommunicationDType,
    ) -> List[int]:
        original_dims = list(param.size())
        merged_dims = (
            merge_small_dims(original_dims, block_size)
            if block_size is not None and use_merge_dims
            else original_dims
        )
        splits = [block_size] * len(merged_dims)
        if use_merge_dims:
            param = param.view(merged_dims)
        return [
            ShampooPreconditioner.get_dist_buffer_size(split_param, communication_dtype)
            for split_param in multi_dim_split(param, splits)
        ]

    def preconditioned_grad_to_dist_buffer(self, grad: Tensor, iteration: int) -> None:
        split_grads = self.combine_and_split_dims(grad)
        assert self.num_preconditioners() == len(
            split_grads
        ), f"BlockShampooPreconditioner {self._idx} has {self.num_preconditioners()} preconditioners but grad was split into {len(split_grads)} blocks!"
        for preconditioner, grad in zip(self._split_preconditioners, split_grads):
            preconditioner.preconditioned_grad_to_dist_buffer(grad, iteration)

    def get_from_dist_buffer(self) -> Tensor:
        split_grads = [
            preconditioner._dist_buffer
            for preconditioner in self._split_preconditioners
        ]
        return self._multi_dim_cat(split_grads)

    def get_split_dist_buffers(self) -> List[Tensor]:
        return [
            preconditioner._dist_buffer
            for preconditioner in self._split_preconditioners
        ]

    def num_preconditioners(self) -> int:
        return len(self._split_preconditioners)

    def get_split_parameters(self, param: Tensor) -> List[Tensor]:
        if self._cache_split_params:
            if (
                len(self._cached_split_params) == 0
                or self._cached_split_params[0].storage().data_ptr()
                != param.storage().data_ptr()
            ):
                # Cache new split parameters.
                self._cached_split_params = self.combine_and_split_dims(param)
            return self._cached_split_params
        else:
            return self.combine_and_split_dims(param)

    def reset_preconditioners(self) -> None:
        for preconditioner in self._split_preconditioners:
            preconditioner.reset_preconditioners()

    def get_num_bytes(self, group_rank: int = -1) -> int:
        return sum(
            [
                preconditioner.get_num_bytes(group_rank=group_rank)
                for preconditioner in self._split_preconditioners
            ]
        )


###### GRAFTING CLASSES ######
class Grafting(OptimizerModule):
    """Grafting base class.

    We graft the method by storing and maintaining the preconditioner for the grafted method.
    Therefore, any additional modifications including gradient EMA/filtering and momentum are
    not included in grafting.

    Args:
        param (Tensor): Parameter of interest.

    """

    def __init__(
        self,
        param: Tensor,
    ):
        super(Grafting, self).__init__()
        self._parameter_count = 0
        self._num_bytes = 0

    def update_preconditioners(self, grad: Tensor, iteration: int):
        pass

    def precondition(self, grad: Tensor, iteration: int) -> Tensor:
        pass

    def direction_norm(self, grad: Tensor, iteration: int) -> Tensor:
        pass

    @property
    def parameter_count(self):
        return self._parameter_count

    @property
    def num_bytes(self) -> int:
        return self._num_bytes


class SGDGrafting(Grafting):
    """SGD grafting.

    Grafts the stochastic gradient method by returning the norm of the gradient.

    Args:
        param (Tensor): Parameter of interest.


    """

    def __init__(
        self,
        param: Tensor,
    ):
        super(SGDGrafting, self).__init__(param)

    def precondition(self, grad: Tensor, iteration: int) -> Tensor:
        return grad

    def direction_norm(self, grad: Tensor, iteration: int) -> Tensor:
        return torch.linalg.norm(grad)


class AdagradGrafting(Grafting):
    """Adagrad grafting.

    Supports RMSProp and Adam by determining beta2 and use_bias_correction.

    Note: beta1 is not included since that is shared between both Shampoo and the grafted optimizer.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)
        normalize_gradient (bool): Flag for normalizing the gradient. (Default: False)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner. (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param: Tensor,
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
        normalize_gradient: bool = False,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(AdagradGrafting, self).__init__(param)
        self._preconditioner = AdagradPreconditioner(
            param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=use_bias_correction,
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
            use_dtensor=use_dtensor,
            communication_dtype=communication_dtype,
        )
        self.normalize_gradient = normalize_gradient
        self._parameter_count += self._preconditioner.parameter_count
        self._num_bytes += self._preconditioner.get_num_bytes()

    def _normalize_grad(self, grad: Tensor) -> Tensor:
        return grad / torch.norm(grad) if self.normalize_gradient else grad

    def update_preconditioners(self, grad: Tensor, iteration: int):
        self._preconditioner.update_preconditioners(
            self._normalize_grad(grad), iteration
        )

    def precondition(self, grad: Tensor, iteration: int) -> Tensor:
        return self._preconditioner.precondition(grad, iteration)

    def direction_norm(self, grad: Tensor, iteration: int) -> Tensor:
        return self._preconditioner.compute_norm(grad, iteration)


class RMSPropGrafting(AdagradGrafting):
    """RMSProp grafting.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner. (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(RMSPropGrafting, self).__init__(
            param=param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=False,
            normalize_gradient=False,
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
            use_dtensor=use_dtensor,
            communication_dtype=communication_dtype,
        )


class AdamGrafting(AdagradGrafting):
    """Adam grafting.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner. (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(AdamGrafting, self).__init__(
            param=param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=True,
            normalize_gradient=False,
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
            use_dtensor=use_dtensor,
            communication_dtype=communication_dtype,
        )


class AdagradNormalizedGrafting(AdagradGrafting):
    """Adagrad grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner. (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        epsilon: float = 1e-10,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(AdagradNormalizedGrafting, self).__init__(
            param=param,
            beta2=1.0,
            epsilon=epsilon,
            use_bias_correction=False,
            normalize_gradient=True,
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
            use_dtensor=use_dtensor,
            communication_dtype=communication_dtype,
        )


class RMSPropNormalizedGrafting(AdagradGrafting):
    """RMSProp grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner. (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(RMSPropNormalizedGrafting, self).__init__(
            param=param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=False,
            normalize_gradient=True,
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
            use_dtensor=use_dtensor,
            communication_dtype=communication_dtype,
        )


class AdamNormalizedGrafting(AdagradGrafting):
    """Adam grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)
        group (Optional[dist.ProcessGroup]): Process group for distributed computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner. (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication between ranks. (Default: DEFAULT)

    """

    def __init__(
        self,
        param,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        group: Optional[dist.ProcessGroup] = None,
        group_source_rank: int = 0,
        dist_buffer: Optional[Tensor] = None,
        use_dtensor: bool = True,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ):
        super(AdamNormalizedGrafting, self).__init__(
            param=param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=True,
            normalize_gradient=True,
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
            use_dtensor=use_dtensor,
            communication_dtype=communication_dtype,
        )
