# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
from functools import partial
import torch
from torch import optim as optim


sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from myoptim import LocalOptimizer_GGT,  MyRmsProp

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.utils.shampoo_utils import GraftingType,LargeDimMethod

def build_optimizer(config, model, simmim=False, is_pretrain=False):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}

    assert simmim is False
    if simmim:
        if is_pretrain:
            parameters = get_pretrain_param_groups(model, skip, skip_keywords)
        else:
            depths = (
                config.MODEL.SWIN.DEPTHS
                if config.MODEL.TYPE == "swin"
                else config.MODEL.SWINV2.DEPTHS
            )
            num_layers = sum(depths)
            get_layer_func = partial(
                get_swin_layer, num_layers=num_layers + 2, depths=depths
            )
            scales = list(
                config.TRAIN.LAYER_DECAY**i for i in reversed(range(num_layers + 2))
            )
            parameters = get_finetune_param_groups(
                model,
                config.TRAIN.BASE_LR,
                config.TRAIN.WEIGHT_DECAY,
                get_layer_func,
                scales,
                skip,
                skip_keywords,
            )
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)

    print("------------------------------------------")
    print(
        config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, config.TRAIN.OPTIMIZER.MOMENTUM
    )
    print("------------------------------------------")

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == "sgd":
        print('using sgd')
        optimizer = optim.SGD(
            parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    elif opt_lower == "adamw":
        print("use AdamW damping:", config.TRAIN.OPTIMIZER.DAMPING, )
        print("use betas:", (config.TRAIN.OPTIMIZER.MOMENTUM, 1.0-config.TRAIN.OPTIMIZER.LR_COV) )
        print("use lr:", config.TRAIN.BASE_LR )
        print("use wt:", config.TRAIN.WEIGHT_DECAY )
        optimizer = optim.AdamW(
            parameters,
            eps=config.TRAIN.OPTIMIZER.DAMPING,
            betas=(config.TRAIN.OPTIMIZER.MOMENTUM, 1.0-config.TRAIN.OPTIMIZER.LR_COV),
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    elif opt_lower.find( "ifshampoo" ) >=0:
        print('using IF-Shampoo')
        assert config.TRAIN.OPTIMIZER.STRUCTURE == 'dense'
        optimizer = LocalOptimizer_GGT(
            model,
            lr=config.TRAIN.BASE_LR,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            damping=config.TRAIN.OPTIMIZER.DAMPING,
            beta2=config.TRAIN.OPTIMIZER.BETA2,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            T=config.TRAIN.OPTIMIZER.T,
            lr_cov=config.TRAIN.OPTIMIZER.LR_COV,
            batch_size=config.DATA.BATCH_SIZE,
            cast_dtype=torch.bfloat16,
        )


    elif opt_lower.find( "shampoo" ) >=0:
        print('using shampoo', config.TRAIN.BASE_LR, config.TRAIN.OPTIMIZER.MOMENTUM, config.TRAIN.OPTIMIZER.LR_COV,
                config.TRAIN.OPTIMIZER.DAMPING, config.TRAIN.WEIGHT_DECAY, config.TRAIN.OPTIMIZER.T)
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=config.TRAIN.BASE_LR,
            betas=(config.TRAIN.OPTIMIZER.MOMENTUM, 1.0-config.TRAIN.OPTIMIZER.LR_COV),
            epsilon=config.TRAIN.OPTIMIZER.DAMPING,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            max_preconditioner_dim=102400,
            precondition_frequency=config.TRAIN.OPTIMIZER.T,
            use_decoupled_weight_decay=True,
            grafting_type=GraftingType.NONE,
            use_bias_correction = False,
            num_trainers_per_group=1,
            preconditioner_dtype = torch.float64,
            large_dim_method= LargeDimMethod.ADAGRAD,
        )

    elif opt_lower.startswith("rfrmsprop"):
        print('using rfrmsprop')
        dtype = torch.bfloat16
        optimizer = MyRmsProp(
            model.parameters(),
            lr=config.TRAIN.BASE_LR,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            eps=config.TRAIN.OPTIMIZER.DAMPING,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            alpha= 1.0 - config.TRAIN.OPTIMIZER.LR_COV,
            batch_averaged=True,
            cast_dtype=dtype,
            batch_size=config.DATA.BATCH_SIZE,
            dummy_init = config.TRAIN.OPTIMIZER.DUMMY_INIT,
            dummy_scaling = config.TRAIN.OPTIMIZER.DUMMY_SCALING,
        )

    else:
        raise NotImplementedError


    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        has_decay.append(param)
    return [
        {"params": has_decay},
    ]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split(".")[1])
        block_id = name.split(".")[3]
        if block_id == "reduction" or block_id == "norm":
            return sum(depths[: layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(
    model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()
):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())
