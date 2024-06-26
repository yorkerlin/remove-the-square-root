""" ConvMixer

"""
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d

# from ._registry import register_model, generate_default_cfgs
from timm.models._builder import build_model_with_cfg

# from ._manipulate import checkpoint_seq

__all__ = ["convmixer_180_16"]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerNew(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        kernel_size=9,
        patch_size=7,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        drop_rate=0.0,
        act_layer=nn.GELU,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.grad_checkpointing = False

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            act_layer(),
            nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            # nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                            nn.Conv2d(
                                dim,
                                dim,
                                kernel_size,
                                groups=dim,
                                padding=int((kernel_size - 1) / 2),
                            ),
                            act_layer(),
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    act_layer(),
                    nn.BatchNorm2d(dim),
                )
                for i in range(depth)
            ]
        )
        self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r"^stem", blocks=r"^blocks\.(\d+)")
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x):
        x = self.stem(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        # x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.pooling(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_convmixer(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ConvMixerNew, variant, pretrained, **kwargs)


def convmixer_180_16(pretrained=False, **kwargs) -> ConvMixerNew:
    model_args = dict(dim=180, depth=16, kernel_size=7, patch_size=7, **kwargs)
    return _create_convmixer("convmixer_180_16", pretrained, **model_args)
