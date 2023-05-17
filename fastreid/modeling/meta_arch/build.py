# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import shutil
import inspect
import torch

from fastreid.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    shutil.copy(inspect.getfile(META_ARCH_REGISTRY.get(meta_arch)), cfg.OUTPUT_DIR)
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    # model.to(torch.device(cfg.MODEL.DEVICE))
    model.cuda()
    return model
