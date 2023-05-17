# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .mgn import MGN
from .BSl import BSL
from .CAMA import CAMA
from .OAP import OAP
from .OAP2 import OAP2
from .GSS import GSS
from .SDC import SDC
