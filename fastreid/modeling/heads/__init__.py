# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import REID_HEADS_REGISTRY, build_heads

# import all the meta_arch, so they will be registered
from .embed_head import EmbedHead
from .embedding_head import EmbeddingHead
from .attr_head import AttrHead
from .mlt_head import MultiHead
from .mlt_head_v2 import MultiHeadV2
