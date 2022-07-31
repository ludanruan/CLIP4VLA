# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, pdb
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from .file_utils import cached_path
from .until_module import PreTrainedModel
from .tokenization import END_TOKEN
from .module_clip import LayerNorm, QuickGELU

logger = logging.getLogger(__name__)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple #x:[L,N,d]
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]



class CrossModel_Clip(nn.Module):
    def __init__(self, 
                max_position_embeddings,
                hidden_size,
                type_vocab_size,
                num_hidden_layers,
                num_attention_heads,
                hidden_dropout_prob=0.1
                ):

        super(CrossModel_Clip, self).__init__()
       
        
        self.positional_embedding = nn.Parameter(torch.empty(max_position_embeddings, hidden_size))
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.dropout = nn.Dropout(hidden_dropout_prob)
     
        self.transformer = Transformer(
            width = hidden_size,
            layers = num_hidden_layers,
            heads = num_attention_heads,

        )
       
    @property
    def dtype(self):
        
        return self.state_dict()['transformer.resblocks.0.attn.in_proj_weight'].dtype
        
    def build_attention_mask(self, attention_mask):
        # attention_mask_cls = torch.ones((attention_mask.shape[0],1),device=attention_mask.device).to(dtype=attention_mask.dtype)
        # extended_attention_mask = torch.cat([attention_mask_cls,attention_mask], dim=1).unsqueeze(1)
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1)
            extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        else:
            extended_attention_mask = attention_mask
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        return extended_attention_mask
        
    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=False, input_ids=None):
 
        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        
        extended_attention_mask = self.build_attention_mask(attention_mask)
        concat_input = concat_input.type(self.dtype)  # [batch_size, n_ctx, d_model]
        pos_emd = self.positional_embedding[:concat_input.size(1), :].type(self.dtype)
        token_type_embeddings = self.token_type_embeddings(concat_type)
        concat_embed = concat_input + pos_emd + token_type_embeddings

        concat_embed = self.dropout(concat_embed)
        concat_embed = concat_embed.permute(1, 0, 2)  # NLD -> LND
        
        hidden = self.transformer(concat_embed, extended_attention_mask)
        hidden = hidden.permute(1, 0, 2)  # LND -> NLD
        pooled_output= hidden[:,0] 
        
        return hidden, pooled_output