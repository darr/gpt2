#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : model_lm_head.py
# Create date : 2019-03-20 18:28
# Modified date : 2019-03-23 13:47
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from pybase import pylog

class GPT2LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
#        pylog.info("embed_shape :%s" % embed_shape.__str__())
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
#        pylog.info("hidden_state size:%s" % hidden_state.size().__str__())
        lm_logits = self.decoder(hidden_state)
#        pylog.info("lm_logits:%s" % lm_logits.size().__str__())
        return lm_logits
