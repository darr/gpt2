#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : model_multiple_choice_head.py
# Create date : 2019-03-16 15:53
# Modified date : 2019-03-20 17:04
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class OpenAIGPTMultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config):
        super(OpenAIGPTMultipleChoiceHead, self).__init__()
        self.n_embd = config.n_embd
        # self.multiple_choice_token = multiple_choice_token
        self.dropout = nn.Dropout2d(config.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
        self.linear = nn.Linear(config.n_embd, 1)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        # Classification logits
        # hidden_state (bsz, num_choices, seq_length, hidden_size)
        # mc_token_ids (bsz, num_choices)
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        # (bsz, num_choices, 1, hidden_size)
        multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        # (bsz, num_choices, hidden_size)
        multiple_choice_h = self.dropout(multiple_choice_h.transpose(1, 2)).transpose(1, 2)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        # (bsz, num_choices)
        return multiple_choice_logits
