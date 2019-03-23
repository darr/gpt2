#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : gpt2_lm_head_model.py
# Create date : 2019-03-05 23:07
# Modified date : 2019-03-23 13:45
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from pybase import pylog

from .gpt2.base_gpt.model_lm_head import GPT2LMHead
from .gpt2.pretrained_model import GPT2PreTrainedModel
from .gpt2.gpt2_model import GPT2Model

class GPT2LMHeadModel(GPT2PreTrainedModel):
    """OpenAI GPT-2 model with a Language Modeling head ("Language Models are Unsupervised Multitask Learners").

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, config.vocab_size[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs:
        if `lm_labels` is not `None`:
            Outputs the language modeling loss.
        else a tuple:
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, sequence_length, config.vocab_size]
                (or more generally [d_1, ..., d_n, config.vocab_size] were d_1 ... d_n are the dimension of input_ids)
            `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2LMHeadModel(config)
    lm_logits, presents = model(input_ids)
    ```
    """

    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        #pylog.info("len presents:%s item size:%s" % (len(presents), presents[0].size().__str__()))
        #pylog.info("hidden_states.size:%s" % hidden_states.size().__str__())
        lm_logits = self.lm_head(hidden_states)
        #pylog.info("lm_logits.size:%s" % lm_logits.size().__str__())
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents
