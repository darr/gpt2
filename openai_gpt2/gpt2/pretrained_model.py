#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pretrained_model.py
# Create date : 2019-03-05 22:52
# Modified date : 2019-03-22 21:53
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function
import os

import torch
import torch.nn as nn

from .base_gpt.model_base import BertLayerNorm as LayerNorm
from .model_config import GPT2Config
from pybase import pylog

def change_state_dict(state_dict):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    return state_dict

def _check_model(model, state_dict):
    start_model = model

    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    return start_model

def load_state_dict_to_model(module, state_dict, missing_keys, unexpected_keys, error_msgs, prefix=""):
    metadata = state_dict._metadata
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
    for name, child in module._modules.items():
        if child is not None:
            load_state_dict_to_model(child, state_dict, missing_keys, unexpected_keys, error_msgs, prefix + name + ".")

def load_model(module, state_dict, class_name, prefix=""):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    metadata = state_dict._metadata
    load_state_dict_to_model(module, state_dict, missing_keys, unexpected_keys, error_msgs, prefix="")
    show_msg(class_name, missing_keys, unexpected_keys, error_msgs)

def show_msg(class_name, missing_keys, unexpected_keys, error_msgs):
    if len(missing_keys) > 0:
        pylog.info("Weights of %s not initialized from pretrained model: %s"% (class_name, missing_keys))
    if len(unexpected_keys) > 0:
        pylog.info("Weights from pretrained model not used in %s: %s" % (class_name, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(class_name, "\n\t".join(error_msgs)))

class GPT2PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def set_tied(self):
        pass

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained_tf(cls, model_file, config_file, state_dict=None, cache_dir=None, *inputs, **kwargs):
        # Directly load from a TensorFlow checkpoint (stored as NumPy array)
        return load_tf_model.load_tf_weights_in_gpt2(model, model_file)

    @classmethod
    def from_pretrained(cls, model_file, config_file, state_dict=None, cache_dir=None, *inputs, **kwargs):

        model_config = GPT2Config.from_json_file(config_file)
        model = cls(model_config, *inputs, **kwargs)
        state_dict = torch.load(model_file, map_location='cpu' if not torch.cuda.is_available() else None)

        state_dict = change_state_dict(state_dict)
        start_model = _check_model(model, state_dict)

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        class_name = model.__class__.__name__
        load_model(start_model, state_dict, class_name, prefix="")
        model.set_tied()
        return model
