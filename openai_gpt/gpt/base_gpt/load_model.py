#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : load_model.py
# Create date : 2019-03-19 10:32
# Modified date : 2019-03-20 14:37
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

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

def show_msg(class_name, missing_keys, unexpected_keys, error_msgs):
    if len(missing_keys) > 0:
        pylog.info("Weights of %s not initialized from pretrained model: %s"% (class_name, missing_keys))
    if len(unexpected_keys) > 0:
        pylog.info("Weights from pretrained model not used in %s: %s" % (class_name, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(class_name, "\n\t".join(error_msgs)))
