#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : func.py
# Create date : 2019-03-15 22:03
# Modified date : 2019-03-20 15:01
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import torch

def _set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

def _create_output_dir(config):
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def init_app(config):
    seed = config["seed"]
    _set_random_seed(seed)
    _create_output_dir(config)

def get_output_model_file_full_path(config):
    return os.path.join(config["output_dir"], "pytorch_model.bin")

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
