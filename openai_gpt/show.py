#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : show.py
# Create date : 2019-03-15 17:20
# Modified date : 2019-03-20 15:08
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import torch
from pybase import pylog

def show_model_paramter_size(state_dict):
    count = 0
    for key in state_dict:
        pylog.info(key)
        pylog.info(state_dict[key].size())
        size_lt = state_dict[key].size()
        size_num = 1
        for i in size_lt:
            size_num *= i
        count += size_num
        pylog.info(size_num)

    pylog.info("paramete count:%d" % count)
    pylog.info("paramete count :%d bytes" % (count * 4 ))
    pylog.info("paramete count :%d M" % ((count * 4) / (1024* 1024)))

def show_devices(config):
    device = config["device"]
    n_gpu = torch.cuda.device_count()
    pylog.info("device:%s, number of gpus:%s"% (device, n_gpu))

def _show_result(result, config):
    output_eval_file = os.path.join(config["output_dir"], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        pylog.info("***** Eval results *****")
        for key in sorted(result.keys()):
            pylog.info("%s = %s" %  (key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))

def show_result_detail(eval_loss, eval_accuracy, train_loss, config):
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'train_loss': train_loss}
    _show_result(result, config)
