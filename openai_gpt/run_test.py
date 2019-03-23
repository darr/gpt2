#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : run_test.py
# Create date : 2019-03-16 12:23
# Modified date : 2019-03-23 16:53
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from pybase import pylog
from .etc import config
from . import show
from . import train_graph
from . import eval_graph
from .down_cache import down_gpt_model_cache_files
from . import down_cache

def _set_config_files(config):
    vocab_file = down_cache.get_model_file_path("vocab", config)
    config["vocab_file"] = vocab_file
    merges_file = down_cache.get_model_file_path("merges", config)
    config["merges_file"] = merges_file
    if config["model_name"] == "gpt":
        en_spacy_path = down_cache.get_spacy_file_path("spacy", config)
        config["en_spacy_path"] = en_spacy_path
    data_file= down_cache.get_dataset_file_path("data", config)
    config["data_file"] = data_file

def run_directly(config):
    _set_config_files(config)
    train_loss, model_config = train_graph.direct_save(config)
    eval_loss, eval_accuracy = eval_graph.do_eval(model_config, config)
    show.show_result_detail(eval_loss, eval_accuracy, train_loss, config)

def run_finetuned(config):
    _set_config_files(config)
    train_loss, model_config = train_graph.do_train(config)
    eval_loss, eval_accuracy = eval_graph.do_eval(model_config, config)
    show.show_result_detail(eval_loss, eval_accuracy, train_loss, config)

def run():
    down_gpt_model_cache_files(config)
    show.show_devices(config)
#    run_directly(config)
    run_finetuned(config)
