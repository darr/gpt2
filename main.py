#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-03-20 20:40
# Modified date : 2019-03-23 20:28
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import openai_gpt2.run_test
import openai_gpt.run_test

from pybase import pylog
from etc import config
import show
import train_graph
import eval_graph
import down_cache

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
    model = None
    train_loss, model_config, model = train_graph.do_train(config, model)
    eval_loss, eval_accuracy = eval_graph.do_eval(model_config, config)
    show.show_result_detail(eval_loss, eval_accuracy, train_loss, config)

def run_finetuned20(config):
    _set_config_files(config)
    model = None
    epochs_count = 0
    while True:
        train_loss, model_config, model = train_graph.do_train(config, model)
        epochs_count +=1
        eval_loss, eval_accuracy = eval_graph.do_eval(model_config, config)
        pylog.info(epochs_count)
        show.show_result_detail(eval_loss, eval_accuracy, train_loss, config)
        if epochs_count > 20:
            break

if __name__ == '__main__':
    openai_gpt2.run_test.run()
    openai_gpt.run_test.run()

    #run_directly(config["gpt_config"])
    #run_directly(config["gpt2_config"])

    run_finetuned(config["gpt_config"])
    run_finetuned(config["gpt2_config"])

    #run_finetuned20(config["gpt2_config"])
