#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : write_story.py
# Create date : 2019-03-21 15:22
# Modified date : 2019-03-22 18:00
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import task_func

def _check_config_length(model, config):
    if config["length"] == -1:
        config["length"] = model.config.n_ctx // 2
    elif config["length"] > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

def _check_config_batch_size(config):
    if config["batch_size"] == -1:
        config["batch_size"] = 1
    assert config["nsamples"] % config["batch_size"] == 0

def init_and_get_model_tokenizer(config):
    _check_config_batch_size(config)
    task_func.init_random(config)
    model, tokenizer = task_func.get_model_and_tokenizer(config)
    _check_config_length(model, config)
    return model, tokenizer

def read_text_and_deal(raw_text, model, tokenizer, config):
    context_tokens = tokenizer.encode(raw_text)
    task_func.generate_result(context_tokens, model, tokenizer, config)

def get_text_from_terminal():
    raw_text = input("Model prompt >>> ")
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input("Model prompt >>> ")
    return raw_text

def continue_to_write_sotry(raw_text, config):
    model, tokenizer = init_and_get_model_tokenizer(config)
    read_text_and_deal(raw_text, model, tokenizer, config)
