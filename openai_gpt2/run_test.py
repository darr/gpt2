#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : run_test.py
# Create date : 2019-03-20 20:40
# Modified date : 2019-03-23 20:06
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from .etc import config
from . import write_story

from . import down_cache

def _set_config_files(config):
    vocab_file = down_cache.get_model_file_path("vocab", config)
    config["vocab_file"] = vocab_file
    merges_file = down_cache.get_model_file_path("merges", config)
    config["merges_file"] = merges_file

def run_model_with_teiminal():
    _set_config_files(config)
    model, tokenizer = write_story.init_and_get_model_tokenizer(config)
    while True:
        raw_text = write_story.get_text_from_terminal()
        write_story.read_text_and_deal(raw_text, model, tokenizer, config)

def run_model():
    raw_text = "I love my motherland."
    _set_config_files(config)
    write_story.continue_to_write_sotry(raw_text, config)

def run():
    down_cache.down_gpt2_model_cache_files(config)
    run_model()
    #run_model_with_teiminal()
