#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : etc_model_dict.py
# Create date : 2019-03-20 13:16
# Modified date : 2019-03-23 11:46
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function


def add_url(dic, url_head):
    dic["url"] = "%s%s" % (url_head,dic["file_name"])

cache_dict = {}

model_url = "http://lzygzh-low.oss-cn-beijing.aliyuncs.com/deep_model/nlp/openai-gpt2/"

config_dict = {}
config_dict["file_name"] = "gpt2-config.json"
add_url(config_dict, model_url)

model_dict = {}
model_dict["file_name"] = "gpt2-pytorch_model.bin"
add_url(model_dict, model_url)

merges_dict = {}
merges_dict["file_name"] = "gpt2-merges.txt"
add_url(merges_dict, model_url)

vocab_dict = {}
vocab_dict["file_name"] = "gpt2-vocab.json"
add_url(vocab_dict, model_url)

data_dict = {}
data_url = "http://lzygzh-low.oss-cn-beijing.aliyuncs.com/dataset/nlp/"
data_dict["name"] = "ROCStories"
data_dict["file_name"] = "%s.tar.gz" % data_dict["name"]
add_url(data_dict, data_url)

cache_dict["config"] = config_dict
cache_dict["merges"] = merges_dict
cache_dict["model"] = model_dict
cache_dict["vocab"] = vocab_dict
cache_dict["data"] = data_dict

for item in cache_dict:
    print(cache_dict[item])
