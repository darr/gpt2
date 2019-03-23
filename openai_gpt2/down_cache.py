#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : down_cache.py
# Create date : 2019-03-17 17:50
# Modified date : 2019-03-23 21:19
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import tarfile

from pybase import pycache
from pybase import pylog

def _download_file_from_web(url, cache_dir):
    return pycache.download_file_from_web(url, cache_dir)

def _get_model_file(url, cache_dir):
    is_cached, cache_path = _download_file_from_web(url, cache_dir)
    if not is_cached:
        pass
    return cache_path

def _get_file_path(url):
    _,file_path = pycache._get_file_path(url)
    file_path = "/".join(file_path.split("/")[:-1])
    return file_path

def _get_url_and_cache_dir(name, config):
    cache_dict = config["cache_dict"]
    dic = cache_dict[name]
    url = dic["url"]
    file_path = _get_file_path(url)
    cache_dir = config["data_dir"]
    cache_dir = "%s/%s" % (cache_dir, file_path)
    return url, cache_dir

def get_model_file_path(name, config):
    url, cache_dir = _get_url_and_cache_dir(name, config)
    file_path = _get_model_file(url, cache_dir)
    return file_path

def down_gpt2_model_cache_files(config):
    name = "config"
    file_path = get_model_file_path(name, config)
    name = "merges"
    file_path = get_model_file_path(name, config)
    name = "model"
    file_path = get_model_file_path(name, config)
    name = "vocab"
    file_path = get_model_file_path(name, config)
#   cache_dict = config["cache_dict"]
#   for name in cache_dict:
#       file_path = get_model_file_path(name, config)
#       print(file_path)
