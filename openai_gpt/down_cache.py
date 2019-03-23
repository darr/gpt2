#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : down_cache.py
# Create date : 2019-03-17 17:50
# Modified date : 2019-03-23 18:29
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

def _get_file_path(url):
    _,file_path = pycache._get_file_path(url)
    file_path = "/".join(file_path.split("/")[:-1])
#    pylog.info(file_path)
    return file_path

def _get_url_and_cache_dir(name, config):
    cache_dict = config["cache_dict"]
    dic = cache_dict[name]
    url = dic["url"]
    file_path = _get_file_path(url)
    cache_dir = config["data_dir"]
    cache_dir = "%s/%s" % (cache_dir, file_path)
    return url, cache_dir

def _check_and_decompression(filename, cache_dir):
    _check_and_untar(filename, cache_dir)

def _check_and_untar(filename, cache_dir):
    name_lt = filename.split('.')
    if name_lt[-1] == "gz" and name_lt[-2] == 'tar':
        cache_path = os.path.join(cache_dir, filename)
        _untar(cache_path, cache_dir)

def _untar(filename, dirs):
    t = tarfile.open(filename)
    pylog.info("decompression %s" % filename)
    t.extractall(path=dirs)

def _install_spacy(cache_path):
    pylog.info(cache_path)
    cmd_str = "pip install %s" % cache_path
    pylog.info(cmd_str)
    os.system(cmd_str)

def _get_spacy_en_install_path(name, config):
    url, cache_dir = _get_url_and_cache_dir(name, config)
    spacy_dict = config["cache_dict"][name]
    env_path = os.path.dirname(os.__file__)
    install_path = "%s/site-packages/%s/%s" % (env_path, spacy_dict["name"], spacy_dict["name_ver"])
    pylog.info(install_path)
    return install_path

def _get_spacy(url, cache_dir):
    is_cached, cache_path = _download_file_from_web(url, cache_dir)
    if not is_cached:
        _install_spacy(cache_path)
    return cache_path

def _get_dataset(url, cache_dir):
    is_cached, cache_path = _download_file_from_web(url, cache_dir)
    if not is_cached:
        filename = pycache._get_file_name(url)
        _check_and_decompression(filename, cache_dir)
    return cache_path

def _get_model_file(url, cache_dir):
    is_cached, cache_path = _download_file_from_web(url, cache_dir)
    if not is_cached:
        pass
    return cache_path

def get_model_file_path(name, config):
    url, cache_dir = _get_url_and_cache_dir(name, config)
    file_path = _get_model_file(url, cache_dir)
    return file_path

def get_spacy_file_path(name, config):
    url, cache_dir = _get_url_and_cache_dir(name, config)
    file_path = _get_spacy(url, cache_dir)
    install_path = _get_spacy_en_install_path(name, config)
    return install_path

def get_dataset_file_path(name, config):
    url, cache_dir = _get_url_and_cache_dir(name, config)
    file_path = _get_dataset(url, cache_dir)
    return file_path

def down_gpt_model_cache_files(config):
    name = "config"
    file_path = get_model_file_path(name, config)
    name = "merges"
    file_path = get_model_file_path(name, config)
    name = "model"
    file_path = get_model_file_path(name, config)
    name = "vocab"
    file_path = get_model_file_path(name, config)
    name = "data"
    file_path = get_dataset_file_path(name, config)
    name = "spacy"
    file_path = get_spacy_file_path(name, config)
