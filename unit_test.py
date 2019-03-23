#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : unit_test.py
# Create date : 2018-10-07 10:58
# Modified date : 2019-03-22 11:04
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
#sys.path.append("..")
sys.path.append("./openai_gpt")
sys.path.append("./openai_gpt2")
from pybase import pylinux

from auto_test import rocstories_dataset_test

def show_env():
    print(pylinux.get_system_name_version())
    print(pylinux.get_platform_unname()[3])
    print(pylinux.get_architecture())
    print("Python:%s" % sys.version)

def getAllModule():
    module_str = os.popen("pip list").read()
    lt = module_str.split('\n')
    for i in lt:
        print(i)

def run():
    show_env()
    getAllModule()
    #rocstories_dataset_test.test()

if __name__ == "__main__":
    run()
