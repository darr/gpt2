#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : __init__.py
# Create date : 2019-03-15 14:45
# Modified date : 2019-03-23 10:49
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from .gpt.gpt_token import OpenAIGPTTokenizer
from .gpt.gpt_opt import get_gpt_optimizer

from .gpt_double_heads_model import OpenAIGPTDoubleHeadsModel
from .gpt_lm_head_model import OpenAIGPTLMHeadModel
