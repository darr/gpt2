#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : task_func.py
# Create date : 2019-03-22 17:48
# Modified date : 2019-03-23 19:49
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from pybase import pylog

from .gpt2.gpt2_token import GPT2Tokenizer
from .gpt2_lm_head_model import GPT2LMHeadModel
from .down_cache import get_model_file_path

def _top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def _change_context_to_tensor(start_token, context, config):
    batch_size=config["batch_size"]
    device=config["device"]
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    return context

def _get_log_probs(model, prev, past, config):
    temperature=config["temperature"]
    top_k=config["top_k"]

    logits, past = model(prev, past=past)
    logits = logits[:, -1, :] / temperature
    logits = _top_k_logits(logits, k=top_k)
    log_probs = F.softmax(logits, dim=-1)
    return log_probs, past

def _sample_sequence(model, context_tokens, tokenizer, config, sample=True):
    length=config["length"]
    context=context_tokens
    batch_size=config["batch_size"]
    temperature=config["temperature"]
    top_k=config["top_k"]

    start_token = None
    context = _change_context_to_tensor(start_token, context, config)
#    pylog.info(type(context))
#    pylog.info(context)

    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            log_probs, past = _get_log_probs(model, prev, past, config)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)

    return output

def _decode_input_text(out, context_tokens, tokenizer, config):
    input_lt = out[:,:len(context_tokens)].tolist()
    print("len input_lt:%s len context_tokens:%s" % (len(input_lt), len(context_tokens)))
    for i in range(config["batch_size"]):
        print("%s" % i)
        print("input text:")
        input_text = tokenizer.decode(input_lt[i])
        print(input_text)

def _decode_output_text(out, context_tokens, tokenizer, generated, config):
    out = out[:, len(context_tokens):].tolist()
    for i in range(config["batch_size"]):
        generated += 1
        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
        print("output text:")
        text = tokenizer.decode(out[i])
        print(text)
    return generated

def init_random(config):
    seed = config["seed"]
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def generate_result(context_tokens, model, tokenizer, config):
    generated = 0
    for _ in range(config["nsamples"] // config["batch_size"]):
        out = _sample_sequence(model=model, context_tokens=context_tokens, tokenizer=tokenizer, config=config)
        _decode_input_text(out, context_tokens, tokenizer, config)
        generated = _decode_output_text(out, context_tokens, tokenizer, generated, config)
    print("=" * 80)

def get_model_and_tokenizer(config):
    device = config["device"]
    tokenizer = GPT2Tokenizer.from_pretrained(config)
    config_file = get_model_file_path("config", config)
    model_file = get_model_file_path("model", config)
    model = GPT2LMHeadModel.from_pretrained(model_file, config_file)

    model.to(device)
    model.eval()
    return model, tokenizer
