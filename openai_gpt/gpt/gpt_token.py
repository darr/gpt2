#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : gpt_token.py
# Create date : 2019-03-20 11:14
# Modified date : 2019-03-23 11:09
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import unicode_literals

import json
import re
import sys
from io import open
from tqdm import tqdm
from pybase import pylog

from .base_gpt.token_basic import BasicTokenizer
from .base_gpt.token_bpe import bpe

def _add_to_split_tokens(cache, token, bpe_ranks, split_tokens):
    word = bpe(cache, token, bpe_ranks)
    word_lt = word.split(' ')
    split_tokens.extend(word_lt)
    #split_tokens.extend([t for t in bpe(self.cache, token.text.lower(), self.bpe_ranks).split(' ')])

def _get_max_len(max_len):
    if max_len is not None:
        return max_len
    else:
        return int(1e12)

def _get_encoder(vocab_file, config):
    f = json.load(open(vocab_file, encoding="utf-8"))
    return f

def _get_decoder(encoder):
    dic = {}
    for k, v in encoder.items():
        dic[v] = k
    return dic

def _get_merges(merges_file):
    f = open(merges_file, encoding='utf-8')
    con = f.read()
    f.close()
    merges = con.split('\n')[1:-1]
    merges = [tuple(merge.split()) for merge in merges]
    return merges

def _get_bpe_ranks(merges_file, config):
    merges = _get_merges(merges_file)
    dic = dict(zip(merges, range(len(merges))))
    return dic

def _get_special_tokens(special_tokens_lt, encoder):
    dic = {}
    for i, tok in enumerate(special_tokens_lt):
        dic[tok] = len(encoder) + i
    return dic

def _get_special_tokens_decoder(special_tokens):
    return {v : k for k, v in special_tokens.items()}

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

class OpenAIGPTTokenizer(object):
    """
    BPE tokenizer. Peculiarities:
        - lower case all inputs
        - uses SpaCy tokenizer and ftfy for pre-BPE tokenization if they are installed, fallback to BERT's BasicTokenizer if not.
        - argument special_tokens and function set_special_tokens:
            can be used to add additional symbols (ex: "__classify__") to a vocabulary.
    """
    @classmethod
    def from_pretrained(cls, config, *inputs, **kwargs):

        vocab_file = config["vocab_file"]
        merges_file = config["merges_file"]
        en_spacy_path = config["en_spacy_path"]

        tokenizer = cls(vocab_file, merges_file, en_spacy_path,config, *inputs, **kwargs)
        return tokenizer

    def __init__(self, vocab_file, merges_file, en_spacy_path, config, special_tokens_lt=None, max_len=None):

        self.cache = {}
        self.use_spacy = config["use_spacy"]
        self.nlp_type = ""

        self._load_tokenizer(special_tokens_lt, en_spacy_path, config)
        self.max_len = _get_max_len(max_len)

        self.encoder = _get_encoder(vocab_file, config)
        self.decoder = _get_decoder(self.encoder)

        self.bpe_ranks = _get_bpe_ranks(merges_file, config)

        self.set_special_tokens(special_tokens_lt)

    def _load_spacy_tokenizer(self, special_tokens_lt, en_spacy_path, config):
        try:
            import ftfy
            import spacy

            self.nlp = spacy.load(en_spacy_path, disable=['parser', 'tagger', 'ner', 'textcat'])
            self.fix_text = ftfy.fix_text
            self.nlp_type = "spacy"

        except Exception as err:
            pylog.error("Error:%s \nftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy." % err)
            self._load_bert_toenizer(special_tokens_lt)

    def _load_tokenizer(self, special_tokens_lt, en_spacy_path, config):
        if self.use_spacy:
            self._load_spacy_tokenizer(special_tokens_lt, en_spacy_path, config)
        else:
            self._load_bert_toenizer(special_tokens_lt)

    def _load_bert_toenizer(self, special_tokens_lt):
        self.nlp = BasicTokenizer(do_lower_case=True, never_split=special_tokens_lt if special_tokens_lt is not None else [])
        self.fix_text = None
        self.nlp_type = "bert"

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens_lt):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """

        if not special_tokens_lt:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return

        self.special_tokens = _get_special_tokens(special_tokens_lt, self.encoder)
        self.special_tokens_decoder = _get_special_tokens_decoder(self.special_tokens)

        if self.nlp_type == "bert":
            # Using BERT's BasicTokenizer: we can update the tokenizer
            self.nlp.never_split = special_tokens_lt
        pylog.info("Special tokens %s" % self.special_tokens)

    def tokenize(self, text):
        return self._get_split_tokens(text)

    def _get_split_tokens(self, text):
        if self.nlp_type == "bert":
            return self._get_split_tokens_with_bert(text)
        else:
            return self._get_split_tokens_with_spacy(text)

    def _get_split_tokens_with_bert(self, text):
        # Using BERT's BasicTokenizer
        split_tokens = []
        text = self.nlp.tokenize(text)
        for token in text:
            _add_to_split_tokens(self.cache, token, self.bpe_ranks, split_tokens)
        return split_tokens

    def _get_split_tokens_with_spacy(self, text):
        # Using SpaCy & ftfy (original tokenization process of OpenAI GPT)
        split_tokens = []
        text = self.nlp(text_standardize(self.fix_text(text)))
        for token in text:
            _add_to_split_tokens(self.cache, token.text.lower(), self.bpe_ranks, split_tokens)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))

        if len(ids) > self.max_len:
            pylog.warning( "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model (%s > %s). Running this"
                " sequence through the model will result in indexing errors" % (len(ids), self.max_len))

        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        """Converts a sequence of ids in a string."""
        tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        out_string = ''.join(tokens).replace('</w>', ' ').strip()
        if clean_up_tokenization_spaces:
            out_string = out_string.replace('<unk>', '')
            out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').replace(' ,', ','
                    ).replace(" n't", "n't").replace(" 'm", "'m").replace(" 're", "'re").replace(" do not", " don't"
                    ).replace(" 's", "'s").replace(" t ", "'t ").replace(" s ", "'s ").replace(" m ", "'m "
                    ).replace(" 've", "'ve")
        return out_string
