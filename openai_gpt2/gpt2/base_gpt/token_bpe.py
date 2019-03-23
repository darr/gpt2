#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : token_bpe.py
# Create date : 2019-03-21 15:56
# Modified date : 2019-03-21 15:58
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def bpe(token, cache, bpe_ranks):
    if token in cache:
        return cache[token]
    word = tuple(token)
    pairs = get_pairs(word)

    if not pairs:
        return token

    while True:
        bigram = min(pairs, key = lambda pair: bpe_ranks.get(pair, float('inf')))
        if bigram not in bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = ' '.join(word)
    cache[token] = word
    return word
