# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 19:58:18 2025

@author: boisv
"""

from functools import lru_cache

# The maximum word length we are interested in
MAX_WORD_LENGTH = 21

# Default word list
WORD_LIST = 'xwordlist_sorted_trimmed.txt'

@lru_cache(maxsize=32)
def load_words(wordlist_file=WORD_LIST, max_word_length=MAX_WORD_LENGTH):
    words = []
    with open(wordlist_file, 'r') as fid:
        for line in fid:
            word, score = line.split(';')
            # Normalize to all uppercase words
            word = word.upper()
            if len(word) > max_word_length:
                continue
            words.append(word)
    return words