#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:17:46 2021

@author: Alex Boisvert
"""

import re
import itertools
import argparse
import logging
import json
from collections import Counter

# Global variables

# The number of results to report
NUM_RESULTS = 1000
# The minimum score in the word list
MIN_SCORE = 80
# The word list itself
WORD_LIST = 'xwordlist.dict'

# Make partitions of a string
# We will need to change this when it gets more advanced
def multiSlice(s,cutpoints):
    """
    Helper function for allPartitions
    """
    k = len(cutpoints)
    if k == 0:
        return [s]
    else:
        multislices = [s[:cutpoints[0]]]
        multislices.extend(s[cutpoints[i]:cutpoints[i+1]] for i in range(k-1))
        multislices.append(s[cutpoints[k-1]:])
        return multislices
 
def allPartitions(s, num=None):
    n = len(s)
    cuts = list(range(1,n))
    if num:
        num_arr = [num-1]
    else:
        num_arr = range(n)
    for k in num_arr:
        for cutpoints in itertools.combinations(cuts,k):
            yield multiSlice(s,cutpoints)
            
def input_to_regex(i, combine_letters=False):
    """
    Create a regular expression pattern from an input string
    This regex will just tell us if a word is a candidate for the pattern
    """
    # For now we allow uppercase, lowercase, periods and asterisks
    if re.match(r'[^A-Za-z\*\.]', i):
        logging.error(f"Input string {i} has bad characters")
        return None
    # Replace asterisks with ".*"
    i = i.replace('*', '.*')
    # Capital letter replacement is slightly complicated
    # The first occurrence is replaced with a `(.+)`
    # subsequent ones have to be replaced with appropriate backrefs
    capital_letters = re.findall(r'[A-Z]', i)
    # if we're combining letters we only do this
    # for letters that appear multiple times
    if combine_letters:
        capital_counter = Counter(capital_letters)
        capital_letters = [k for k,v in capital_counter.items() if v > 1]
    ctr = 1
    used_letters = dict()
    for c in capital_letters:
        # first replace one
        i = i.replace(c, '(.+)', 1)
        # then replace the rest
        i = i.replace(c, f'\\{ctr}')
        used_letters[c] = ctr
        ctr += 1
    # if we're combining letters, take the rest in groups
    other_letters = re.findall(r'[A-Z]+', i)
    for c in other_letters:
        i = i.replace(c, '(.+)')
        used_letters[c] = ctr
        ctr += 1
    i = '^' + i + '$'
    if not combine_letters:
        return i
    else:
        return i, used_letters


def split_input(i):
    """
    Split an input string along the split character
    
    Returns: list
    """
    return i.split(';')

def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def input_variables(inputs):
    """
    Determine all the variables in a list of inputs
    Returns: set
    """
    input_str = ''.join(inputs)
    return set(re.findall('[A-Z]', input_str))

def get_set_cover(inputs):
    """
    Find a minimal covering for our variables
    We just do this brute-force for now as there shouldn't be many
    
    Returns: cover, others (lists)
    """
    # Figure out all the variables in the inputs
    
    ps = powerset(inputs)
    allvars = input_variables(inputs)
    # Since the powerset is ordered by size we can just break when we find a match
    for myset in ps:
        myvars = input_variables(myset)
        if myvars == allvars:
            return set(myset), set(inputs).difference(myset)
    
# store a word and all its partitions 
class Word:
    def __init__(self, word, score, pattern):
        self.word = word
        self.pattern = pattern
        self.score = score
        #self.partitions = self.all_partitions()
        
    # Prints object information
    def __repr__(self):
        j = {'word': self.word, 'score': self.score, 'pattern': self.pattern}
        return f'Word({json.dumps(j)})'   
  
    # Prints readable form
    def __str__(self):
        return self.word
        
    def matches_pattern(self):
        # check that the word matches the pattern
        r = input_to_regex(self.pattern)
        return re.match(r, self.word) is not None
    
    def all_partitions(self):
        # return all the partitions of the word that match the pattern
        # get our regex, allowing for multiple letters at a time
        i, letter_map = input_to_regex(self.pattern, True)
        # keep just the letters, in order
        letter_array = sorted(letter_map.keys(), key=letter_map.get)
        # find our matches
        matches = re.findall(i, self.word, re.IGNORECASE)[0]
        if type(matches) == str:
            matches = [matches]
        # partition if necessary
        mylist = [allPartitions(m, len(letter_array[k])) for k, m in enumerate(matches)]
        partitions = []
        for p in itertools.product(*mylist):
            thesePartitions = dict()
            for j, p1 in enumerate(p):
                for k, char in enumerate(letter_array[j]):
                    thesePartitions[char] = p1[k]
            partitions.append(thesePartitions)
        return partitions
        

# For testing purposes
class MyArgs:
    def __init__(self):
        self.input = 'AB;BC;CD'
        #self.input = '*.AredABCA.*'
        
def solve_equation(_input):
    # Split the input
    inputs = split_input(_input)
    # Get the variables we iterate over, and those we don't
    cover, others = get_set_cover(inputs)
    
    # The number of inputs is the number of patterns in the cover
    num_inputs = len(cover)
    
    # Set up lists of candidate words
    # and our regular expressions
    words = []
    regexes = dict()
    for i in cover:
        words.append([])
        pattern = input_to_regex(i)
        regexes[i] = re.compile(pattern, re.IGNORECASE)
    
    # Go through the word list and get words that match the "cover" pattern(s)
    # we also store all the words for "others" matching
    # TODO: optimize "others" in the same way
    all_words = set()
    with open(WORD_LIST, 'r') as fid:
        for line in fid:
            word, score = line.split(';')
            score = int(score)
            if score < MIN_SCORE:
                continue
            for i, patt in enumerate(cover):
                if regexes[patt].match(word) is not None:
                    w = Word(word, score, patt)
                    words[i].append(w)
            all_words.add(word)
            
    # Now loop through all the necessary lists
    # and see if the "others" match something
    for word_tuple in itertools.product(*words):
        #print(word_tuple)
        partitions = [w.all_partitions() for w in word_tuple]
        #print(partitions)
        for p1 in itertools.product(*partitions):
            print(p1)
            break
        break
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    
    args = MyArgs()
    
    solve_equation(args.input)
    
            
        