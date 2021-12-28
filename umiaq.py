#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:17:46 2021

@author: Alex Boisvert
"""

import re
import itertools
import argparse

# Global variables

# The number of results to report
NUM_RESULTS = 1000
# The minimum score in the word list
MIN_SCORE = 50

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
    def __init__(self, word, pattern):
        self.word = word
        self.pattern = pattern
        self.partitions = self.all_partitions()
        
    def matches_pattern(self):
        # check that the word matches the pattern
        return True
    
    def all_partitions(self):
        # return all the partitions of the word that match the pattern
        return ()
        

# For testing purposes
class MyArgs:
    input = 'AB;BC;CD'
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    
    args = MyArgs()
    
    # Split the input
    inputs = split_input(args.input)
    # Get the variables we iterate over, and those we don't
    cover, others = get_set_cover(inputs)
    
    # 