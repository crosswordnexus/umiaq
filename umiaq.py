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
import time
import sys
from collections import defaultdict
import lark_split

from functools import lru_cache

## Global variables ##
# The number of results to report
NUM_RESULTS = 100
# The minimum score in the word list
MIN_SCORE = 50
# The maximum word length we are interested in
MAX_WORD_LENGTH = 21
# The word list itself
WORD_LIST = 'xwordlist_sorted_trimmed.txt'

UPPERCASE_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# The maximum number of "words" we search through
MAX_WORD_COUNT = 50_000

# A standard big number
BIG_NUMBER = 1e6

# Default length endpoints
DEFAULT_LENGTHS = [1, BIG_NUMBER]

def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

@lru_cache(maxsize=None)
def get_variables(_input):
    """Get the "variables" (uppercase letters) in a string"""
    return set(_ for _ in _input if _ in UPPERCASE_LETTERS)

class Pattern:
    """Details about a particular pattern"""
    def __init__(self, patt_str):
        self.set_string(patt_str)
        self.lookup_keys = None

    def __repr__(self):
        return f"Pattern({self.string})"

    def from_input(self, _input):
        self.string = _input
        
    def set_string(self, patt_str):
        """Create the string for lark"""
        self.string = patt_str
        
    def variables(self):
        return get_variables(self.string)

class Patterns:
    # takes the _input (with semi-colons) as input
    def __init__(self, _input):
        self._input = _input
        self.list = []
        self.var_constraints = {}
        # this next bit makes the list and constraints
        self.make_list(_input)
        self.ordered_list = self.ordered_partitions()

    def __repr__(self):
        return f"patterns: {self.ordered_list}, constraints: {self.var_constraints}"

    def __iter__(self):
        # Return an iterator for the list
        return iter(self.ordered_list)
    
    def make_list(self, _input):
        _list = _input.split(';')
        
        # deal with '=' queries first
        var_constraints = defaultdict(dict)
        for x in _list:
            if '=' not in x:
                continue
            # Look for just length queries
            len_match = re.match(r'^\|([A-Z])\|=(\d+)$', x)
            if len_match is not None:
                this_var, _len = len_match.groups()
                var_constraints[this_var]['min_length'] = int(_len)
                var_constraints[this_var]['max_length'] = int(_len)
            # Look for more complex queries
            q_match = re.match(r'^([A-Z])=\(?([\d\-]*)\:?([^\(\)]*)\)?$', x)
            if q_match is not None:
                this_var, _len, _pattern = q_match.groups()
                min_length = int(_len.split('-')[0]) if _len.split('-')[0] else None
                max_length = int(_len.split('-')[-1]) if _len.split('-')[-1] else None
                if min_length:
                    var_constraints[this_var]['min_length'] = min_length
                if max_length:
                    var_constraints[this_var]['max_length'] = max_length
                # Don't bother with trivial patterns
                if _pattern and _pattern != '*':
                    var_constraints[this_var]['pattern'] = _pattern
                    
        self.var_constraints = dict(var_constraints)
                
        # Now make the list of "pattern"s
        this_list = []
        for x in _list:
            if '=' not in x:
                this_list.append(Pattern(x))
        self.list = this_list

    def all_variables(self):
        # Get all variables in the patterns
        return self.variables

    def ordered_partitions(self):
        """
        order partitions the way we want them
        the first one has the most variables
        after that we take the one with the largest overlap with what we've got
        """
        patt_list = self.list.copy()

        # set up the return object
        op = []
        # Find the index of the largest set
        ix = max(range(len(patt_list)), key=lambda i: len(patt_list[i].variables()))

        # Pop the largest set
        patt = patt_list.pop(ix)
        op.append(patt)

        # now loop through the others
        # we also keep track of the "lookup list"
        # i.e. the variables we want to index on
        
        while patt_list:
            # take the set of all variables found so far
            found_vars = set().union(*[_.variables() for _ in op])
            # find the index of largest overlap
            ix2 = max(range(len(patt_list)), key=lambda i: len(patt_list[i].variables() & found_vars))
            # pop it
            lookup_keys = frozenset(patt_list[ix2].variables() & found_vars)
            patt2 = patt_list.pop(ix2)
            patt2.lookup_keys = lookup_keys
            op.append(patt2)

        return op

    def set_cover(self):
        # Find a covering set for our variables
        # TODO: this could be optimized
        av = self.all_variables()

        # case where there are no variables
        if not av:
            return set(self.list), set()
        for myset in powerset(self.list):
            patterns_tmp = Patterns(list(myset))
            myvars = patterns_tmp.all_variables()
            if myvars == av:
                return set(myset), set(self.list).difference(myset)


# For testing purposes
class MyArgs:
    def __repr__(self):
        return self.input

    def __init__(self, _input):
        self.input = _input
        self.debug = False
        self.num_results = NUM_RESULTS
        self.minscore = 0

def solve_equation(_input, num_results=NUM_RESULTS, max_word_length=MAX_WORD_LENGTH, return_json=False):

    # Set up our patterns object
    pattern_obj = Patterns(_input)

    # Set up lists of candidate words
    # and our regular expressions
    words = []
    for patt in pattern_obj:
        words.append(defaultdict(list))

    # Go through the word list and get words that match the pattern(s)
    # we also store all the words for "others" matching
    t1 = time.time()
    # We also maintain a dictionary of "entry" to "score"
    entry_to_score = dict()

    # Keep track of words; don't add too many
    word_counts = [0] * len(words)
    
    # Parse our patterns
    parsed_patterns = dict((patt, lark_split.parse_pattern(patt.string)) for patt in pattern_obj)

    with open(WORD_LIST, 'r') as fid:
        for line in fid:
            word, score = line.split(';')
            # Normalize to all uppercase words
            word = word.upper()
            score = int(score)
            if score < MIN_SCORE or len(word) > max_word_length:
                continue
            entry_to_score[word] = score
            # do the cover words
            for i, patt in enumerate(pattern_obj):
                pattern_parsed = parsed_patterns[patt]
                for part in lark_split.match_pattern(
                                        word, 
                                        pattern_parsed, 
                                        all_matches=True, 
                                        var_constraints=pattern_obj.var_constraints):
                    # get the key where we want to insert this
                    if not patt.lookup_keys:
                        words[i][None].append(part)
                    else:
                        _key = frozenset(dict((let, part[let]) for let in patt.lookup_keys).items())
                        words[i][_key].append(part)

                    word_counts[i] += 1
            if word_counts[i] >= MAX_WORD_COUNT:
                break
            #END for i in others
        #END for line in fid
    #END with open

    t2 = time.time()
    logging.debug(f'Initial pass through word list: {(t2-t1):.3f} seconds')

    # If there's only one input, there's no need to loop through everything again
    # if len(patterns.list) == 1:
    #     s = set()
    #     for w in words[0][None][:num_results]:
    #         s.add((w,))
    #     ret = list(s)
    #     return ret

    # Recursively search our words for matches
    t3 = time.time()

    # helper function for the recursive search
    def recursive_filter(current_list, current_index=0, selected=None, current_dict=None, results=None):
        """
        Recursively loops through lists in `words`, filtering the next list based on current selections.

        :param words: A list of dictionaries. Each contains values to loop through.
        :param current_list: Index of the current list being processed.
        :param selected: List of selected values from previous lists.
        :param current_dict: a dictionary for filtering
        :param results: a list to store results
        """

        if selected is None:
            selected = []

        if current_dict is None:
            current_dict = {}

        if results is None:
            results = []

        if len(results) >= num_results:
            return results

        if current_index == len(words):  # Base case: All lists processed
            results.append(selected)
            return results  # Return the current selection as a valid result

        # Loop through the current list
        for w in current_list:
            # Filter the next list based on the current value
            if current_index + 1 >= len(words):
                next_list = current_list
                d = current_dict
            else:
                # keys in the upcoming index
                lookup_keys = pattern_obj.ordered_list[current_index + 1].lookup_keys
                
                d = dict((let, w[let]) for let in pattern_obj.ordered_list[current_index].variables())
               
                d.update(current_dict)
                
                # Restrict d to just keys in the upcoming index
                d1 = dict((k, d[k]) for k in lookup_keys)
                
                _key = frozenset(d1.items())
                next_list = words[current_index + 1][_key]

            # Recurse and accumulate results
            recursive_filter(next_list, current_index + 1, selected + [w], d, results)

            # Stop recursion if the desired number of results is reached
            if len(results) >= num_results:
                break

        return results

    ret = recursive_filter(words[0][None])

    t4 = time.time()
    logging.debug(f'Final pass: {(t4-t3):.3f} seconds')
    ret = ret[:num_results]
    ret = strip_defaultdict(ret)
    if return_json:
        return json.dumps(ret)
    else:
        return ret

def strip_defaultdict(obj):
    """Recursively convert defaultdicts to dicts"""
    if isinstance(obj, list):
        return [strip_defaultdict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: strip_defaultdict(v) for k, v in obj.items()}
    elif isinstance(obj, defaultdict):
        return dict(obj)
    else:
        return obj

def score_tuple(word_tuple):
    """
    Score a tuple of words.
    For now this is just the sum of the individual scores
    """
    return sum(w.score for w in word_tuple)

def test_cases():
    """
    Run some basic tests to confirm results
    """
    arr = ['l.....x', '..i[sz]e', '#@#@#@#@#@#@#@', '*xj*', 'AA', 'A~A',
           'AB;BA;|A|=2;B=(3-:*)', 'AkB;AlB', 'A###B;A@@@B;A=(h*)']
    
    for _input in arr:
        print(f"-- {_input} --")
        t1 = time.time()
        res = solve_equation(_input, num_results = 5)
        for word_tuple in res:
            print(" • ".join([w['word'] for w in word_tuple]))
        t2 = time.time()
        print(f"Total time: {t2-t1:.2f} seconds")
        print()

def main():
    # Parse the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument("-d", "--debug"
                        , help="Turn on debugging"
                        , action="store_true")
    parser.add_argument("-n", "--num_results"
                        , type=int
                        , help="The maximum number of results to output"
                        , default=NUM_RESULTS)

    # If we can't parse the inputs, assume we're testing
    try:
        args = parser.parse_args()
        args.input
    except:
        args = MyArgs('AB;BA')

    # Set up logging
    loglevel = 'INFO'
    if args.debug:
        loglevel = 'DEBUG'
    logging.basicConfig(format='%(levelname)s [%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)


    # Set up a timer
    t1 = time.time()
    # Solve the inputs
    ret = solve_equation(args.input, args.num_results)
    # Sort on score
    #ret_list = sorted(ret, key=score_tuple, reverse=True)
    ret_list = ret
    # Print the output
    for word_tuple in ret_list:
        print(" • ".join([w['word'] for w in word_tuple]))
    if len(ret) >= NUM_RESULTS:
        print("Maximum number of outputs reached")
    t2 = time.time()
    print(f"Total time: {t2-t1:.3f} seconds")
    return 0


#%%
if __name__ ==  '__main__':
    sys.exit(main())
