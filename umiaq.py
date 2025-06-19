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
import umiaq_split

# Global variables

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

# Regex to match named groups like (?P<name>...)
NAMED_GROUP_PATTERN = r"\(\?P<\w+>.*?\)"

# Greek letters are for extended variables
GREEK_LETTERS = [ '\u03B1', '\u03B2', '\u03B3', '\u03B4', '\u03B5', '\u03B6'
                , '\u03B7', '\u03B8', '\u03B9', '\u03BA', '\u03BB', '\u03BC']

# Make partitions of a string
# We will need to change this when it gets more advanced
def multiSlice(s, cutpoints):
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

# This includes partitions of length 0
def allPartitions(s, num=None):
    n = len(s)
    cuts = list(range(0,n+1))
    if num:
        num_arr = [num-1]
    else:
        num_arr = range(n)
    for k in num_arr:
        for cutpoints in itertools.combinations_with_replacement(cuts,k):
            yield multiSlice(s,cutpoints)

def replace_first_and_others(original, to_replace, first_replacement, other_replacements):
    """
    Replace the first occurrence of `to_replace` in `original` with `first_replacement`
    and all subsequent occurrences with `other_replacements`.

    Parameters:
        original (str): The original string.
        to_replace (str): The substring or character to be replaced.
        first_replacement (str): The replacement for the first occurrence.
        other_replacements (str): The replacement for all subsequent occurrences.

    Returns:
        str: The modified string with replacements applied.
    """
    first_index = original.find(to_replace)

    # If `to_replace` is not found, return the original string as is.
    if first_index == -1:
        return original

    result_chars = []
    found_first = False

    # Iterate through each character in the original string
    for char in original:
        if char == to_replace:
            if not found_first:
                # Replace the first occurrence
                result_chars.append(first_replacement)
                found_first = True
            else:
                # Replace subsequent occurrences
                result_chars.append(other_replacements)
        else:
            # Keep all other characters unchanged
            result_chars.append(char)

    return "".join(result_chars)

# Function to combine dictionaries, with conflict resolution
def combine_dicts(*dlist):
    """
    Combine a tuple of dictionaries into one.
    If there is a conflict in keys, return an empty dictionary

    Returns: dict
    """
    ret = dict()
    for d in dlist:
        for k, v in d.items():
            if ret.get(k, d[k]) != d[k]:
                return dict()
            else:
                ret[k] = d[k]
    return ret

def deduplicate_in_order(arr):
    """Helper function to de-dupe a list"""
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def split_input(i):
    """
    Split an input string along the split character

    Returns: Patterns object
    """
    patt_list = []
    greek_letters = GREEK_LETTERS[::-1]

    inputs = i.split(';')

    # Split out inputs with equals signs and those without
    i1 = [_ for _ in inputs if '=' in _]
    i2 = [_ for _ in inputs if '=' not in _]

    # deal with '=' queries first
    # for now we just accept length queries
    all_lengths = {}
    for x in i1:
        len_match = re.match(r'^\|([A-Z])\|=(\d+)$', x)
        if len_match is not None:
            all_lengths[len_match.groups()[0]] = int(len_match.groups()[1])

    for x in i2:
        values = {}; lengths = {}
        x1 = x
        # handle dots and stars
        for dot_star in re.findall(r'[\.\*]+', x1):
            greek_letter = greek_letters.pop()
            x1 = x1.replace(dot_star, greek_letter)
            min_length = dot_star.count('.')
            max_length = BIG_NUMBER if '*' in dot_star else min_length
            lengths[greek_letter] = [min_length, max_length]
        # handle lowercase letters
        for let in re.findall(r'[a-z]+', x1):
            greek_letter = greek_letters.pop()
            x1 = x1.replace(let, greek_letter)
            lengths[greek_letter] = [len(let), len(let)]
            values[greek_letter] = let

        # handle "global" lengths
        for k, v in all_lengths.items():
            if k in x:
                lengths[k] = [all_lengths[k], all_lengths[k]]

        patt = Pattern(x1, values, lengths)
        patt_list.append(patt)
    return Patterns(patt_list)

def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

class Patterns:
    def __init__(self, pattern_list):
        self.list = pattern_list
        self.ordered_list = self.ordered_partitions()

    def __repr__(self):
        return f"patterns: {self.ordered_list}"

    def __iter__(self):
        # Return an iterator for the list
        return iter(self.ordered_list)

    def all_variables(self):
        # Get all variables in the patterns
        ret = set()
        for p in self.list:
            s = set(_ for _ in p.string if _ in UPPERCASE_LETTERS)
            ret = ret | s
        return ret

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

# store a pattern and all the things that go with it
class Pattern:
    def __init__(self, patt_str, values={}, lengths={}):
        self.string = patt_str
        self.values = values
        self.lengths = lengths
        self.regex = self.to_regex()
        self.lookup_keys = None
        self.var_dict = self.get_var_dict()

    def __repr__(self):
        return f"Pattern(string: {self.string}, values: {self.values}, lengths: {self.lengths})"

    def from_input(self, _input):
        self.string = _input

    def variables(self):
        return set([_ for _ in self.string if _ in UPPERCASE_LETTERS])

    def get_var_dict(self):
        """
        returns a dictionary like {"A": 0, "B": 2}
        where keys are variables names
        and values are indexes to the partition
        """
        lk = dict()
        for ix, r in enumerate(umiaq_split.split_named_regex(self.regex)):
            if re.fullmatch(NAMED_GROUP_PATTERN, r):
                lk[r[4]] = ix

        return lk

    def is_deterministic(self):
        """
        A deterministic pattern is one that is all
        either capital letters or Greek letters with values
        """
        for v in self.variables():
            if re.match(r'[A-Z]', v):
                continue
            if self.values.get(v) is None:
                return False
        return True

    def to_word(self, d):
        """
        Given a dictionary associating variables to strings,
        return the word created.
        Note that the pattern must be a "deterministic" one.

        Returns: str
        """
        assert self.is_deterministic()
        ret = self.string
        # Replace the given values
        for k, v in d.items():
            ret = ret.replace(k, v.lower())
        # Replace any Greek letters with values
        for k, v in self.values.items():
            ret = ret.replace(k, v.lower())
        ret = ret.upper()
        return ret

    def to_regex(self):
        # Return a regex that will match the pattern
        i = self.string
        lengths = self.lengths

        # Allow only certain characters (not for now)
        #if re.match(r'[^A-Za-z\*\.\#\@]', i):
        #    logging.error(f"Input string {i} has bad characters")
        #    return None

        # Replace asterisks with ".*"
        i = i.replace('*', '.*')

        # Capital letter replacement is slightly complicated
        # The first occurrence is replaced with a `(.+)` or a length string
        # subsequent ones have to be replaced with appropriate backrefs
        capital_letters = re.findall(r'[A-Z]', i)

        ctr = 1
        used_letters = dict()
        for c in deduplicate_in_order(capital_letters):
            # get the pattern, which depends on length
            c_patt = f'(?P<{c}>.+)'
            if lengths.get(c):
                periods = '.' * lengths.get(c)[0]
                c_patt = f'(?P<{c}>{periods})'

            # replace using our helper function
            i = replace_first_and_others(i, c, c_patt, f'\\{ctr}')
            used_letters[c] = ctr
            ctr += 1

        # take the rest in groups
        other_letters = re.findall(r'[a-z]+', i)
        for c in other_letters:
            dots = '.' * len(c)
            i = i.replace(c, f'({dots}+)')
            used_letters[c] = ctr
            ctr += 1

        # Replace @ with vowels, # with consonants
        # I'm going to count "y" as both
        i = i.replace('@', '[AEIOUY]')
        i = i.replace('#', '[B-DF-HJ-NP-TV-XZ]')

        #i = '^' + i + '$'

        # Now fix the Greek letters
        ret = i
        for char in [x for x in ret if x in GREEK_LETTERS]:
            if self.values.get(char):
                ret = ret.replace(char, self.values[char].lower())
            elif self.lengths.get(char):
                r = ''
                min_l, max_l = self.lengths.get(char)
                for _ in range(min_l):
                    r += '.'
                if max_l == BIG_NUMBER:
                    r += '.*'
                else:
                    for _ in range(max_l - min_l):
                        r += '.'
                ret = ret.replace(char, r)
            else:
                ret = ret.replace(char, '.+')
        return ret

# store a word and all its partitions
class Word:
    def __init__(self, word, score, pattern):
        self.word = word
        self.pattern = pattern
        self.score = score

    # Prints object information
    def __repr__(self):
        j = {'word': self.word, 'score': self.score, 'pattern': self.pattern.string}
        return f'Word({json.dumps(j)})'


    # Prints readable form
    def __str__(self):
        return self.word

    # Create the dictionary of variables to matches
    def get_match_dict(self):
        # If no lettered variables, return empty dict
        if re.search('[A-Z]', self.pattern.string) is None:
            return {}
        r = self.pattern.to_regex()
        m = re.match(r, self.word)
        d = m.groupdict()
        d['word'] = self.word
        return d

    def matches_pattern(self):
        # check that the word matches the pattern
        r = self.pattern.to_regex()
        return re.fullmatch(r, self.word) is not None

# For testing purposes
class MyArgs:
    def __repr__(self):
        return self.input

    def __init__(self, _input):
        self.input = _input
        self.debug = False
        self.num_results = NUM_RESULTS

def solve_equation(_input, num_results=NUM_RESULTS, max_word_length=MAX_WORD_LENGTH, return_json=False):
    # Split the input into some patterns
    patterns = split_input(_input)

    # Set up lists of candidate words
    # and our regular expressions
    words = []
    regexes = dict()
    for patt in patterns:
        words.append(defaultdict(list))
        reg = patt.to_regex()
        regexes[patt] = re.compile(reg, re.IGNORECASE)

    # Go through the word list and get words that match the pattern(s)
    # we also store all the words for "others" matching
    t1 = time.time()
    # We also maintain a dictionary of "entry" to "score"
    entry_to_score = dict()

    # Keep track of words; don't add too many
    word_counts = [0] * len(words)

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
            for i, patt in enumerate(patterns):
                if regexes[patt].fullmatch(word) is not None:
                    #w = Word(word, score, patt)
                    # I don't want to just add the "word"
                    # I want to add all of its partitions
                    for part in umiaq_split.split_word_on_pattern(word, patt):
                        # get the key where we want to insert this
                        if not patt.lookup_keys:
                            words[i][None].append(part)
                        else:
                            _key = frozenset(dict((let, part[let]) for let in patt.lookup_keys).items())
                            words[i][_key].append(part)

                        word_counts[i] += 1
                if word_counts[i] >= MAX_WORD_COUNT:
                    break

                #END if regexes
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
                lookup_keys = patterns.ordered_list[current_index + 1].lookup_keys
                
                d = dict((let, w[let]) for let in patterns.ordered_list[current_index].variables())
               
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
