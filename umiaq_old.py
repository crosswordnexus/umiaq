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
import time
import sys

# Global variables

# The number of results to report
NUM_RESULTS = 100
# The minimum score in the word list
MIN_SCORE = 80
# The maximum word length we are interested in
MAX_WORD_LENGTH = 21
# The word list itself
WORD_LIST = 'xwordlist_sorted_trimmed.txt'

# A standard big number
BIG_NUMBER = 1e6

# Default length endpoints
DEFAULT_LENGTHS = [1, BIG_NUMBER]

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

    def __repr__(self):
        return f"patterns: {self.list}"

    def all_variables(self):
        # Get all variables in the patterns
        ret = set()
        for p in self.list:
            s = set(_ for _ in p.string if _ not in GREEK_LETTERS)
            ret = ret | s
        return ret

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

    def __repr__(self):
        return f"Pattern(string: {self.string}, values: {self.values}, lengths: {self.lengths})"

    def from_input(self, _input):
        self.string = _input

    def variables(self):
        return set(self.string)

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
            c_patt = '(.+)'
            if lengths.get(c):
                periods = '.' * lengths.get(c)[0]
                c_patt = f'({periods})'
            # first replace one
            i = i.replace(c, c_patt, 1)
            # then replace the rest
            i = i.replace(c, f'\\{ctr}')
            used_letters[c] = ctr
            ctr += 1
            
        # take the rest in groups
        other_letters = re.findall(r'[A-Z]+', i)
        for c in other_letters:
            dots = '.' * len(c)
            i = i.replace(c, f'({dots}+)')
            used_letters[c] = ctr
            ctr += 1

        # Replace @ with vowels, # with consonants
        # I'm going to count "y" as both
        i = i.replace('@', '[AEIOUY]')
        i = i.replace('#', '[B-DF-HJ-NP-TV-Z]')

        i = '^' + i + '$'
        
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
        #self.partitions = self.all_partitions()

    # Prints object information
    def __repr__(self):
        j = {'word': self.word, 'score': self.score, 'pattern': self.pattern.string}
        return f'Word({json.dumps(j)})'

    # Prints readable form
    def __str__(self):
        return self.word

    def matches_pattern(self):
        # check that the word matches the pattern
        r = self.pattern.to_regex()
        return re.match(r, self.word) is not None

    def all_partitions(self):
        # return all the partitions of the word that match the pattern
        p = self.pattern
        mylen = len(p.string)
        partitions = allPartitions(self.word.lower(), mylen)
        # only keep the partition if it matches the pattern
        # TODO: this needs to be heavily optimized
        good_partitions = []
        for partition in partitions:
            is_good_partition = True
            thisPartDict = {}
            for i, char in enumerate(p.string):
                this_part = partition[i]
                if re.match(r'[A-Z]', char):
                    if this_part != thisPartDict.get(char, this_part):
                        is_good_partition = False
                        break
                    thisPartDict[char] = this_part
                if this_part != (p.values.get(char) or this_part):
                    is_good_partition = False
                    break
                char_len = p.lengths.get(char, DEFAULT_LENGTHS)
                if len(this_part) < char_len[0] or len(this_part) > char_len[1]:
                    is_good_partition = False
                    break
            if is_good_partition:
                good_partitions.append(thisPartDict)
        return good_partitions

# For testing purposes
class MyArgs:
    def __repr__(self):
        return self.input

    def __init__(self, _input):
        self.input = _input
        self.debug = False
        self.num_results = NUM_RESULTS

def solve_equation(_input, num_results=NUM_RESULTS, max_word_length=MAX_WORD_LENGTH):
    # Split the input into some patterns
    patterns = split_input(_input)
    # Get the variables we iterate over, and those we don't
    cover, others = patterns.set_cover()

    # Set up lists of candidate words
    # and our regular expressions
    words = []
    regexes = dict()
    for patt in patterns.list:
        if patt in cover:
            words.append([])
        reg = patt.to_regex()
        regexes[patt] = re.compile(reg, re.IGNORECASE)

    # Go through the word list and get words that match the "cover" pattern(s)
    # we also store all the words for "others" matching
    t1 = time.time()
    other_dict = dict()
    for patt in others:
        other_dict[patt] = dict()
    # We also maintain a dictionary of "entry" to "score"
    entry_to_score = dict()
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
            for i, patt in enumerate(cover):
                if regexes[patt].match(word) is not None:
                    w = Word(word, score, patt)
                    words[i].append(w)
            # do the "other" words
            for i, patt in enumerate(others):
                # We keep a dictionary for easy lookup
                if regexes[patt].match(word) is not None:
                    w = Word(word, score, patt)
                    # determine if this is a "deterministic" pattern
                    is_det = patt.is_deterministic()
                    if not is_det:
                        all_partitions = w.all_partitions()
                        for p in all_partitions:
                            this_key = []
                            for char in re.findall(r'[A-Z]', patt):
                                this_key.append(p[char])
                            this_key = tuple(this_key)
                            # add to the appropriate dictionary
                            try:
                                other_dict[patt][this_key].add(w)
                            except:
                                other_dict[patt][this_key] = set([w])
                    else:
                        # for deterministic patterns we just store the word itself
                        try:
                            other_dict[patt].add(word)
                        except:
                            other_dict[patt] = set([word])
                    #END if/else is_deterministic
                #END if regex match
            #END for i in others
        #END for line in fid
    #END with open

    t2 = time.time()
    logging.debug(f'Initial pass through word list: {(t2-t1):.3f} seconds')

    # If there's only one input, there's no need to loop through everything again
    if len(patterns.list) == 1:
        s = set()
        for w in words[0]:
            s.add((w,))
        ret = list(s)[:num_results]
        return ret

    # Now loop through all the necessary lists
    # and see if the "others" match something
    t3 = time.time()
    ret = set()
    for word_tuple in itertools.product(*words):
        this_dict_orig = dict([(w.pattern, set([w])) for w in word_tuple])
        partitions = [w.all_partitions() for w in word_tuple]
        for p1 in itertools.product(*partitions):
            this_dict = this_dict_orig.copy()
            #print(p1)
            # Combine these dictionaries, ensuring there are no conflicts
            combined_dict = combine_dicts(*p1)
            # If there's a conflict we move on to the next
            if not combined_dict:
                this_dict = dict()
            for other in others:
                is_det = other.is_deterministic()
                if is_det:
                    # If deterministic, create the word and see if it's there
                    this_word = other.to_word(combined_dict)
                    if this_word in other_dict[other]:
                        w = Word(this_word, entry_to_score[this_word], other)
                        this_dict[other] = set([w])
                    else:
                        this_dict = dict()
                        break
                else:
                    # We need to create the key for lookup
                    other_vars = re.findall(r'[A-Z]', other)
                    this_tuple = tuple([combined_dict[_] for _ in other_vars])
                    try:
                        this_dict[other] = other_dict[other][this_tuple]
                    except:
                        this_dict = dict()
                        break
            #END for other
            # If we've got a match, add it to the return set
            if this_dict:
                entries = [this_dict[_] for _ in patterns.list]
                for entries1 in itertools.product(*entries):
                    ret.add(tuple(entries1))
                    if len(ret) >= num_results:
                        t4 = time.time()
                        logging.debug(f'Final pass: {(t4-t3):.3f} seconds')
                        return ret
        #END for p1
    #END for word_tuple
    t4 = time.time()
    logging.debug(f'Final pass: {(t4-t3):.3f} seconds')
    return ret

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
    ret_list = sorted(ret, key=score_tuple, reverse=True)
    # Print the output
    for word_tuple in ret_list:
        print(" • ".join([w.word for w in word_tuple]))
    if len(ret) >= NUM_RESULTS:
        print("Maximum number of outputs reached")
    t2 = time.time()
    print(f"Total time: {t2-t1:.3f} seconds")
    return 0


#%%
if __name__ ==  '__main__':
    sys.exit(main())
