from lark import Lark, Transformer
from functools import lru_cache
from collections import Counter
import re

# --- Grammar Definition ---
# This grammar defines a pattern language where:
# - Uppercase letters (A-Z) are variables
# - Lowercase letters (a-z) are literals
# - "." is a wildcard for any single character
# - ~A means "reverse of the string bound to A"
# - [abc] matches one of the listed characters
# - "*" matches any sequence of characters (non-empty)
# - "@" matches any vowel (including Y)
# - "#" matches any consonant (excluding Y)
grammar = r"""
    start: expr+ -> start

    ?expr: anagram
         | varref
         | revref
         | charset
         | star
         | vowel
         | consonant
         | literal
         | dot

    varref: VAR               -> var
    revref: "~" VAR           -> reverse
    charset: "[" /[a-z]+/ "]" -> charset
    star: "*"                 -> star
    vowel: "@"               -> vowel
    consonant: "#"           -> consonant
    literal: LITERAL+        -> literal
    anagram: "/" LITERAL+ -> anagram
    dot: "." -> dot

    VAR: /[A-Z]/
    LITERAL: /[a-z]/

    %import common.WS_INLINE
    %ignore WS_INLINE
"""

# --- Transformer ---
class PatternTransformer(Transformer):
    def anagram(self, chars):
        return ('anagram', ''.join(str(c) for c in chars))
    def start(self, parts):
        return parts

    def var(self, name):
        return ('var', name[0].value)

    def reverse(self, name):
        return ('rev', name[0].value)

    def literal(self, chars):
        return ('lit', ''.join(str(c) for c in chars))

    def dot(self, _=None):
        return ('dot',)

    def charset(self, token):  # pull from list
        return ('set', ''.join(set(token)).upper())

    def star(self, _=None):
        return ('star',)

    def vowel(self, _=None):
        return ('vowel',)

    def consonant(self, _=None):
        return ('cons',)

class Pattern:
    def __init__(self, parts):
        self.parts = tuple(parts)

# Set up the parser
PARSER = Lark(grammar, parser="lalr", transformer=PatternTransformer())

# Freeze length contraints
def freeze_lengths(d):
    if d is None:
        return None
    arr = []
    for k in sorted(d.keys()):
        v = d[k]
        _tup = (k, str(v.get('min_length', 1)), str(v.get('max_length', 100)))
        arr.append(_tup)
    return tuple(arr)

@lru_cache(maxsize=256)
def pattern_parts_to_regex(pattern_parts, frozenconstraints=None):
    """
    Convert a pattern to a regex for quick matching
    """
    
    # Convert frozenconstraints to a dict
    if frozenconstraints:
        var_constraints = dict((k[0], k[1:]) for k in frozenconstraints)
    else:
        var_constraints = {}
    
    def _convert(node):
        if isinstance(node, str):
            return re.escape(node)  # fallback: treat as literal
        tag = node[0]

        if tag == 'anagram':
            subpatterns = node[1]
            return '.' * len(subpatterns)

        elif tag in ('var', 'rev'):
            this_var = node[1]
            if var_constraints.get(this_var):
                return ".{" \
                    + var_constraints.get(this_var, ["1", "100"])[0] \
                    + ',' \
                    + var_constraints.get(this_var, ["1", "100"])[1] \
                    + '}'
            else:
                return ".+"

        elif tag == 'set':
            charset = re.escape(node[1])
            return f'[{charset}]'

        elif tag == 'star':
            return '.*'

        elif tag == 'vowel':
            return '[AEIOUY]'

        elif tag == 'cons':
            return '[B-DF-HJ-NP-TV-XZ]'

        elif tag == 'lit':
            return re.escape(node[1].upper())

        elif tag == 'dot':
            return '.'

        else:
            raise ValueError(f"Unknown pattern type: {tag}")

    ret = ''
    for node in pattern_parts:
        ret += _convert(node)
    return re.compile(ret)

@lru_cache(maxsize=256)
def parse_pattern(text):
    parts = PARSER.parse(text)
    return Pattern(parts)

# --- Constraint Validator ---
@lru_cache(maxsize=1024)
def is_valid_binding(val, frozen_constraints, frozen_bindings, name):
    constraints = dict(frozen_constraints)
    bindings = dict(frozen_bindings)

    pattern = constraints.get('pattern')
    if pattern:
        from_pattern = parse_pattern(pattern)
        if not match_pattern(val, from_pattern):
            return False

    not_equal = constraints.get('not_equal', [])
    for other in not_equal:
        if other in bindings and bindings[other] == val:
            return False

    return True

# Helper to freeze dictionaries for caching
def freeze_dict(d):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items())) if d else ()

# --- Pattern Matcher ---
def match_pattern(word, pattern, all_matches=False, var_constraints=None):
    """
    Check if a word matches a pattern
    Return one match or all matches, as appropriate
    """
    
    word = word.upper()  # normalize word to uppercase
    
    # Do the regex check first
    vc_freeze = freeze_lengths(var_constraints)
    regex = pattern_parts_to_regex(pattern.parts, vc_freeze)
    
    if not regex.fullmatch(word):
        return [] if all_matches else None
    
    results = []  # stores all valid bindings
    memo = set()  # stores visited states to prevent reprocessing

    VOWELS = set("AEIOUY")
    CONSONANTS = set("BCDFGHJKLMNPQRSTVWXZ")

    def helper(i, pi, bindings):
        # i = current index in word
        # pi = current index in pattern.parts
        # bindings = current variable bindings

        # use a hashable key to memoize search
        key = (i, pi, tuple(sorted(bindings.items())))
        if key in memo:
            return

        if pi == len(pattern.parts):
            if i == len(word):
                result = dict(bindings)
                result["word"] = word
                results.append(result)
                if not all_matches:
                    raise StopIteration # stop early if only first match desired
            return

        part = pattern.parts[pi] # get current pattern part

        # fail early if we've run past the end of the word
        if i > len(word):
            memo.add(key)
            return

        kind = part[0]

        # Match a single character wildcard
        if kind == 'dot':
            if i < len(word):
                helper(i+1, pi+1, bindings)

        # Match an exact literal character (or characters)
        elif kind == 'lit':
            literal = part[1].upper()
            if word[i:i+len(literal)] == literal:
                helper(i + len(literal), pi + 1, bindings)

        # Match one of a specified set of characters
        elif kind == 'set':
            if i < len(word) and word[i] in set(c.upper() for c in part[1]):
                helper(i+1, pi+1, bindings)

        # Match any vowel (including Y)
        elif kind == 'vowel':
            if i < len(word) and word[i] in VOWELS:
                helper(i+1, pi+1, bindings)

        # Match any consonant (not Y)
        elif kind == 'cons':
            if i < len(word) and word[i] in CONSONANTS:
                helper(i+1, pi+1, bindings)

        # Match zero or more characters (greedy)
        elif kind == 'star':
            for j in range(i, len(word)+1):
                helper(j, pi+1, bindings)
                if not all_matches and results:
                    return

        # Match an anagram
        elif kind == 'anagram':
            target = Counter(part[1].upper())
            if i + len(target) <= len(word):
                actual = Counter(word[i:i + len(target)])
                if target == actual:
                    helper(i + len(target), pi + 1, bindings)

        # Match a variable or reversed variable
        elif kind in ('var', 'rev'):
            name = part[1]
            if name in bindings:
                val = bindings[name]
                if kind == 'rev':
                    val = val[::-1]
                if word.startswith(val, i):
                    helper(i+len(val), pi+1, bindings)
            else:
                # Determine constraints
                minlen = 1
                maxlen = len(word) - i
                c = var_constraints.get(name) if var_constraints else None
                if c:
                    if 'min_length' in c:
                        minlen = max(minlen, c['min_length'])
                    if 'max_length' in c:
                        maxlen = min(maxlen, c['max_length'])

                frozen_constraints = freeze_dict(c) if c else ()
                frozen_bindings = freeze_dict(bindings)

                # Try all substrings from minlen to maxlen
                for L in range(minlen, maxlen+1):
                    val = word[i:i+L]
                    bound_val = val[::-1] if kind == 'rev' else val
                    if is_valid_binding(bound_val, frozen_constraints, frozen_bindings, name):
                        new_bindings = dict(bindings)
                        new_bindings[name] = bound_val
                        helper(i+L, pi+1, new_bindings)
                        if not all_matches and results:
                            return

        memo.add(key) # mark this state as visited

    try:
        helper(0, 0, {}) # start recursive search
    except StopIteration:
        pass # stop early if not collecting all matches

    return results if all_matches else (results[0] if results else None)
