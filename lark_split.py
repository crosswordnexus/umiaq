from lark import Lark, Transformer, v_args

# --- Grammar Definition ---
# This grammar defines a pattern language where:
# - Uppercase letters (A-Z) are variables, optionally with length constraints like A{3}
# - Lowercase letters (a-z) are literals
# - "." is a wildcard for any single character
# - ~A means "reverse of the string bound to A"
# - ~A{3} means reverse a substring of length 3 and bind it to A
# - [abc] matches one of the listed characters
# - "*" matches any sequence of characters (non-empty)
# - "@" matches any vowel (including Y)
# - "#" matches any consonant (excluding Y)
grammar = r"""
    start: expr+ -> start

    ?expr: varref
         | revref
         | charset
         | star
         | vowel
         | consonant
         | literal
         | dot

    varref: VAR length?          -> var
    revref: "~" VAR length?       -> reverse
    length: "{" NUMBER "}"       -> length
    charset: "[" /[a-z]+/ "]"    -> charset
    star: "*"                     -> star
    vowel: "@"                   -> vowel
    consonant: "#"               -> consonant
    literal: LITERAL             -> literal
    dot: "."                     -> dot

    VAR: /[A-Z]/
    LITERAL: /[a-z]/
    NUMBER: /\d+/

    %import common.WS_INLINE
    %ignore WS_INLINE
"""

# --- Transformer ---
# Converts the parse tree into a list of tagged tuples representing pattern components
@v_args(inline=True)
class PatternTransformer(Transformer):
    def start(self, *parts):
        return list(parts)

    def var(self, name, length=None):
        return ('var', str(name), int(length) if length else None)

    def reverse(self, name, length=None):
        return ('rev', str(name), int(length) if length else None)

    def literal(self, char):
        return ('lit', str(char))

    def dot(self):
        return ('dot',)

    def charset(self, chars):
        return ('set', set(str(chars)))

    def star(self):
        return ('star',)

    def vowel(self):
        return ('vowel',)

    def consonant(self):
        return ('cons',)

    def length(self, n):
        return n

# --- Pattern Container ---
# Wraps the list of parsed pattern parts
class Pattern:
    def __init__(self, parts):
        self.parts = parts

# --- Pattern Parser ---
# Parses a string pattern into a Pattern object
from functools import lru_cache

@lru_cache(maxsize=256)
def parse_pattern(text):
    parser = Lark(grammar, parser="lalr", transformer=PatternTransformer())
    parts = parser.parse(text)
    return Pattern(parts)

# --- Unified Pattern Matcher ---
# Matches a word against a Pattern object
# - If all_matches=False (default): return the first successful match (dict of variable bindings)
# - If all_matches=True: return a list of all valid matches (each a dict of variable bindings)
def match_pattern(word, pattern, all_matches=False):
    results = []      # store successful matches
    memo = set()      # memoization set to avoid redundant states

    VOWELS = set("aeiouy")
    CONSONANTS = set("bcdfghjklmnpqrstvwxz")

    def helper(i, pi, bindings):
        key = (i, pi, tuple(sorted(bindings.items())))
        if key in memo:
            return

        # Base case: pattern is fully consumed
        if pi == len(pattern.parts):
            if i == len(word):
                result = dict(bindings)
                result["word"] = word
                if all_matches:
                    results.append(result)
                else:
                    results.append(result)
                    raise StopIteration  # short-circuit if only first match is needed
            return

        part = pattern.parts[pi]

        if i > len(word):
            memo.add(key)
            return

        if part[0] == 'dot':
            if i < len(word):
                helper(i+1, pi+1, bindings)

        elif part[0] == 'lit':
            if i < len(word) and word[i] == part[1]:
                helper(i+1, pi+1, bindings)

        elif part[0] == 'set':
            if i < len(word) and word[i] in part[1]:
                helper(i+1, pi+1, bindings)

        elif part[0] == 'vowel':
            if i < len(word) and word[i] in VOWELS:
                helper(i+1, pi+1, bindings)

        elif part[0] == 'cons':
            if i < len(word) and word[i] in CONSONANTS:
                helper(i+1, pi+1, bindings)

        elif part[0] == 'star':
            for j in range(i+1, len(word)+1):
                helper(j, pi+1, bindings)
                if not all_matches and results:
                    return

        elif part[0] == 'var':
            name = part[1]
            fixed_len = part[2]
            if name in bindings:
                val = bindings[name]
                if word.startswith(val, i):
                    helper(i+len(val), pi+1, bindings)
            else:
                maxlen = len(word) - i
                lengths = [fixed_len] if fixed_len else range(1, maxlen+1)
                for L in lengths:
                    val = word[i:i+L]
                    new_bindings = dict(bindings)
                    new_bindings[name] = val
                    helper(i+L, pi+1, new_bindings)
                    if not all_matches and results:
                        return

        elif part[0] == 'rev':
            name = part[1]
            fixed_len = part[2]
            if name in bindings:
                revval = bindings[name][::-1]
                if word.startswith(revval, i):
                    helper(i+len(revval), pi+1, bindings)
            else:
                maxlen = len(word) - i
                lengths = [fixed_len] if fixed_len else range(1, maxlen+1)
                for L in lengths:
                    revval = word[i:i+L]
                    val = revval[::-1]
                    new_bindings = dict(bindings)
                    new_bindings[name] = val
                    helper(i+L, pi+1, new_bindings)
                    if not all_matches and results:
                        return

        memo.add(key)

    try:
        helper(0, 0, {})
    except StopIteration:
        pass

    if all_matches:
        return results
    return results[0] if results else None


#%%
if __name__ == '__main__':
    print(match_pattern("schwa", parse_pattern("A###B"), all_matches=True))
