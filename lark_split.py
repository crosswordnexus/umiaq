from lark import Lark, Transformer, v_args
from functools import lru_cache

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

    ?expr: varref
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
    literal: LITERAL         -> literal
    dot: "."                 -> dot

    VAR: /[A-Z]/
    LITERAL: /[a-z]/

    %import common.WS_INLINE
    %ignore WS_INLINE
"""

# --- Transformer ---
@v_args(inline=True)
class PatternTransformer(Transformer):
    def start(self, *parts):
        return list(parts)

    def var(self, name):
        return ('var', str(name), None)

    def reverse(self, name):
        return ('rev', str(name), None)

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

class Pattern:
    def __init__(self, parts):
        self.parts = parts

@lru_cache(maxsize=256)
def parse_pattern(text):
    parser = Lark(grammar, parser="lalr", transformer=PatternTransformer())
    parts = parser.parse(text)
    return Pattern(parts)

# --- Constraint Validator ---
def is_valid_binding(name, val, constraints):
    c = constraints.get(name) if constraints else None
    if not c:
        return True
    if 'min_length' in c and len(val) < c['min_length']:
        return False
    if 'max_length' in c and len(val) > c['max_length']:
        return False
    if 'pattern' in c:
        from_pattern = parse_pattern(c['pattern'])
        if not match_pattern(val.lower(), from_pattern):
            return False
    return True

# --- Pattern Matcher ---
def match_pattern(word, pattern, all_matches=False, var_constraints=None):
    results = []
    memo = set()
    
    word = word.upper()

    VOWELS = set("AEIOUY")
    CONSONANTS = set("BCDFGHJKLMNPQRSTVWXZ")

    def helper(i, pi, bindings):
        key = (i, pi, tuple(sorted(bindings.items())))
        if key in memo:
            return

        if pi == len(pattern.parts):
            if i == len(word):
                result = dict(bindings)
                result["word"] = word
                if all_matches:
                    results.append(result)
                else:
                    results.append(result)
                    raise StopIteration
            return

        part = pattern.parts[pi]

        if i > len(word):
            memo.add(key)
            return

        if part[0] == 'dot':
            if i < len(word):
                helper(i+1, pi+1, bindings)

        elif part[0] == 'lit':
            if i < len(word) and word[i] == part[1].upper():
                helper(i+1, pi+1, bindings)

        elif part[0] == 'set':
            if i < len(word) and word[i] in set(c.upper() for c in part[1]):
                helper(i+1, pi+1, bindings)

        elif part[0] == 'vowel':
            if i < len(word) and word[i] in VOWELS:
                helper(i+1, pi+1, bindings)

        elif part[0] == 'cons':
            if i < len(word) and word[i] in CONSONANTS:
                helper(i+1, pi+1, bindings)

        elif part[0] == 'star':
            for j in range(i, len(word)+1):  # include j = i for empty string match
                helper(j, pi+1, bindings)
                if not all_matches and results:
                    return

        elif part[0] == 'var':
            name = part[1]
            if name in bindings:
                val = bindings[name]
                if word.startswith(val, i):
                    helper(i+len(val), pi+1, bindings)
            else:
                maxlen = len(word) - i
                for L in range(1, maxlen+1):
                    val = word[i:i+L]
                    if is_valid_binding(name, val, var_constraints):
                        new_bindings = dict(bindings)
                        new_bindings[name] = val
                        helper(i+L, pi+1, new_bindings)
                        if not all_matches and results:
                            return

        elif part[0] == 'rev':
            name = part[1]
            if name in bindings:
                revval = bindings[name][::-1]
                if word.startswith(revval, i):
                    helper(i+len(revval), pi+1, bindings)
            else:
                maxlen = len(word) - i
                for L in range(1, maxlen+1):
                    revval = word[i:i+L]
                    val = revval[::-1]
                    if is_valid_binding(name, val, var_constraints):
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
