import re
from typing import List, Dict, Tuple

NAMED_GROUP_PATTERN = r"\(\?P<\w+>.*?\)"
NAMED_GROUP_COMPILED = re.compile(r"\(\?P<(\w+)>")
BACKREF_PATTERN = r"\\\d+"
BACKREF_COMPILED = re.compile(r"\\(\d+)")

CAPITAL_GREEK_LETTERS = [None, "Γ", "Δ", "Λ", "Ξ", "Π", "Σ", "Φ", "Ψ", "Ω", "Θ"]

def extract_backrefs(regex):
    """
    Extract backreferences from a regex pattern
    return a dictionary mapping backreference indices to their referenced group names
    """
    named_groups = {i+1: name for i, name in enumerate(NAMED_GROUP_COMPILED.findall(regex))}
    backrefs = {}
    for m in BACKREF_COMPILED.finditer(regex):
        backref_index = int(m.group(1))
        if backref_index in named_groups:
            backrefs[backref_index] = named_groups[backref_index]
    return backrefs
        

def split_named_regex(pattern: str):
    """
    Split a regex pattern into an array where named groups and other parts
    are separate elements.
   
    :param pattern: The regex pattern to split.
    :return: A list of strings with named groups and other components.
    """
    # Regex to match named groups like (?P<name>...)
    
    pattern_for_names = f"({NAMED_GROUP_PATTERN})|({BACKREF_PATTERN})"
    pattern_for_split = r"(\(\?P<\w+>.*?\)|\\\d+)"
    
    # Find all named groups
    named_groups = re.findall(pattern_for_names, pattern)
    named_groups = [item for pair in named_groups for item in pair if item]
   
    # Split the pattern at the named groups, keeping the separators
    result = re.split(pattern_for_split, pattern)
    # Filter out components that explicitly match, but keep empty strings
    parts = [part for part in result if part != None and not re.fullmatch(pattern_for_split, part)]
   
    # Interleave the split parts and the named groups
    result = []
    for part, group in zip(parts, named_groups + [""]):  # Add an empty string to handle the last split
        if part:
            result.append(part)
        if group:
            result.append(group)
   
    return result

def parse_pattern(pattern):
    """
    Parse a regex into "group" and "fixed" components
    Return an array of "group" or "fixed" components
    and optionally a "backref" component
    """
    # Pull out backrefs, if any
    backrefs = extract_backrefs(pattern)
    # split the regex
    split_regex = split_named_regex(pattern)
    arr = []
    named = ''
    for patt in split_regex:
        if re.fullmatch(NAMED_GROUP_PATTERN, patt):
            named += patt[4]
        elif BACKREF_COMPILED.fullmatch(patt):
            letter = CAPITAL_GREEK_LETTERS[int(BACKREF_COMPILED.fullmatch(patt).group(1))]
            named += letter
        else:
            if named:
                arr.append(('group', named))
                named = ''
            arr.append(('fixed', patt.upper()))
    
    if named:
        arr.append(('group', named))
        
    backrefs_greek = {CAPITAL_GREEK_LETTERS[k]: v for k, v in backrefs.items()}

    return arr, backrefs_greek

def find_fixed_positions(word: str, fixed_components: List[str]) -> List[List[Tuple[int, int]]]:
    """
    Find all start and end positions of fixed regex components in the word, including overlapping matches.
    :param word: The input word.
    :param fixed_components: List of fixed regex patterns.
    :return: A list of lists, where each inner list contains tuples of (start, end)
             positions of matches for a fixed regex component, including overlapping matches.
    """
    positions = []
    for component in fixed_components:
        regex = re.compile(component)
        matches = []
        # Manually check for overlapping matches
        for i in range(len(word)):
            match = regex.match(word[i:])
            if match:
                matches.append((i, i + len(match.group())))
        positions.append(matches)
    return positions

def find_regex_partitions_fixed(word: str, components: List[Tuple[str, str]], lengths: dict = {}, backrefs: dict = {}) -> List[Dict[str, str]]:
    """
    Partition a word into substrings matching a parsed regex pattern with mixed group and fixed components,
    ensuring fixed components fully match before proceeding to the next group.
    :param word: The input word to partition.
    :param components: Parsed components of the regex pattern.
    :return: A list of dictionaries with matched named groups.
    """
    # Separate fixed and group components
    fixed_components = [(i, c[1]) for i, c in enumerate(components) if c[0] == 'fixed']
    group_components = [(i, c[1]) for i, c in enumerate(components) if c[0] == 'group']
    
    last_fixed_idx = max((i for i, _ in fixed_components), default=-1)
    
    # get the "group" letters
    group_letters = frozenset([_[1] for _ in group_components])
    
    # Precompute positions of fixed components with overlapping matches
    fixed_positions = find_fixed_positions(word, [fc[1] for fc in fixed_components])

    # Ensure the positions are sequential
    valid_combinations = []
    def generate_combinations(index=0, current=[]):
        if index == len(fixed_positions):
            valid_combinations.append(current[:])
            return
        last_position = current[-1][1] if current else -1
        for start, end in fixed_positions[index]:
            if start > last_position:  # Ensure sequential order
                current.append((start, end))
                generate_combinations(index + 1, current)
                current.pop()
    
    generate_combinations()

    # Build partitions from valid combinations
    results = []
    for combination in valid_combinations:
        partition = {}
        current_start = 0
        group_index = 0
        for fixed_index, (start, end) in enumerate(combination):
            # Assign named groups before the current fixed component
            while group_index < len(group_components) and group_components[group_index][0] < fixed_components[fixed_index][0]:
                group_name = group_components[group_index][1]
                if current_start < start:  # Ensure non-empty group substrings
                    partition[group_name] = word[current_start:start]
                    current_start = start
                    group_index += 1
                else:
                    break

            # Move past the fixed component
            fixed_name = f"fixed_{fixed_index}"
            partition[fixed_name] = word[start:end]
            current_start = end

        # Assign any remaining named groups that actually come *after* the last fixed component
        if group_index < len(group_components):
            for comp_idx, group_name in group_components[group_index:]:
                # only consume it if it lives after that fixed component
                if comp_idx > last_fixed_idx and current_start < len(word):
                    partition[group_name] = word[current_start:]
                    current_start = len(word)

        # add the word itself
        #partition["word"] = word
        results.append(partition)

    # Filter out invalid results
    def is_valid_partition(partition):
        # Concatenate all values and compare to the original word
        reconstructed = ''.join(partition.values())
        if reconstructed != word:
            return False
        # also check that all group letters are keys
        return group_letters.issubset(set(partition.keys()))
    
    results = [partition | {"word": word} for partition in results if is_valid_partition(partition)]
    
    # If we have any "multiple" named groups, split them
    final_results = results
    multiple_groups = [_[1] for _ in group_components if len(_[1]) > 1]
    if multiple_groups:
        final_results = []
        for group_name in multiple_groups:
            for r in results:
                num_partitions = len(group_name)
                if len(r[group_name]) >= num_partitions:
                    this_str = r.pop(group_name, None)
                    for p in generate_partitions(this_str, num_partitions):
                        d = dict((group_name[i], p[i]) for i in range(num_partitions))
                        d.update(r)
                        final_results.append(d)
                        
    
    # Check that the lengths line up
    final_results2 = final_results
    if lengths:
        final_results2 = []
    for part in final_results:
        for k, v in lengths.items():
            if len(part[k]) != v:
                continue
        final_results2.append(part)
    
    # We need to do a check for backreferences
    if not backrefs:
        return final_results2
    actual_final_results = []
    for k, v in backrefs.items():
        for part in final_results2:
            if part.get(k) == part.get(v):
                actual_final_results.append(part)

    return actual_final_results

def split_word_on_pattern(word, pattern):
    regex = pattern.regex
    # Assume if a variable has a length, it is fixed
    # TODO: do we need to change this down the road?
    lengths = dict((k, v[0]) for k, v in pattern.lengths.items() if re.fullmatch(r'[A-Z]', k))
    # if there are no named components, return something simple
    if not re.search(NAMED_GROUP_PATTERN, regex):
        return [{"word": word}]
    parsed_components, backrefs = parse_pattern(regex)
    result = find_regex_partitions_fixed(word, parsed_components, lengths=lengths, backrefs=backrefs)
    return result

def generate_partitions(string: str, num_partitions: int) -> List[List[str]]:
    """
    Generate all non-zero-length partitions of a string into a given number of partitions.
    :param string: The input string to partition.
    :param num_partitions: The number of partitions to generate.
    :return: A list of partitions, each represented as a list of substrings.
    """
    def helper(start: int, partitions: List[str]):
        # If we've filled the required number of partitions
        if len(partitions) == num_partitions - 1:
            # The last partition takes the remaining substring
            remaining = string[start:]
            if remaining:  # Ensure it's non-zero-length
                results.append(partitions + [remaining])
            return
        
        # Iterate over possible splits for the current partition
        for end in range(start + 1, len(string) - (num_partitions - len(partitions) - 1) + 1):
            helper(end, partitions + [string[start:end]])

    results = []
    helper(0, [])
    return results
