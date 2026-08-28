"""Sample candidate: *intentionally* naive brute-force solution.

Functionally correct but O(n^2) in the worst case (a string of all-distinct
characters). Included so the harness can demonstrate that two candidates with
identical accuracy are still distinguishable by runtime scaling — the
empirical complexity estimator should label this one **O(n^2)** while
``candidate_solution.py`` comes out **O(n)**.
"""

from __future__ import annotations


def length_of_longest_substring(s: str) -> int:
    """Return the length of the longest substring of *s* with unique characters.

    For every start index, extend a set of seen characters until the first
    duplicate. Correct, but quadratic when long unique runs exist.
    """
    best = 0
    for start in range(len(s)):
        seen: set[str] = set()
        for char in s[start:]:
            if char in seen:
                break
            seen.add(char)
        best = max(best, len(seen))
    return best
