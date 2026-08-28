"""Sample candidate under evaluation (plays the role of AI-generated code).

Problem: **Longest Substring Without Repeating Characters** (LeetCode #3).
This is the reference-quality submission: a sliding-window solution running
in O(n) time and O(min(n, alphabet)) space.
"""

from __future__ import annotations


def length_of_longest_substring(s: str) -> int:
    """Return the length of the longest substring of *s* with unique characters.

    Uses a sliding window over ``s`` while tracking the most recent index of
    every character. When a character repeats inside the current window, the
    window start jumps just past its previous occurrence.

    Args:
        s: Input string (may be empty; any Unicode characters).

    Returns:
        Length of the longest run of pairwise-distinct characters.

    Examples:
        >>> length_of_longest_substring("abcabcbb")
        3
        >>> length_of_longest_substring("")
        0
    """
    last_seen: dict[str, int] = {}
    window_start = 0
    best = 0
    for index, char in enumerate(s):
        previous = last_seen.get(char)
        if previous is not None and previous >= window_start:
            window_start = previous + 1
        last_seen[char] = index
        best = max(best, index - window_start + 1)
    return best
