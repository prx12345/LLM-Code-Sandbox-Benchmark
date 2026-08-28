"""Sample candidate: *intentionally* buggy solution.

Contains the classic sliding-window mistake LLMs (and humans) make on this
problem: on a repeated character it moves the window start to
``last_seen[char] + 1`` **unconditionally**, which can move the window
*backwards* when the previous occurrence lies before the current window
(e.g. ``"abba"`` → returns 3 instead of 2).

Included so the harness can demonstrate failure detection, the FAIL verdict
path, and the failure-details section of the scorecard.
"""

from __future__ import annotations


def length_of_longest_substring(s: str) -> int:
    """Buggy variant: window start can move backwards on stale duplicates."""
    last_seen: dict[str, int] = {}
    window_start = 0
    best = 0
    for index, char in enumerate(s):
        if char in last_seen:  # BUG: ignores whether the match is inside the window
            window_start = last_seen[char] + 1
        last_seen[char] = index
        best = max(best, index - window_start + 1)
    return best
