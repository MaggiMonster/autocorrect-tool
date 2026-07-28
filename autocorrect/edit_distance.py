"""Levenshtein edit distance via dynamic programming."""

from functools import lru_cache


def levenshtein_distance(word_a: str, word_b: str) -> int:
    """Minimum number of single-character insertions, deletions, or
    substitutions needed to turn ``word_a`` into ``word_b``.

    Computed bottom-up with an O(len(word_a)) rolling row instead of a full
    O(n*m) matrix, since only the previous row is ever needed.
    """
    if word_a == word_b:
        return 0
    if len(word_a) < len(word_b):
        word_a, word_b = word_b, word_a

    previous_row = list(range(len(word_b) + 1))
    for i, char_a in enumerate(word_a, start=1):
        current_row = [i] + [0] * len(word_b)
        for j, char_b in enumerate(word_b, start=1):
            deletion = previous_row[j] + 1
            insertion = current_row[j - 1] + 1
            substitution = previous_row[j - 1] + (char_a != char_b)
            current_row[j] = min(deletion, insertion, substitution)
        previous_row = current_row

    return previous_row[-1]


@lru_cache(maxsize=100_000)
def cached_levenshtein_distance(word_a: str, word_b: str) -> int:
    """Memoized wrapper around :func:`levenshtein_distance`.

    Useful when the same word gets compared against a fixed vocabulary
    repeatedly (e.g. the brute-force fallback in ``Autocorrect``).
    """
    return levenshtein_distance(word_a, word_b)
