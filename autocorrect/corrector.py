"""Candidate generation and ranking for the autocorrect model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .corpus import WordFrequency, DEFAULT_CORPUS_PATH
from .edit_distance import cached_levenshtein_distance

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def edits1(word: str) -> set[str]:
    """All strings reachable from ``word`` by a single insertion, deletion,
    substitution, or transposition — i.e. every word at edit distance 1.

    Generating candidates this way and checking them against the known
    vocabulary is far cheaper than computing the Levenshtein distance (DP)
    between ``word`` and every word in the dictionary.
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [left + right[1:] for left, right in splits if right]
    transposes = [
        left + right[1] + right[0] + right[2:]
        for left, right in splits
        if len(right) > 1
    ]
    replaces = [
        left + c + right[1:]
        for left, right in splits
        if right
        for c in ALPHABET
    ]
    inserts = [left + c + right for left, right in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def edits2(word: str) -> set[str]:
    """All strings reachable from ``word`` within edit distance 2."""
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


@dataclass
class Suggestion:
    word: str
    edit_distance: int
    probability: float


class Autocorrect:
    """Suggests corrections for a misspelled word.

    Ranking is primarily by edit distance (fewer edits = more likely typo),
    computed cheaply via ``edits1``/``edits2`` generation, with word
    frequency probability from the corpus used to rank candidates that tie
    on distance. Only if no word within edit distance 2 is known does it
    fall back to a brute-force Levenshtein (DP) scan of the vocabulary.
    """

    def __init__(self, word_frequency: WordFrequency):
        self.word_frequency = word_frequency

    @classmethod
    def from_corpus(cls, path: Path = DEFAULT_CORPUS_PATH) -> "Autocorrect":
        return cls(WordFrequency.from_corpus(path))

    def is_known(self, word: str) -> bool:
        return word in self.word_frequency.vocabulary

    def _rank(self, words: set[str], distance: int, limit: int) -> list[Suggestion]:
        ranked = sorted(
            words, key=lambda w: self.word_frequency.probability(w), reverse=True
        )
        return [
            Suggestion(w, distance, self.word_frequency.probability(w))
            for w in ranked[:limit]
        ]

    def _fallback_candidates(self, word: str, limit: int) -> list[Suggestion]:
        """Brute-force nearest-neighbor search using the DP edit distance,
        used only when the word is unrecognizable within 2 edits."""
        scored = [
            (cached_levenshtein_distance(word, vocab_word), vocab_word)
            for vocab_word in self.word_frequency.vocabulary
        ]
        if not scored:
            return []
        min_distance = min(d for d, _ in scored)
        nearest = {w for d, w in scored if d == min_distance}
        return self._rank(nearest, min_distance, limit)

    def suggest(self, word: str, limit: int = 5) -> list[Suggestion]:
        """Return up to ``limit`` ranked corrections for ``word``."""
        word = word.lower()

        if self.is_known(word):
            return [Suggestion(word, 0, self.word_frequency.probability(word))]

        candidates_1 = self.word_frequency.known(edits1(word))
        if candidates_1:
            return self._rank(candidates_1, 1, limit)

        candidates_2 = self.word_frequency.known(edits2(word))
        if candidates_2:
            return self._rank(candidates_2, 2, limit)

        return self._fallback_candidates(word, limit)

    def correct(self, word: str) -> str:
        """Return the single most likely correction for ``word``."""
        suggestions = self.suggest(word, limit=1)
        return suggestions[0].word if suggestions else word
