"""Autocorrect Tool — NLP + dynamic programming spelling corrector."""

from .edit_distance import levenshtein_distance
from .corpus import WordFrequency, load_corpus
from .corrector import Autocorrect, Suggestion

__all__ = [
    "levenshtein_distance",
    "WordFrequency",
    "load_corpus",
    "Autocorrect",
    "Suggestion",
]

__version__ = "1.0.0"
