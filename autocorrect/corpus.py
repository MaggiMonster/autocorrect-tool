"""Corpus loading and word-frequency modeling."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "shakespeare.txt"


def tokenize(text: str) -> list[str]:
    """Lowercase and split raw text into word tokens.

    Keeps internal apostrophes (e.g. "thou'rt", "th'") since they're common
    in Early Modern English and matter for matching real Shakespearean words.
    """
    return _WORD_RE.findall(text.lower())


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


class WordFrequency:
    """Word-frequency probability model built from a text corpus."""

    def __init__(self, tokens: list[str]):
        self.counts = Counter(tokens)
        self.total = sum(self.counts.values())

    @classmethod
    def from_corpus(cls, path: Path = DEFAULT_CORPUS_PATH) -> "WordFrequency":
        return cls(tokenize(load_corpus(path)))

    @property
    def vocabulary(self) -> set[str]:
        return set(self.counts.keys())

    def known(self, words) -> set[str]:
        """Subset of ``words`` that actually occur in the corpus."""
        return {w for w in words if w in self.counts}

    def probability(self, word: str) -> float:
        """P(word) estimated by relative frequency in the corpus."""
        if self.total == 0:
            return 0.0
        return self.counts[word] / self.total

    def frequency(self, word: str) -> int:
        return self.counts[word]

    def __len__(self) -> int:
        return len(self.counts)
