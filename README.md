# Autocorrect Tool — NLP + Dynamic Programming

A spelling autocorrect engine trained on the Sherlock Holmes canon (Arthur
Conan Doyle). It suggests the most likely correct word for a misspelling
using **Levenshtein edit distance (computed via dynamic programming)**,
ranked by **word-frequency probability** learned from the corpus.

```
> ellementary
Suggestions for 'ellementary':
  1. elementary      edit distance=1  P(word)=0.000010

> wathson
Suggestions for 'wathson':
  1. watson          edit distance=1  P(word)=0.001334
```

## How it works

1. **Corpus & frequency model** ([autocorrect/corpus.py](autocorrect/corpus.py))
   Seven Sherlock Holmes books (public domain, via
   [Project Gutenberg](https://www.gutenberg.org)) are tokenized and counted
   into a word-frequency table. This gives P(word) — the probability of any
   given word, estimated by how often it appears in the corpus.

2. **Levenshtein distance via dynamic programming** ([autocorrect/edit_distance.py](autocorrect/edit_distance.py))
   The classic DP edit-distance algorithm computes the minimum number of
   single-character insertions, deletions, and substitutions needed to turn
   one word into another, using a rolling-row implementation (O(n·m) time,
   O(min(n,m)) space).

3. **Candidate generation, optimized to 1- and 2-edit transformations** ([autocorrect/corrector.py](autocorrect/corrector.py))
   Running the DP edit-distance function against every word in a ~17,000-word
   vocabulary for every keystroke would be wasteful. Instead, the corrector
   *generates* every string reachable from the input by a single insertion,
   deletion, substitution, or transposition (`edits1`), and — only if none of
   those are real words — every string reachable within two edits
   (`edits2`). Checking generated candidates against the vocabulary is far
   cheaper than brute-force scanning it, while still being equivalent to
   "find all known words within edit distance ≤ 2."

4. **Ranking by frequency probability**
   Among the known candidates at the closest edit distance, the corrector
   picks the ones most likely to be the *intended* word — ranked by how
   frequently they occur in the corpus — rather than an arbitrary candidate
   that happens to share the same edit distance.

5. **Fallback**
   If no known word is found within 2 edits, the corrector falls back to a
   brute-force nearest-neighbor search using the DP edit-distance function
   directly against the full vocabulary, so it always returns its best guess.

## Installation

```bash
git clone https://github.com/MaggiMonster/autocorrect-tool.git
cd autocorrect-tool
pip install -e ".[dev]"
```

Requires Python 3.9+. No third-party runtime dependencies — only the
standard library.

## Usage

### Command line

```bash
# One-shot correction
python -m autocorrect.cli "ellementary"

# Interactive session
python -m autocorrect.cli

# Show more suggestions
python -m autocorrect.cli "wich" -n 5
```

### As a library

```python
from autocorrect import Autocorrect

model = Autocorrect.from_corpus()

model.correct("wathson")        # -> "watson"

for s in model.suggest("wich", limit=3):
    print(s.word, s.edit_distance, s.probability)
# with   1  0.007876
# which  1  0.006483
# wish   1  0.000250
```

## Project structure

```
autocorrect-tool/
├── autocorrect/
│   ├── __init__.py        # public API
│   ├── edit_distance.py   # Levenshtein distance (DP)
│   ├── corpus.py          # tokenizer + word-frequency model
│   ├── corrector.py        # candidate generation + ranking
│   └── cli.py               # command-line interface
├── data/
│   └── sherlock_holmes.txt  # training corpus (public domain)
├── tests/
│   ├── test_edit_distance.py
│   ├── test_corpus.py
│   └── test_corrector.py
├── pyproject.toml
└── requirements.txt
```

## Running tests

```bash
pip install -r requirements.txt
pytest
```

## Data source

`data/sherlock_holmes.txt` combines seven Arthur Conan Doyle novels/story
collections, all public domain in the United States, from Project Gutenberg:

- [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661)
- [The Return of Sherlock Holmes](https://www.gutenberg.org/ebooks/108)
- [The Sign of the Four](https://www.gutenberg.org/ebooks/2097)
- [A Study in Scarlet](https://www.gutenberg.org/ebooks/244)
- [The Valley of Fear](https://www.gutenberg.org/ebooks/3289)
- [The Memoirs of Sherlock Holmes](https://www.gutenberg.org/ebooks/834)
- [His Last Bow](https://www.gutenberg.org/ebooks/2350)

## License

MIT — see [LICENSE](LICENSE).
