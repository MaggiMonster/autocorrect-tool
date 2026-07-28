# Autocorrect Tool — NLP + Dynamic Programming

A spelling autocorrect engine trained on the complete works of Shakespeare.
It suggests the most likely correct word for a misspelling using
**Levenshtein edit distance (computed via dynamic programming)**, ranked by
**word-frequency probability** learned from the corpus.

```
> thnie
Suggestions for 'thnie':
  1. thine           edit distance=1  P(word)=0.000513

> wherefre
Suggestions for 'wherefre':
  1. wherefore       edit distance=1  P(word)=0.000148
```

## How it works

1. **Corpus & frequency model** ([autocorrect/corpus.py](autocorrect/corpus.py))
   The full text of *The Complete Works of William Shakespeare* (public
   domain, via [Project Gutenberg](https://www.gutenberg.org/ebooks/100)) is
   tokenized and counted into a word-frequency table. This gives P(word) —
   the probability of any given word, estimated by how often it appears in
   the corpus.

2. **Levenshtein distance via dynamic programming** ([autocorrect/edit_distance.py](autocorrect/edit_distance.py))
   The classic DP edit-distance algorithm computes the minimum number of
   single-character insertions, deletions, and substitutions needed to turn
   one word into another, using a rolling-row implementation (O(n·m) time,
   O(min(n,m)) space).

3. **Candidate generation, optimized to 1- and 2-edit transformations** ([autocorrect/corrector.py](autocorrect/corrector.py))
   Running the DP edit-distance function against every word in a ~24,000-word
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
   frequently they occur in Shakespeare's text — rather than an arbitrary
   candidate that happens to share the same edit distance.

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
python -m autocorrect.cli "wherefre"

# Interactive session
python -m autocorrect.cli

# Show more suggestions
python -m autocorrect.cli "wich" -n 5
```

### As a library

```python
from autocorrect import Autocorrect

model = Autocorrect.from_corpus()

model.correct("thnie")          # -> "thine"

for s in model.suggest("wich", limit=3):
    print(s.word, s.edit_distance, s.probability)
# with   1  0.008620
# which  1  0.002600
# wish   1  0.000270
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
│   └── shakespeare.txt    # training corpus (public domain)
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

`data/shakespeare.txt` is *The Complete Works of William Shakespeare*
([Project Gutenberg eBook #100](https://www.gutenberg.org/ebooks/100)),
public domain in the United States.

## License

MIT — see [LICENSE](LICENSE).
