# Autocorrect Tool — Project Documentation

## 1. Overview

**What it is:** an NLP spelling-correction engine that takes a misspelled
word and returns ranked suggestions for the intended word.

**How it decides:** two signals, combined —

1. **Edit distance** — how many single-character changes separate the typo
   from a candidate word (computed with a dynamic-programming Levenshtein
   distance algorithm).
2. **Word frequency** — how common the candidate word is in a training
   corpus, used as a probability estimate of how likely it is to be the
   intended word.

**Training data:** seven Arthur Conan Doyle "Sherlock Holmes" books (public
domain, via Project Gutenberg), tokenized into a vocabulary of **17,096
unique words** across **524,118 total word occurrences**.

**Language/stack:** pure Python 3.9+, standard library only (no ML
frameworks, no third-party runtime dependencies). `pytest` for testing.

**Repository:** [github.com/MaggiMonster/autocorrect-tool](https://github.com/MaggiMonster/autocorrect-tool)

---

## 2. Problem statement

Given a possibly-misspelled word (e.g. `"wathson"`), return the most likely
intended word(s) from a known vocabulary (e.g. `"watson"`), ranked by
plausibility — without needing a hand-built dictionary of common
misspellings, and without needing to check every word in the vocabulary
against the input for every query.

---

## 3. Architecture

```
                     ┌─────────────────────┐
                     │  data/               │
                     │  sherlock_holmes.txt │  (raw corpus text)
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │  corpus.py            │
                     │  tokenize()            │  → lowercased word tokens
                     │  WordFrequency          │  → Counter + P(word)
                     └──────────┬───────────┘
                                │
   ┌────────────────────────────▼────────────────────────────┐
   │  corrector.py                                             │
   │  edits1(word) / edits2(word)  → candidate strings          │
   │  Autocorrect.suggest(word)     → ranked Suggestion list     │
   │    (falls back to edit_distance.py's DP function            │
   │     only when nothing is found within 2 edits)                │
   └───────────────────────────┬────────────────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │  edit_distance.py      │
                     │  levenshtein_distance() │  → DP edit distance
                     └────────────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │  cli.py                │  → command-line front end
                     └────────────────────────┘
```

Data flows one direction: raw text → tokens → frequency table → candidate
generation → ranked suggestions → printed output. The DP edit-distance
function is a leaf dependency used in two places: as the algorithmic
justification/fallback in `corrector.py`, and directly testable on its own
in `edit_distance.py`.

---

## 4. Module-by-module reference

### 4.1 `autocorrect/edit_distance.py`

| Function | Purpose |
|---|---|
| `levenshtein_distance(a, b)` | Computes minimum edit distance between two strings via bottom-up dynamic programming. |
| `cached_levenshtein_distance(a, b)` | Same, memoized with `functools.lru_cache` — used in the fallback path where repeated comparisons against the same vocabulary are likely. |

**Algorithm.** Classic Levenshtein DP. Let `dp[i][j]` = edit distance
between the first `i` characters of word A and first `j` characters of
word B. Recurrence:

```
dp[i][j] = dp[i-1][j-1]                          if A[i] == B[j]
         = 1 + min(dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1])  # substitute
```

**Space optimization implemented:** instead of a full `n × m` matrix, only
the previous row is kept (`previous_row` / `current_row` in the code),
since row `i` only ever depends on row `i-1`. This reduces space complexity
from `O(n·m)` to `O(min(n,m))` (the shorter word is swapped to the
`word_b` position to guarantee the smaller row).

**Complexity:** Time `O(n·m)`, Space `O(min(n,m))`, where `n`, `m` are the
lengths of the two words.

---

### 4.2 `autocorrect/corpus.py`

| Component | Purpose |
|---|---|
| `tokenize(text)` | Lowercases text and extracts word tokens via regex `[a-z]+(?:'[a-z]+)?`, preserving internal apostrophes so contractions (`"don't"`, `"it's"`) stay as single tokens. |
| `load_corpus(path)` | Reads the raw corpus text file from disk. |
| `WordFrequency` | Wraps a `collections.Counter` over all tokens; the frequency/probability model. |

**`WordFrequency` API:**

- `.vocabulary` — the set of all known words.
- `.known(words)` — filters an iterable down to only words present in the corpus.
- `.probability(word)` — `count(word) / total_tokens`, a maximum-likelihood
  unigram probability estimate, P(word).
- `.frequency(word)` — raw occurrence count.

This is the **language model** half of the system — it doesn't know
anything about spelling or edit distance, only about how likely any given
word is to occur, based on the training corpus.

---

### 4.3 `autocorrect/corrector.py`

The core decision-making module. Two responsibilities:

**A. Candidate generation (`edits1`, `edits2`)**

`edits1(word)` generates every string reachable from `word` by exactly one
of:
- **deletion** — remove one character
- **transposition** — swap two adjacent characters
- **substitution** — replace one character with each letter a–z
- **insertion** — insert each letter a–z at every position

This produces `O(26 · L)` candidate strings for a word of length `L`
(roughly 234 candidates for a 5-letter word).

`edits2(word)` applies `edits1` a second time to every result of
`edits1(word)`, producing every string reachable within exactly two edits.

**Why generate instead of scan:** the alternative — computing the DP edit
distance between the input and *every* word in the 17,096-word vocabulary —
would mean 17,096 DP calls per lookup. Generating edit-distance-1/2
candidates and checking them against the vocabulary via hash-set lookup
(`WordFrequency.known()`) finds the same set of nearby known words, but
each check is O(1) instead of O(n·m), and most generated strings are
discarded instantly rather than fully scored.

**B. Ranking (`Autocorrect.suggest`)**

```
def suggest(word):
    if word in vocabulary:
        return [word] at distance 0

    candidates = known(edits1(word))
    if candidates:
        rank by frequency, return              # distance-1 bucket

    candidates = known(edits2(word))
    if candidates:
        rank by frequency, return              # distance-2 bucket

    # nothing known within 2 edits — brute-force fallback
    compute levenshtein_distance(word, v) for every v in vocabulary
    return the words with minimum distance, ranked by frequency
```

Ranking logic: **edit distance is the primary sort key** (fewer edits =
structurally closer to what was typed), and **frequency probability is
the secondary/tie-break key** within a distance bucket (of several equally
plausible typo-corrections, prefer the one that's actually a common word).
This mirrors the "noisy channel" idea behind Peter Norvig's well-known
spelling corrector: `argmax P(candidate) among candidates similar to input`.

**`Suggestion` dataclass:** `(word, edit_distance, probability)` — returned
to callers so the UI/caller can see *why* a suggestion was ranked where it
was, not just the final word.

---

### 4.4 `autocorrect/cli.py`

Thin `argparse`-based command-line wrapper around `Autocorrect`.

- `python -m autocorrect.cli "word"` — one-shot correction, prints ranked
  suggestions to stdout.
- `python -m autocorrect.cli` (no argument) — interactive REPL; type words,
  `quit`/`exit`/Ctrl+C to stop.
- `-n / --num-suggestions` — how many ranked suggestions to display
  (default 5).

No algorithmic logic lives here — it only formats and prints results from
`Autocorrect.suggest()`.

---

## 5. Design decisions & rationale

| Decision | Why |
|---|---|
| Generate edits1/edits2 instead of scanning the vocabulary with DP | Avoids O(vocab_size) DP calls per query; reduces the common case to O(1) hash lookups. |
| DP edit-distance kept as its own module, used only in the fallback | Keeps the "textbook" DP algorithm implemented and testable in isolation (as the resume claims), while not being on the hot path. |
| Rolling-row DP (O(min(n,m)) space) instead of full matrix | Standard space optimization; only the previous row is ever needed for the recurrence. |
| Unigram frequency model (no bigrams/context) | Keeps the model simple and fast to train/query; sufficient for single-word correction without surrounding context. |
| Public-domain literary corpus (Project Gutenberg) | Free, legally reusable, large enough (~500K tokens) to build a meaningful frequency distribution, and easy to swap for another corpus without touching the algorithm. |
| No third-party runtime dependencies | Keeps the tool trivially installable/portable; only `pytest` is needed, and only for development. |
| `from __future__ import annotations` throughout | Lets modern `list[str]`, `X | None` type hints work on Python 3.9, without requiring 3.10+. |

---

## 6. Testing

18 unit tests across 3 files, run with `pytest`:

- **`tests/test_edit_distance.py`** — correctness of the DP algorithm:
  identical strings, single insert/delete/substitute, multi-edit cases
  (`"kitten"` → `"sitting"` = 3), symmetry, empty-string edge cases.
- **`tests/test_corpus.py`** — tokenizer behavior (lowercasing, punctuation
  stripping, apostrophe handling) and `WordFrequency` counting/probability
  math, using small synthetic token lists (not the full corpus, for
  determinism and speed).
- **`tests/test_corrector.py`** — `edits1`/`edits2` generation correctness,
  frequency-based ranking, the zero-distance/known-word shortcut, and the
  brute-force fallback path, all against a small in-memory toy vocabulary.

Tests intentionally avoid depending on the real corpus file's exact
contents (aside from a couple of manual/CLI smoke checks done during
development) so they stay fast and stable if the corpus is swapped again.

---

## 7. Setup & usage

```bash
git clone https://github.com/MaggiMonster/autocorrect-tool.git
cd autocorrect-tool
pip install -e ".[dev]"       # installs pytest for testing; no runtime deps

pytest                         # run the test suite

python -m autocorrect.cli "wathson"       # -> watson
python -m autocorrect.cli                 # interactive mode
```

```python
from autocorrect import Autocorrect

model = Autocorrect.from_corpus()
model.correct("wathson")                  # "watson"
model.suggest("wich", limit=3)            # ranked Suggestion objects
```

---

## 8. Data provenance

`data/sherlock_holmes.txt` is a concatenation of seven Arthur Conan Doyle
works, all public domain in the U.S., downloaded from Project Gutenberg and
stripped of Gutenberg's license header/footer boilerplate:

1. The Adventures of Sherlock Holmes (Gutenberg #1661)
2. The Return of Sherlock Holmes (#108)
3. The Sign of the Four (#2097)
4. A Study in Scarlet (#244)
5. The Valley of Fear (#3289)
6. The Memoirs of Sherlock Holmes (#834)
7. His Last Bow (#2350)

Swapping the corpus only touches `corpus.py`'s `DEFAULT_CORPUS_PATH` and
the data file — the algorithm and its guarantees are unchanged.

---

## 9. Possible future improvements

(Not implemented — noted here as natural extensions if asked in an
interview or revisited later.)

- **Bigram/trigram context model** — currently ranks purely on unigram
  frequency; a real autocorrect (e.g. phone keyboards) also weighs the
  surrounding words ("their" vs "there" depends on context).
- **BK-tree or trie-based vocabulary index** — would make the edits2
  candidate explosion and the brute-force fallback scale better on a much
  larger dictionary.
- **Keyboard-layout-aware substitution costs** — weight substitutions of
  adjacent keys (e.g. 'e'↔'r') as more likely than distant ones, instead of
  treating all 26 substitutions as equally probable.
- **Larger/more diverse corpus** — blend multiple public-domain sources to
  reduce bias toward one author's vocabulary and style.
