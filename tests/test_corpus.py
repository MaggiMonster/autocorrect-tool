from autocorrect.corpus import WordFrequency, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("You know my methods, Watson!") == [
        "you",
        "know",
        "my",
        "methods",
        "watson",
    ]


def test_tokenize_keeps_internal_apostrophes():
    assert tokenize("don't o'clock Watson's") == ["don't", "o'clock", "watson's"]


def test_word_frequency_counts_and_probability():
    wf = WordFrequency(["the", "game", "is", "afoot", "the", "game"])
    assert wf.frequency("the") == 2
    assert wf.frequency("is") == 1
    assert wf.probability("the") == 2 / 6
    assert wf.probability("nonexistent") == 0.0


def test_known_filters_to_vocabulary():
    wf = WordFrequency(["holmes", "watson", "holmes"])
    assert wf.known(["holmes", "moriarty", "watson"]) == {"holmes", "watson"}
