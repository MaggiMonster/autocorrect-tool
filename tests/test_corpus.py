from autocorrect.corpus import WordFrequency, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("To be, or not to be!") == ["to", "be", "or", "not", "to", "be"]


def test_tokenize_keeps_internal_apostrophes():
    assert tokenize("op'd o'er cut") == ["op'd", "o'er", "cut"]


def test_word_frequency_counts_and_probability():
    wf = WordFrequency(["to", "be", "or", "not", "to", "be"])
    assert wf.frequency("to") == 2
    assert wf.frequency("or") == 1
    assert wf.probability("to") == 2 / 6
    assert wf.probability("nonexistent") == 0.0


def test_known_filters_to_vocabulary():
    wf = WordFrequency(["thou", "art", "thou"])
    assert wf.known(["thou", "ghost", "art"]) == {"thou", "art"}
