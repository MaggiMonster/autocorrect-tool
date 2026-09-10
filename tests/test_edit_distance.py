from autocorrect.edit_distance import levenshtein_distance


def test_identical_words():
    assert levenshtein_distance("holmes", "holmes") == 0


def test_single_substitution():
    assert levenshtein_distance("hallo", "hello") == 1


def test_single_insertion():
    assert levenshtein_distance("cat", "cats") == 1


def test_single_deletion():
    assert levenshtein_distance("cats", "cat") == 1


def test_completely_different_words():
    assert levenshtein_distance("kitten", "sitting") == 3


def test_empty_strings():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("", "abc") == 3


def test_symmetry():
    assert levenshtein_distance("flower", "lower") == levenshtein_distance(
        "lower", "flower"
    )
