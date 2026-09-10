import pytest

from autocorrect.corpus import WordFrequency
from autocorrect.corrector import Autocorrect, edits1, edits2


@pytest.fixture
def toy_model():
    tokens = (
        ["holmes"] * 50
        + ["watson"] * 30
        + ["lestrade"] * 10
        + ["moor"] * 25
        + ["door"] * 5
        + ["london"] * 20
    )
    return Autocorrect(WordFrequency(tokens))


def test_edits1_includes_known_single_edit_typos():
    candidates = edits1("holmes")
    assert "banana" not in candidates  # sanity: not every string is generated
    assert "olmes" in candidates  # deletion
    assert "holmesa" in candidates  # insertion
    assert "holmey" in candidates  # substitution
    assert "holmse" in candidates  # transposition


def test_edits2_is_superset_reachable_in_two_steps():
    # "holm" is two deletions away from "holmes"
    assert "holm" in edits2("holmes")


def test_known_word_returns_itself_with_zero_distance(toy_model):
    suggestions = toy_model.suggest("holmes")
    assert suggestions[0].word == "holmes"
    assert suggestions[0].edit_distance == 0


def test_single_typo_is_corrected(toy_model):
    # "halmes" -> "holmes" via one substitution
    suggestions = toy_model.suggest("halmes")
    assert suggestions[0].word == "holmes"
    assert suggestions[0].edit_distance == 1


def test_ranking_prefers_higher_frequency_on_tied_distance(toy_model):
    # "boor" is exactly one substitution from both "moor" and "door", so the
    # tie is broken purely on corpus frequency: moor (25) outranks door (5).
    suggestions = toy_model.suggest("boor")
    assert {s.word for s in suggestions[:2]} == {"moor", "door"}
    assert suggestions[0].edit_distance == suggestions[1].edit_distance == 1
    assert suggestions[0].word == "moor"


def test_correct_returns_top_suggestion_word(toy_model):
    assert toy_model.correct("holmes") == "holmes"
    assert toy_model.correct("halmes") == "holmes"


def test_unrecognizable_word_falls_back_to_nearest_neighbor(toy_model):
    # far outside edit distance 2 of anything in the tiny vocabulary
    suggestions = toy_model.suggest("xxxxxxxxxxxx")
    assert suggestions  # falls back rather than raising or returning empty
