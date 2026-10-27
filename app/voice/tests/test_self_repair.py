import pytest

from psirens_voice.commit.self_repair import apply_self_repair

FIXTURES = [
    # (input words, expected clean text, expected n repairs)
    (["make", "it", "blue", "no,", "green"], "make it green", 1),
    (["make", "it", "blue,", "no", "wait,", "green"], "make it green", 1),
    (["make", "it", "blue", "actually", "green"], "make it green", 1),
    (["blue", "no", "make", "it", "green"], "make it green", 1),
    (["make", "the", "sky", "darker"], "make the sky darker", 0),
    # utterance-initial "no" is semantic negation, must be preserved
    (["no", "people", "in", "the", "scene"], "no people in the scene", 0),
    (["no,", "people", "should", "stay"], "no, people should stay", 0),
    # utterance-initial "wait"/"actually" are just discourse markers
    (["wait,", "make", "the", "sky", "darker"], "make the sky darker", 1),
    (["actually,", "make", "it", "dusk"], "make it dusk", 1),
    (["i", "mean", "a", "castle"], "a castle", 1),
    (["a", "cottage,", "i", "mean", "a", "castle"], "a castle", 1),
    (["make", "it", "red", "scratch", "that", "make", "it", "gold"], "make it gold", 1),
    # trailing marker: drop marker only
    (["make", "it", "green,", "wait"], "make it green", 1),
    # multiple repairs
    (["blue", "no", "red", "no", "green"], "green", 2),
    (["a", "dog", "actually", "a", "wolf", "no,", "a", "fox"], "a fox", 2),
    # marker words inside content are left alone when not matched
    (["there", "is", "nothing", "waiting"], "there is nothing waiting", 0),
    # "make it warmer, no cooler"
    (["make", "it", "warmer,", "no", "cooler"], "make it cooler", 1),
    # long correction span
    (["a", "house", "on", "a", "hill", "no", "a", "boat", "on", "a", "lake"],
     "a boat on a lake", 1),
    (["pull", "the", "camera", "back"], "pull the camera back", 0),
    (["umm", "make", "it", "nope", "keep", "it"], "umm keep it", 1),
]


@pytest.mark.parametrize("words,expected,n_repairs", FIXTURES)
def test_self_repair(words, expected, n_repairs):
    clean, repairs = apply_self_repair(words)
    assert clean == expected
    assert len(repairs) == n_repairs
