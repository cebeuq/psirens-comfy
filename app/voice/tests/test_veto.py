"""The anti-coercion invariant: the reflex acts only on classifications the
spoken words support; everything else defers to the LLM layers."""
from psirens_voice.reflex.lexicon import intent_supported


def intent(**kw):
    base = {"action": "move", "target": "selected", "direction": "none",
            "amount": "medium", "sign": "plus", "style_axis": "none"}
    base.update(kw)
    return base


def test_unknown_action_words_are_vetoed():
    # "rotate it clockwise" misclassified as move -> vetoed (no motion word)
    assert intent_supported(intent(action="move"),
                            ["rotate", "clockwise"], ["chair"]) is None
    # "flip it" misclassified as scale -> vetoed
    assert intent_supported(intent(action="scale"),
                            ["flip"], ["chair"]) is None
    # "make it grainier" misclassified as style greener -> vetoed
    assert intent_supported(intent(action="style", style_axis="greener"),
                            ["grainier"], []) is None


def test_other_and_none_always_defer():
    assert intent_supported(intent(action="other"), ["rotate"], []) is None
    assert intent_supported(intent(action="none"), ["a", "story"], []) is None


def test_supported_intents_pass_through():
    out = intent_supported(intent(action="move", direction="left"),
                           ["nudge", "left"], ["chair"])
    assert out is not None and out["direction"] == "left"
    out = intent_supported(intent(action="style", style_axis="warmer"),
                           ["warmer"], [])
    assert out is not None


def test_hallucinated_direction_downgraded_to_continuation():
    out = intent_supported(intent(action="move", direction="right"),
                           ["a", "little", "more"], ["chair"])
    assert out is not None and out["direction"] == "none"


def test_unmentioned_target_downgraded_to_selected():
    # "slide it over" classified with target=chair, but chair wasn't said
    out = intent_supported(intent(action="move", target="chair",
                                  direction="left"),
                           ["slide", "left"], ["chair", "lamp"])
    assert out is not None and out["target"] == "selected"
    # explicit mention keeps the target
    out = intent_supported(intent(action="move", target="floor_lamp",
                                  direction="left"),
                           ["lamp", "left"], ["floor_lamp"])
    assert out is not None and out["target"] == "floor_lamp"