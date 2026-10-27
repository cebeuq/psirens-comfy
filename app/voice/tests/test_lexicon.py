from psirens_voice.reflex.integrator import ReflexIntegrator
from psirens_voice.reflex.lexicon import (detect_control_intent,
                                          detect_style_intent)
from psirens_voice.scene.state import SceneState

STOP = {"a", "an", "the", "and", "to", "of", "it", "that", "this",
        "in", "on", "with", "is", "its"}


def test_color_detection():
    # bare colors alone are commands ("red." as a whole fragment)
    assert detect_style_intent(["purple"])["style_axis"] == "color:purple"
    # comparatives are always commands
    assert detect_style_intent(["redder", "redder"])["style_axis"] == "color:red"
    assert detect_style_intent(["super", "super", "yellow"])["amount"] == "big"
    # imperative markers make base colors commands; last one wins
    assert detect_style_intent(["make", "red", "green"])["style_axis"] == "color:green"
    assert detect_style_intent(["super", "white"])["style_axis"] == "color:white"
    assert detect_style_intent(["much", "darker"])["style_axis"] == "darker"
    assert detect_style_intent(["move", "left"]) is None


def test_describing_is_not_commanding():
    # "a red Lamborghini" DESCRIBES — must not escalate style (bug C)
    assert detect_style_intent(["red", "lamborghini"], STOP) is None
    # mixed narration without imperative markers defers
    assert detect_style_intent(["red", "shifting", "green"], STOP) is None
    # ...but "make it red" and "more red" command
    assert detect_style_intent(["make", "red"], STOP)["style_axis"] == "color:red"
    assert detect_style_intent(["more", "red"], STOP)["style_axis"] == "color:red"
    # comparative fires even next to a noun ("redder car")
    assert detect_style_intent(["redder", "car"], STOP)["style_axis"] == "color:red"


def test_control_grammar():
    # deterministic move/scale/stop — the LLM classifier is gone
    i = detect_control_intent(["little", "left"], [])
    assert i["action"] == "move" and i["direction"] == "left" and i["amount"] == "small"
    i = detect_control_intent(["move", "chair", "up"], ["chair"])
    assert i["target"] == "chair" and i["direction"] == "up"
    i = detect_control_intent(["way", "bigger"], [])
    assert i["action"] == "scale" and i["sign"] == "plus" and i["amount"] == "big"
    i = detect_control_intent(["bit", "smaller"], [])
    assert i["action"] == "scale" and i["sign"] == "minus"
    assert detect_control_intent(["perfect", "stop", "there"], [])["action"] == "stop"
    # no support -> defer to the editor
    assert detect_control_intent(["beverly", "hills"], []) is None
    # "back" alone is undo vocabulary, never a direction
    assert detect_control_intent(["back"], []) is None
    # continuation repeats the last manipulation (momentum, direction none)
    i = detect_control_intent(["little", "more"], [])
    assert i["action"] == "move" and i["direction"] == "none"
    assert detect_control_intent(["again"], [])["action"] == "move"
    # ...but continuation words mixed with real content defer
    assert detect_control_intent(["more", "chairs"], []) is None


def test_style_levels_persist_across_utterances():
    from psirens_voice.reflex.integrator import ReflexIntegrator
    from psirens_voice.scene.state import SceneState
    it = ReflexIntegrator(lambda: SceneState())
    s = {"action": "style", "target": "none", "direction": "none",
         "amount": "medium", "sign": "plus", "style_axis": "darker"}
    it.apply(dict(s))
    it.apply(dict(s))
    assert it.fields()["lighting"] == "dark shadowy light"
    it.reset_utterance()  # commit boundary
    it.apply(dict(s))     # "darker" in the NEXT utterance
    # continues from level 2 -> 3, no regression to level 1
    assert it.fields()["lighting"] == "near darkness"


def test_color_switch_replaces_accumulation():
    it = ReflexIntegrator(lambda: SceneState())

    def style(axis):
        return {"action": "style", "target": "none", "direction": "none",
                "amount": "medium", "sign": "plus", "style_axis": axis}

    it.apply(style("color:red"))
    it.apply(style("color:red"))
    assert it.fields()["palette"] == "rich red tones"
    it.apply(style("color:green"))
    # switching color drops the old one instead of blending forever
    assert it.fields()["palette"] == "a hint of green"
    it.apply(style("darker"))
    assert "lighting" in it.fields()
