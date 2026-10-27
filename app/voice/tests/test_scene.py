from psirens_voice.scene.renderer import render_prompt
from psirens_voice.scene.state import SceneState
from psirens_voice.scene.timeline import Timeline


def test_render_prompt_skips_empty_and_orders():
    s = SceneState(subject="a lighthouse", setting="on a cliff",
                   lighting="warm dusk glow", medium="oil painting")
    assert render_prompt(s) == \
        "a lighthouse, on a cliff, warm dusk glow lighting, oil painting"
    assert render_prompt(SceneState()) == ""


def test_apply_patch_reports_changes():
    s = SceneState(subject="a lighthouse", lighting="cold daylight")
    new, changed = s.apply_patch({"lighting": "warm glow", "mood": "serene"})
    assert changed == ["lighting", "mood"]
    assert new.lighting == "warm glow" and new.subject == "a lighthouse"
    assert s.lighting == "cold daylight"  # original untouched


def test_timeline_revert_appends_new_entry():
    tl = Timeline()
    s1, _ = tl.current_state.apply_patch({"subject": "a fox"})
    tl.append(kind="edit", state_after=s1, utterance="a fox")
    s2, _ = s1.apply_patch({"subject": "a wolf"})
    tl.append(kind="edit", state_after=s2, utterance="a wolf")

    entry = tl.revert_to(1)
    assert entry is not None
    assert len(tl.entries) == 3           # history is append-only
    assert entry["kind"] == "revert" and entry["revert_target"] == 1
    assert tl.current_state.subject == "a fox"
    assert tl.revert_to(99) is None
