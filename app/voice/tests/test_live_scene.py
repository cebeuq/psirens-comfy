from psirens_voice.scene.live import LiveScene
from psirens_voice.scene.state import SceneState


def test_overlay_and_effective_state():
    live = LiveScene()
    live.timeline.append(kind="edit",
                         state_after=SceneState(subject="a fox", lighting="dawn"))
    fields = live.set_overlay({"lighting": "dusk", "mood": "calm",
                               "subject": "a fox"})  # subject = echo, dropped
    assert sorted(fields) == ["lighting", "mood"]
    eff = live.effective_state()
    assert eff.lighting == "dusk" and eff.mood == "calm" and eff.subject == "a fox"
    # canonical untouched
    assert live.canonical_state.lighting == "dawn"


def test_canonical_changed_clears_overlay_and_bumps_generation():
    live = LiveScene()
    live.set_overlay({"mood": "tense"})
    g = live.generation
    live.canonical_changed()
    assert live.overlay == {}
    assert live.generation == g + 1
    assert live.effective_state() == live.canonical_state


def test_new_draft_replaces_whole_overlay():
    live = LiveScene()
    live.set_overlay({"mood": "tense", "palette": "red"})
    live.set_overlay({"lighting": "noon"})
    assert list(live.overlay) == ["lighting"]
