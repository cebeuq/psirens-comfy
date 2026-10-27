from psirens_voice.scene.live import LiveScene
from psirens_voice.scene.renderer import render_prompt
from psirens_voice.scene.state import Box, Entity, SceneState


def _chair():
    return Entity(id="chair", name="chair", description="a wooden chair",
                  box=Box(x=0.2, y=0.6, w=0.3, h=0.4))


def test_entity_add_update_remove():
    s = SceneState(subject="a room")
    s, ch = s.apply_entity_edits([{"op": "add", "id": "chair", "name": "chair",
                                   "description": "a wooden chair",
                                   "box": {"x": 0.2, "y": 0.6, "w": 0.3, "h": 0.4}}])
    assert ch == ["entity:chair"] and len(s.entities) == 1

    s, ch = s.apply_entity_edits([{"op": "update", "id": "chair",
                                   "box": {"x": 0.1, "y": 0.6, "w": 0.3, "h": 0.4}}])
    assert s.entities[0].box.x == 0.1

    # unknown-id update/remove is a safe no-op
    s, ch = s.apply_entity_edits([{"op": "update", "id": "ghost",
                                   "box": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}}])
    assert ch == [] and len(s.entities) == 1

    s, ch = s.apply_entity_edits([{"op": "remove", "id": "chair"}])
    assert s.entities == [] and ch == ["entity:chair"]


def test_position_phrase_and_prompt():
    e = _chair()
    assert e.position_phrase() == "on the left"
    e.box.y = 0.8
    assert e.position_phrase() == "at the bottom left"
    s = SceneState(subject="a cozy room", entities=[_chair()])
    assert render_prompt(s) == "a cozy room, a wooden chair on the left"


def test_live_overlay_entity_ops():
    live = LiveScene()
    live.timeline.append(kind="edit", state_after=SceneState(subject="a room"))
    changed = live.set_overlay({}, [{"op": "add", "id": "cat", "name": "cat",
                                     "description": "a cat",
                                     "box": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}}])
    assert changed == ["entity:cat"]
    assert live.effective_state().entities[0].id == "cat"
    assert live.canonical_state.entities == []
    live.canonical_changed()
    assert live.effective_state().entities == []
