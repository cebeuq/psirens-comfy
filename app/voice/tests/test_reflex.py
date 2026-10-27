from psirens_voice.reflex.integrator import ReflexIntegrator
from psirens_voice.scene.state import Box, Entity, SceneState


def make_integrator():
    state = SceneState(entities=[
        Entity(id="chair", name="chair", description="a chair",
               box=Box(x=0.5, y=0.6, w=0.3, h=0.4)),
        Entity(id="lamp", name="lamp", description="a lamp",
               box=Box(x=0.8, y=0.5, w=0.1, h=0.3)),
    ])
    return ReflexIntegrator(lambda: state)


def intent(**kw):
    base = {"action": "none", "target": "none", "direction": "none",
            "amount": "medium", "sign": "plus", "style_axis": "none"}
    base.update(kw)
    return base


def test_move_and_continuation():
    it = make_integrator()
    r = it.apply(intent(action="move", target="chair", direction="left"))
    assert r.kind == "move"
    assert abs(it.work.boxes["chair"]["x"] - 0.38) < 1e-9  # 0.5 - 0.12
    # "a little more": no direction, repeats last vector scaled to small
    r = it.apply(intent(action="move", target="selected",
                        direction="none", amount="small"))
    assert r is not None
    assert it.work.boxes["chair"]["x"] < 0.38


def test_undo_restores():
    it = make_integrator()
    it.apply(intent(action="move", target="chair", direction="left"))
    x_after_first = it.work.boxes["chair"]["x"]
    it.apply(intent(action="move", target="selected", direction="left"))
    r = it.apply(intent(action="undo"))
    assert r.kind == "undo"
    assert abs(it.work.boxes["chair"]["x"] - x_after_first) < 1e-9


def test_scale_signs_and_closer():
    it = make_integrator()
    it.apply(intent(action="scale", target="lamp", sign="plus", amount="medium"))
    assert it.work.boxes["lamp"]["w"] > 0.1
    it.apply(intent(action="scale", target="selected", sign="minus", amount="medium"))
    assert abs(it.work.boxes["lamp"]["w"] - 0.1) < 1e-6
    # closer = grow
    it.apply(intent(action="move", target="lamp", direction="closer"))
    assert it.work.boxes["lamp"]["w"] > 0.1


def test_style_accumulation_and_opposites():
    it = make_integrator()
    for _ in range(3):
        it.apply(intent(action="style", style_axis="greener"))
    assert it.fields()["palette"] == "deep saturated green"
    it.apply(intent(action="style", style_axis="warmer"))
    it.apply(intent(action="style", style_axis="cooler"))
    # cooler cancels warmer level instead of stacking both
    assert "lighting" not in it.fields()


def test_none_marks_residue_and_move_without_target():
    it = make_integrator()
    assert it.apply(intent(action="none")) is None
    assert it.residue
    it2 = make_integrator()
    # "move it left" with nothing selected: can't act, marks residue
    r = it2.apply(intent(action="move", target="selected", direction="left"))
    assert r is None and it2.residue


def test_bare_move_selects_then_direction_completes():
    it = make_integrator()
    # "move the chair" (no direction yet) selects, applies nothing
    assert it.apply(intent(action="move", target="chair", direction="none")) is None
    assert it.selected == "chair" and it.actions_applied == 0
    # "left" fragment
    r = it.apply(intent(action="move", target="selected", direction="left"))
    assert r is not None and it.actions_applied == 1


def test_entity_ops_export():
    it = make_integrator()
    it.apply(intent(action="move", target="chair", direction="right", amount="big"))
    ops = it.entity_ops()
    assert len(ops) == 1 and ops[0]["op"] == "update" and ops[0]["id"] == "chair"
    assert abs(ops[0]["box"]["x"] - 0.72) < 1e-9
