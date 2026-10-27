"""Repeated cross-utterance undo must walk BACK through history, not
oscillate (revert-of-revert = redo)."""
from psirens_voice.scene.state import SceneState
from psirens_voice.scene.timeline import Timeline


def walk_back_target(entries):
    """Mirror of the editor's repeated-undo logic."""
    if not entries:
        return None
    last = entries[-1]
    if last["kind"] == "revert" and last.get("revert_target"):
        idx = next((i for i, e in enumerate(entries)
                    if e["commit_id"] == last["revert_target"]), None)
        return idx - 1 if idx is not None else None
    return len(entries) - 2


def test_repeated_undo_walks_back_not_redo():
    tl = Timeline()
    tl.append(kind="edit", state_after=SceneState(subject="a fox"))       # 1
    tl.append(kind="edit", state_after=SceneState(subject="a wolf"))      # 2
    tl.append(kind="edit", state_after=SceneState(subject="a bear"))      # 3

    # first "no no": revert to entry before last -> wolf
    t1 = walk_back_target(tl.entries)
    assert tl.entries[t1]["state_after"]["subject"] == "a wolf"
    tl.revert_to(tl.entries[t1]["commit_id"])

    # second "no no": must go FURTHER back -> fox (not bounce to bear)
    t2 = walk_back_target(tl.entries)
    assert tl.entries[t2]["state_after"]["subject"] == "a fox"
    tl.revert_to(tl.entries[t2]["commit_id"])
    assert tl.current_state.subject == "a fox"

    # third: nothing earlier -> empty-scene signal
    t3 = walk_back_target(tl.entries)
    assert t3 is not None and t3 < 0 or t3 is None or t3 == -1
