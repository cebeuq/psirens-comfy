"""The timeline ledger — the model's readable memory of the picture's past.

One compact line per commit, rebuilt deterministically each call and
injected into the editor prompt, so the existing commit call can choose
restore / restore+replay / forward-edit with zero extra calls. All four
v6 designs converged on exactly this.

Also: the SEXTANT inverse-scan — a deterministic, zero-call detector for
"this directive undoes a previous commit" ("remove the stickers" vs
"add racing stickers"), which becomes a hint line the model sees.
"""
from __future__ import annotations

LEDGER_MAX_LINES = 24

_STOP = {"a", "an", "the", "and", "to", "of", "it", "its", "in", "on",
         "with", "is", "that", "this", "some", "any", "please", "make",
         "put", "them", "these", "those"}
_REMOVAL = {"remove", "delete", "undo", "erase", "without", "rid", "off",
            "away", "gone", "back", "revert", "scrap", "ditch", "lose"}
_ADDITION = {"add", "adds", "added", "put", "give", "gave", "place",
             "placed", "with"}


def _content(text: str) -> set[str]:
    return {w.strip(".,!?\"'").lower() for w in text.split()
            if w.strip(".,!?\"'").lower() not in _STOP
            and len(w.strip(".,!?\"'")) > 2}


def _changed_summary(entry: dict) -> str:
    patch = entry.get("patch") or {}
    parts = [k for k in patch if k not in ("entity_edits", "reflex",
                                           "history_action")]
    for op in patch.get("entity_edits") or []:
        parts.append(f"{op.get('op')}:{op.get('id')}")
    if patch.get("reflex"):
        parts.append("nudge")
    return ",".join(parts) if parts else "-"


def build_ledger(entries: list[dict]) -> str | None:
    """Compact one-line-per-commit history block, newest last."""
    if not entries:
        return None
    lines = []
    for e in entries[-LEDGER_MAX_LINES:]:
        cid = e["commit_id"]
        flags = ""
        if e.get("undone_by"):
            flags = f" [undone by #{e['undone_by']}]"
        elif e.get("replayed_into"):
            flags = f" [replayed into #{e['replayed_into']}]"
        if e["kind"] == "revert":
            src = e.get("revert_target")
            lines.append(f'#{cid} restored frame #{src}'
                         f' "{e.get("utterance", "")[:60]}"{flags}')
        else:
            lines.append(f'#{cid} "{e.get("utterance", "")[:60]}" | '
                         f'{_changed_summary(e)}{flags}')
    return "\n".join(lines)


def inverse_scan(directive: str, entries: list[dict]):
    """Structured version of inverse_hint: returns (hint_line, undone_rev,
    before_rev) or (None, None, None)."""
    h = inverse_hint(directive, entries)
    if h is None:
        return None, None, None
    import re
    m = re.search(r"undo #(\d+).*frame #(\d+)", h)
    if not m:
        return h, None, None
    return h, int(m.group(1)), int(m.group(2))


def inverse_hint(directive: str, entries: list[dict]) -> str | None:
    """Zero-call detector: does this directive likely UNDO a prior commit?
    Word-level: removal vocabulary + content overlap with a prior
    addition-flavored commit. Returns a hint line for the prompt, or None.
    The hint is advisory — the model still decides."""
    words = _content(directive)
    if not ({w.strip(".,!?").lower() for w in directive.split()} & _REMOVAL):
        return None
    live = [e for e in entries if e["kind"] == "edit"
            and not e.get("undone_by")]
    for idx in range(len(live) - 1, -1, -1):
        e = live[idx]
        utt = e.get("utterance", "")
        add_ops = [op for op in
                   (e.get("patch") or {}).get("entity_edits") or []
                   if op.get("op") == "add"]
        added = (bool({w.lower() for w in utt.split()} & _ADDITION)
                 or bool(add_ops))
        if not added:
            continue
        # when the commit tracked WHAT it added, the removal must name
        # that thing — "remove the stuff on the bed" must not match
        # "put a MacBook on the bed" via the incidental word "bed"
        # (that exact false match once rewound 13 edits)
        if add_ops:
            target_words: set[str] = set()
            for op in add_ops:
                for field in ("name", "id"):
                    v = (op.get(field) or "").replace("_", " ").lower()
                    target_words |= {w for w in v.split() if len(w) > 2}
            hit = bool(words & target_words)
        else:
            hit = bool(words & _content(utt))
        if hit:
            # the frame BEFORE this commit is the pre-damage state
            pos = next((i for i, x in enumerate(entries)
                        if x["commit_id"] == e["commit_id"]), None)
            if pos is None or pos == 0:
                return None
            before = entries[pos - 1]["commit_id"]
            return (f'hint: this may undo #{e["commit_id"]} '
                    f'("{utt[:60]}") — frame #{before} is the state just '
                    f'before it.')
    return None
