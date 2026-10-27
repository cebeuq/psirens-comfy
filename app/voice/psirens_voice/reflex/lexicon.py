"""Deterministic color/intensity lexicon for the reflex loop.

Color words are literal — "purple" means purple with zero ambiguity — so they
never go through the classifier (which, being enum-constrained, would be
FORCED to pick a wrong neighbor for colors outside its menu). 0ms, always
right.
"""
from __future__ import annotations

COLOR_WORDS = {
    "red", "green", "blue", "purple", "violet", "yellow", "orange", "pink",
    "teal", "cyan", "magenta", "brown", "gold", "crimson", "emerald",
    "indigo", "turquoise", "lavender",
    # black/white/gray are PALETTE colors (a new one replaces the old —
    # "black then white" must not leave a black palette behind)
    "black", "white", "gray", "grey",
}
# comparatives / variants -> base color
COMPARATIVES = {
    "redder": "red", "greener": "green", "bluer": "blue",
    "purpler": "purple", "yellower": "yellow", "pinker": "pink",
    "oranger": "orange", "browner": "brown", "blacker": "black",
    "whiter": "white",
}
# tonal words that map to existing non-color axes (lighting)
TONAL = {"dark": "darker", "darker": "darker", "dim": "darker",
         "bright": "brighter", "brighter": "brighter", "light": "brighter"}

INTENSIFIERS = {"more", "super", "really", "way", "very", "lot", "lots",
                "much", "extremely", "totally"}

# words that mark a style/color mention as a COMMAND rather than a
# description: "make it red" / "turn it red" / "more red" / "redder".
# Bare attributive mentions ("a red Lamborghini") describe the scene and
# must NOT escalate style — they belong to the editor.
IMPERATIVE_MARKERS = {"make", "makes", "making", "turn", "turns", "paint",
                     "set", "go", "goes", "get", "bit", "little", "less"} | INTENSIFIERS


# ---- generic support veto -------------------------------------------------
# The categorical failure of enum-constrained classification: input outside
# the menu gets coerced into the nearest WRONG slot instead of "none".
# Guard: an intent is only trusted when the spoken words lexically support
# it; otherwise the reflex does NOTHING and the open-vocabulary layers
# (draft/commit LLMs) handle the utterance. High precision beats recall in a
# layer that acts in milliseconds.

MOTION_WORDS = {"move", "nudge", "slide", "push", "scoot", "shift", "put",
                "drag", "bring", "closer", "further", "away", "over",
                "left", "right", "up", "down", "higher", "lower", "more"}
SIZE_WORDS = {"bigger", "smaller", "larger", "tinier", "huge", "tiny",
              "big", "small", "shrink", "grow", "size", "wider", "taller",
              "enlarge", "expand"}
STYLE_WORDS = {"warmer", "warm", "cooler", "cool", "brighter", "bright",
               "darker", "dark", "moodier", "moody", "lighter", "colorful"}
UNDO_WORDS = {"no", "nope", "undo", "back", "revert", "wait"}
STOP_WORDS_OK = {"stop", "there", "perfect", "good", "hold", "thats", "that's"}
DIRECTION_SUPPORT = {
    "left": {"left"}, "right": {"right"},
    "up": {"up", "higher", "top", "above", "ceiling"},
    "down": {"down", "lower", "bottom", "below", "floor"},
    "closer": {"closer", "close", "nearer", "forward"},
    "away": {"away", "further", "farther", "back"},
}


def intent_supported(intent: dict, norm_words: list[str],
                     entity_ids: list[str]) -> dict | None:
    """Returns a (possibly downgraded) intent the words actually support,
    or None if the classification is unsupported and must defer."""
    words = set(norm_words)
    action = intent.get("action")

    if action in ("none", "other"):
        return None
    if action == "undo" and not (words & UNDO_WORDS):
        return None
    if action == "stop" and not (words & STOP_WORDS_OK):
        return None
    if action == "move" and not (words & (MOTION_WORDS | SIZE_WORDS)):
        return None
    if action == "scale" and not (words & (SIZE_WORDS | {"closer", "away"})):
        return None
    if action == "style":
        supported = (words & STYLE_WORDS
                     or words & COLOR_WORDS
                     or any(w in COMPARATIVES or w in TONAL for w in words))
        if not supported:
            return None

    out = dict(intent)
    # claimed direction must be spoken, else it's a continuation ("none")
    d = out.get("direction", "none")
    if d != "none" and not (words & DIRECTION_SUPPORT.get(d, set())):
        out["direction"] = "none"
    # claimed target entity must be mentioned, else fall back to selection
    t = out.get("target", "none")
    if t not in ("selected", "none"):
        target_tokens = set(t.replace("_", " ").split())
        if not (words & target_tokens):
            out["target"] = "selected"
    return out


def detect_style_intent(norm_words: list[str],
                        stopwords: set[str] | None = None) -> dict | None:
    """Deterministic style intent from normalized words, or None.
    The LAST commanding color/tonal word wins (people correct themselves
    left-to-right).

    Describing is not commanding: "a red Lamborghini" names an attribute of
    a thing and must defer to the editor. A color/tonal word only counts
    when it is used imperatively:
      - a comparative form ("redder", "darker") is always a command
      - a base word counts when an imperative marker is present
        ("make it red", "more red") AND the word is not attributive
        (attributive = immediately followed by a content word: "red CAR")
      - a fragment that is ONLY color words ("red." / "purple") is a command
    """
    stopwords = stopwords or set()

    def attributive(i: int) -> bool:
        if i + 1 >= len(norm_words):
            return False
        nxt = norm_words[i + 1]
        return (nxt not in stopwords and nxt not in COLOR_WORDS
                and nxt not in TONAL and nxt not in COMPARATIVES
                and nxt not in INTENSIFIERS and nxt not in IMPERATIVE_MARKERS
                and nxt not in ("it", "them", "everything", "again", "please"))

    has_marker = any(w in IMPERATIVE_MARKERS for w in norm_words)
    only_colors = all(w in COLOR_WORDS or w in TONAL or w in COMPARATIVES
                      or w in INTENSIFIERS for w in norm_words) and norm_words

    axis = None
    for i, w in enumerate(norm_words):
        if w in COMPARATIVES:                       # "redder" — always a command
            axis = f"color:{COMPARATIVES[w]}"
        elif w in ("darker", "brighter"):           # comparative tonal form
            axis = TONAL[w]                         # — always a command
        elif w in COLOR_WORDS and (has_marker or only_colors) and not attributive(i):
            axis = f"color:{w}"
        elif w in TONAL and (has_marker or only_colors) and not attributive(i):
            axis = TONAL[w]
    if axis is None:
        return None
    n_intense = sum(1 for w in norm_words if w in INTENSIFIERS)
    amount = "big" if n_intense >= 2 else "medium"
    return {"action": "style", "target": "none", "direction": "none",
            "amount": amount, "sign": "plus", "style_axis": axis}


# ---- deterministic motion/scale/stop grammar ------------------------------
# Replaces the LLM micro-intent classifier: v3 reflex is pure rules, 0ms,
# zero calls. High precision, low recall — anything not clearly supported
# defers to the editor.

_AMOUNT_TINY = {"tiny", "touch", "hair", "slightly", "tad"}
_AMOUNT_SMALL = {"little", "bit", "small", "nudge"}
_AMOUNT_BIG = {"way", "lot", "lots", "much", "far", "big", "huge"}
_SCALE_MINUS = {"smaller", "tinier", "tiny", "shrink", "small"}
_SCALE_PLUS = {"bigger", "larger", "huge", "grow", "big", "enlarge",
               "expand", "wider", "taller"}
_DIR_BY_WORD = {w: d for d, ws in DIRECTION_SUPPORT.items() for w in ws
                if w != "back"}   # "back" is undo vocabulary, never a direction


def detect_control_intent(norm_words: list[str],
                          entity_ids: list[str]) -> dict | None:
    """Deterministic move/scale/stop intent from normalized words, or None."""
    words = set(norm_words)

    if words & STOP_WORDS_OK and not (words & (MOTION_WORDS | SIZE_WORDS)):
        return {"action": "stop", "target": "none", "direction": "none",
                "amount": "medium", "sign": "plus", "style_axis": "none"}

    direction = "none"
    for w in norm_words:                 # last direction wins
        if w in _DIR_BY_WORD:
            direction = _DIR_BY_WORD[w]

    target = "selected"
    for eid in entity_ids:               # spoken entity name wins
        if set(eid.replace("_", " ").split()) & words:
            target = eid

    amount = "medium"
    if words & _AMOUNT_TINY:
        amount = "tiny"
    elif words & _AMOUNT_SMALL:
        amount = "small"
    elif words & _AMOUNT_BIG:
        amount = "big"

    scale_hit = words & (_SCALE_PLUS | _SCALE_MINUS)
    if scale_hit:
        sign = "minus" if words & _SCALE_MINUS else "plus"
        return {"action": "scale", "target": target, "direction": "none",
                "amount": amount, "sign": sign, "style_axis": "none"}
    if direction != "none" or (words & MOTION_WORDS and target != "selected"):
        return {"action": "move", "target": target, "direction": direction,
                "amount": amount, "sign": "plus", "style_axis": "none"}
    # continuation: "a little more", "again", "keep going" — repeats the
    # LAST manipulation (direction "none" = momentum; the integrator applies
    # its remembered vector, and safely does nothing if there is none)
    cont = {"more", "again", "further", "farther", "keep", "going", "continue"}
    if (words & cont
            and all(w in cont | _AMOUNT_TINY | _AMOUNT_SMALL | _AMOUNT_BIG
                    | INTENSIFIERS for w in norm_words)):
        return {"action": "move", "target": "selected", "direction": "none",
                "amount": amount, "sign": "plus", "style_axis": "none"}
    return None
