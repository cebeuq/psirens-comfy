"""Rule-based self-repair cleanup, run on a committed utterance before the LLM.

"make it blue — no, green"  ->  "make it green"

Algorithm (deliberately dumb for v1, every repair is logged so failures are
visible in the event log):
  - scan words left-to-right for repair markers (normalized, punctuation
    stripped; bigrams matched before unigrams)
  - on a marker at position i with n words following it, drop the marker and
    min(n, i) words before it — the "parallel structure" heuristic:
      "make it blue no green"        -> drop 1 before  -> "make it green"
      "blue no make it green"        -> drop 1 (min(3,1)) -> "make it green"
  - "no" is only a marker when it is NOT utterance-initial: initial "no ..."
    is often semantic negation ("no people in the scene")
  - repeat until no marker matches
"""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w']+")

BIGRAM_MARKERS = [("no", "wait"), ("scratch", "that"), ("i", "mean")]
UNIGRAM_MARKERS = {"no": {"initial_ok": False},
                   "nope": {"initial_ok": False},
                   "wait": {"initial_ok": True},
                   "actually": {"initial_ok": True}}
# words that grammatically LICENSE a following "no" as plain negation
# ("with no furniture", "there are no windows") — never a correction.
# Live failure: 'a modern home with no furniture' -> 'home furniture'.
_NEGATION_LICENSE = {
    "with", "has", "have", "had", "is", "are", "was", "were", "be",
    "there's", "theres", "want", "wants", "wanted", "need", "needs",
    "needed", "contains", "contain", "containing", "include",
    "includes", "including", "but", "and", "or", "of", "almost",
    "absolutely", "still", "show", "shows", "see", "add", "put",
    "use", "draw", "make", "having",
}
_PUNCT_TAIL = re.compile(r"[,.;:!?\u2014\u2013-]$")
MAX_PASSES = 4


def _norm(w: str) -> str:
    return _PUNCT.sub("", w.lower())


def apply_self_repair(words: list[str]) -> tuple[str, list[dict]]:
    """words: raw word strings (may carry punctuation).
    Returns (clean_text, repairs)."""
    repairs: list[dict] = []
    ws = []
    for w in words:
        # a standalone punctuation token ("—") glues onto its
        # predecessor so boundary detection and drop counts see real
        # words, not orphan dashes
        if _norm(w) == "" and ws:
            ws[-1] = ws[-1] + w
        else:
            ws.append(w)

    for _ in range(MAX_PASSES):
        norm = [_norm(w) for w in ws]
        match = None  # (start_idx, marker_len, marker_text)
        for i in range(len(ws)):
            for big in BIGRAM_MARKERS:
                if i + 1 < len(ws) and (norm[i], norm[i + 1]) == big:
                    match = (i, 2, f"{big[0]} {big[1]}")
                    break
            if match:
                break
            rules = UNIGRAM_MARKERS.get(norm[i])
            if rules and (i > 0 or rules["initial_ok"]):
                if norm[i] in ("no", "nope"):
                    # high-precision only: a spoken correction "blue —
                    # no, green" carries a punctuation/pause boundary;
                    # a licensed "with no furniture" carries neither.
                    # Missed corrections are the parser LLM's job.
                    if i > 0 and norm[i - 1] in _NEGATION_LICENSE:
                        continue
                    # a true spoken correction is bracketed by pauses
                    # on BOTH sides ("blue, no, green"); a list negation
                    # only has one ("no windows, no doors")
                    before = (bool(_PUNCT_TAIL.search(ws[i - 1]))
                              if i > 0 else False)
                    after = bool(_PUNCT_TAIL.search(ws[i]))
                    if not (before and after):
                        continue
                match = (i, 1, norm[i])
                break
        if not match:
            break

        i, mlen, marker = match
        n_after = len(ws) - (i + mlen)
        n_drop_before = min(n_after, i) if n_after > 0 else 0
        dropped = ws[i - n_drop_before:i]
        kept_after = ws[i + mlen:]
        repairs.append({"marker": marker,
                        "dropped": dropped,
                        "kept": kept_after})
        ws = ws[:i - n_drop_before] + kept_after

    clean = " ".join(ws)
    clean = re.sub(r"\s+", " ", clean).strip(" ,;")
    return clean, repairs
