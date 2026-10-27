"""Edit-compliance judge: the editor vLLM scores a draft against the
instruction + a deterministic requirements checklist.

Contract (measured on the design bench: 292/296 correct at 256px;
accept ~150-250ms, reject ~300-450ms): before+after images, numbered
requirements as INPUT, boolean `met` array keyed by requirement index
(no string echo — keeps the accept path ~35 output tokens), and one
corrective action from a closed vocabulary. `confidence` deliberately
absent (measured uninformative); code reconciles verdict vs met — a
"pass" with any met:false is a fail, never trust the prose.

Stdlib HTTP on purpose: the renderer image has no httpx and a rebuild
of the torch stack is not worth a dependency. Called from the blocking
render worker, so sync is correct here.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

log = logging.getLogger("renderer")

# Anti-degeneracy patterns (verified enforced by vLLM 0.27 xgrammar):
# without them the model compresses "requirement 1 failed" to a bare
# "1" / "],  " — the minimal string the grammar accepts — and the
# downstream retry router gets junk. Forcing the FIRST character to a
# letter steers decoding off the lazy path entirely (measured: the
# model then writes real prose; a bare minLength instead makes it pad
# garbage after the "1"). what_failed additionally bans digits so
# "Requirement 1" is unrepresentable; "nothing failed" is the literal
# cheap escape for pass verdicts.
_P_ANALYSIS = "^[A-Za-z][A-Za-z0-9 ,.;:'()/-]{7,238}$"
_P_WHAT_FAILED = "^(nothing failed|[A-Za-z][A-Za-z ,.;:'()-]{11,150})$"

_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string", "maxLength": 240,
                     "pattern": _P_ANALYSIS},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "met": {"type": "array", "items": {"type": "boolean"},
                "maxItems": 5},
        "what_failed": {"type": "string", "maxLength": 160,
                        "pattern": _P_WHAT_FAILED},
        "corrective_action": {
            "type": "string",
            "enum": ["none", "reroll_seed", "strengthen_edit",
                     "rephrase_instruction", "escalate_full_res"]},
        "revised_instruction": {"type": "string", "maxLength": 300},
        # grounded profile: what KIND of failure, and WHERE (percent box,
        # null on pass) — the retry aims at fail_box instead of rerolling
        "failure_category": {
            "type": "string",
            "enum": ["none", "target_missing", "wrong_region", "too_weak",
                     "wrong_attribute", "collateral", "not_judgeable"]},
        "fail_box": {
            "anyOf": [{"type": "array", "minItems": 4, "maxItems": 4,
                       "items": {"type": "integer", "minimum": 0,
                                 "maximum": 100}},
                      {"type": "null"}]},
    },
    "required": ["analysis", "verdict", "met", "what_failed",
                 "corrective_action", "revised_instruction",
                 "failure_category", "fail_box"],
    "additionalProperties": False,
}

_RUBRIC = (
    "You judge whether an image EDIT followed its instruction. The first "
    "image is BEFORE, the second is AFTER (a low-res draft). Be HARSH: "
    "an object that is warped, mangled, duplicated, floating in mid-"
    "air, or resting on top of other furniture FAILS the edit even if "
    "no requirement names it — physical plausibility is always an "
    "implicit requirement, and borderline cases FAIL. Also fail if "
    "anything from the BEFORE image was visibly changed, distorted, or "
    "recolored that no requirement asked to change. Judge "
    "whether each numbered requirement is met: layout, presence, count, "
    "gross color/setting. Requirements about an object's DESCRIPTION "
    "(color, texture, patterns) are judgeable when the attribute is "
    "large. NOT judgeable at this resolution: sharpness, fine texture, "
    "and the exact TEXT or brand of any logo/writing — a logo "
    "requirement is met if a prominent logo/emblem is present where "
    "asked. When the edit FAILED, also report failure_category and the "
    "fail_box [x0,y0,x1,y1] in percent marking WHERE it went wrong (the "
    "region that should have changed, or the region wrongly changed); "
    "fail_box is null on pass. "
    "A DESCRIPTION requirement is met when each major attribute "
    "is visibly attempted — never fail it over style interpretation, "
    "object-subtype quibbles (couch vs daybed), or logo text. First "
    "write analysis: one short clause per requirement stating what you "
    "see — keep analysis under 35 words total so it is never cut off. "
    "Then verdict. met[i] answers requirement i in order. If the "
    "edit failed, pick ONE corrective_action: reroll_seed (edit ignored "
    "or malformed - try different noise), strengthen_edit (change "
    "present but too weak), rephrase_instruction (the wording misled "
    "the editor - provide revised_instruction, an imperative under 40 "
    "words), escalate_full_res (draft too coarse to judge). The LAST "
    "requirement is usually a PRESERVATION check: verify nothing OUTSIDE "
    "the requested change materially changed — no reframe/zoom, no "
    "objects appearing or vanishing, no global color/style shift. Judge "
    "it strictly against the BEFORE image. "
    "Describe failures in WORDS: never refer to a requirement by its "
    "number. what_failed must be one full sentence naming the object "
    "and what is wrong (e.g. 'The hot tub is still raised above the "
    "deck instead of sunken'); when the verdict is pass, set "
    "what_failed to exactly 'nothing failed'. When the verdict is "
    "fail, failure_category must never be 'none' — pick the closest "
    "category."
)

_FAILS = 0            # consecutive failures -> circuit breaker

_DEGENERATE_WHAT_FAILED = "unspecified failure (degenerate judge output)"

_REPARSE = object()   # sentinel: server answered, content unparseable


def _degenerate(s) -> bool:
    """A prose field with fewer than 3 alphabetic words — the lazy
    constrained-decoding failure mode ('1', '],  ', 'Requirement one')."""
    return len(re.findall(r"[A-Za-z]+", s or "")) < 3


def judge_edit(before_b64: str, after_b64: str, instruction: str,
               requirements: list, url: str, model: str,
               timeout_s: float, attempts: list | None = None) -> dict | None:
    """Returns the verdict dict, or None when the judge is unavailable
    (timeout/error/breaker) — caller fails open."""
    global _FAILS
    if _FAILS >= 3:
        return None
    reqs = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(requirements))
    memory = ""
    if attempts:
        memory = "\nPrior attempts on this edit (do not repeat them):\n" + \
            "\n".join(f"- {a}" for a in attempts[-3:])
    content = [{"type": "text",
                "text": (f"Edit instruction: {instruction}\n"
                         f"Requirements:\n{reqs}{memory}"
                         + ("" if before_b64 else
                            "\nThere is only ONE image (a fresh "
                            "generation): judge each requirement "
                            "against it directly."))}]
    if before_b64:
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{before_b64}"}})
    content.append({"type": "image_url", "image_url": {
        "url": f"data:image/jpeg;base64,{after_b64}"}})
    body = {
        "model": model,
        "temperature": 0.1,
        # worst case fail verdict fills analysis(240) + what_failed(160) +
        # revised_instruction(300) + fail_box + met[] — 200 truncated the
        # JSON mid-string (observed as a parse error, fail-open promote)
        "max_tokens": 450,
        "messages": [
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "verdict", "schema": _SCHEMA}},
    }
    data = json.dumps(body).encode()

    def _call(last_try: bool) -> dict | None:
        """None = transport failure (breaker counts it). _REPARSE = the
        server answered but the content is not valid JSON — the model
        occasionally pads inter-token whitespace after a mid-word
        maxLength cut until max_tokens truncates the JSON; that is a
        model stall, retriable once, and only the final try feeds the
        breaker."""
        global _FAILS
        req = urllib.request.Request(
            f"{url}/chat/completions", data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                out = json.loads(r.read())
        except Exception as e:
            _FAILS += 1
            log.warning("judge call failed (%d consecutive): %s", _FAILS, e)
            return None
        try:
            v = json.loads(out["choices"][0]["message"]["content"])
        except Exception as e:
            if last_try:
                _FAILS += 1
                log.warning("judge output unparseable (%d consecutive): %s",
                            _FAILS, e)
                return None
            log.warning("judge output unparseable - retrying once: %s", e)
            return _REPARSE
        _FAILS = 0
        met = v.get("met") or []
        # align: pad/truncate to the requirement count; reconcile
        met = (met + [True] * len(requirements))[:max(len(requirements), 1)]
        v["met"] = met
        if v.get("verdict") == "pass" and not all(met):
            v["verdict"] = "fail"
        return v

    v = _call(last_try=False)
    if v is _REPARSE:
        v = _call(last_try=True)
    if v is None:
        return None
    # Degenerate-fail guard: a fail verdict whose prose carries no
    # information is UNTRUSTED — the category is usually junk too, and
    # the retry router stalls on it. One fresh retry; if still
    # degenerate, coerce onto the too_weak ladder (which climbs) rather
    # than none (which stalls).
    if v.get("verdict") == "fail" and (
            _degenerate(v.get("what_failed")) or _degenerate(v.get("analysis"))):
        log.warning("judge fail verdict is degenerate (what_failed=%r) - "
                    "retrying once", v.get("what_failed"))
        v2 = _call(last_try=True)
        if v2 is not None and v2 is not _REPARSE:
            v = v2
        if v.get("verdict") == "fail" and (
                _degenerate(v.get("what_failed"))
                or _degenerate(v.get("analysis"))):
            v["failure_category"] = "too_weak"
            v["what_failed"] = _DEGENERATE_WHAT_FAILED
    if v.get("verdict") == "fail" and v.get("failure_category") in (None, "none"):
        # a real failure routed as "none" stalls the constraint ladder
        v["failure_category"] = "too_weak"
    return v
