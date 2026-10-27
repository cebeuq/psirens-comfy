"""EditStrategy registry: the five user-switchable agentic edit modes.

The strategy decides three things per commit: whether the parse call
always sees the canvas, how the draft dict is enriched (grounding
target / focus / plans / budgets), and which judge behavior the
renderer runs. STANDARD is the null strategy — byte-for-byte today's
behavior — and every other mode degrades to it rung by rung (absent
draft keys = today's code paths).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditStrategy:
    name: str
    fallback: str
    vision_always: bool = False     # attach canvas to every commit parse
    use_target: bool = False        # stencil mask from parser target box
    grounded_judge: bool = False    # failure_category/fail_box + memory
    n_drafts: int = 1               # seeds tried per iteration (atelier)
    budget_ms: int = 2200
    max_iters: int = 2
    contact_verify: bool = False    # crop-zoom contact check (atelier)
    # constraint-ladder climbs allowed above the starting rung: when the
    # judge says the CONSTRAINT was the failure (object unchanged / too
    # weak), the retry runs one rung looser (freeze suffix lifted,
    # stencil widened or dropped, then full-redraw tier + fresh seed).
    # Fast tiers keep their latency promise with one climb; deep tiers
    # may climb to the top.
    max_climb: int = 1


STRATEGIES: dict[str, EditStrategy] = {
    "standard": EditStrategy("standard", "standard"),
    "stencil": EditStrategy("stencil", "standard", use_target=True),
    "surveyor": EditStrategy("surveyor", "stencil", vision_always=True,
                             use_target=True, grounded_judge=True,
                             budget_ms=3200, max_climb=2),
    "atelier": EditStrategy("atelier", "surveyor", vision_always=True,
                            use_target=True, grounded_judge=True,
                            n_drafts=3, budget_ms=15000, max_iters=3,
                            contact_verify=True, max_climb=2),
    "switchboard": EditStrategy("switchboard", "surveyor",
                                vision_always=True, use_target=True,
                                grounded_judge=True, budget_ms=6500,
                                max_climb=2),
}

# switchboard routing: difficulty -> executing strategy
ROUTER = {"trivial": "stencil", "standard": "stencil",
          "relational": "surveyor", "open": "atelier"}


def resolve(name: str) -> EditStrategy:
    return STRATEGIES.get(name, STRATEGIES["standard"])
