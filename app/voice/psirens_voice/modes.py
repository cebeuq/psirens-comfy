"""Architecture modes, selectable per session from the UI.

BATON (the owner's chosen base) plus the v5 mid-sentence designs layered
on it — see MODES below for what each does while the user is talking."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModePolicy:
    name: str
    label: str
    # what happens mid-sentence: "verbatim" (words pasted into the prompt,
    # zero calls) | "speculative" (early editor call, provisional) | "none"
    mid_sentence: str
    # when the editor commits: "pause" (the classic commit gate) |
    # "clause" (short 0.6s clause boundaries commit directly)
    commit_trigger: str
    # when a canvas frame is attached to an editor call:
    # "reactive" (reactive words or verification debt) | "always" |
    # "piggyback" (reactive OR every 3rd call) | "never"
    vision: str
    # append-only conversation (prefix-cache friendly) vs stateless calls
    conversation: bool
    # seconds of post-render silence before a critic call; None = no critic
    idle_critic_s: float | None
    # ---- v4 edit-native fields ----
    # who writes the image-edit instruction:
    #   "user" — the user's own cleaned directive, verbatim. Lossless:
    #            part-level edits ("make the headlights yellow") reach the
    #            image model unmangled instead of being laundered through
    #            the state schema and re-phrased. Moves still composite.
    #   "code" — deterministic phrase_edit from the structured patch
    #   "llm"  — the editor authors it in the same call, validated
    instruction_author: str = "user"
    # committed edits before a drift-cleaning refresh edit replaces the
    # normal delta edit (CONSERVATOR); None = never
    drift_budget: int | None = None
    # ---- v5 mid-sentence fields ----
    # show speculative results provisionally (True) or keep them hidden
    # until promotion proves them right (PRESTIGE = False)
    spec_display: bool = True
    # start the commit parse the moment the EOU model signals done,
    # instead of waiting for the confirmed gate (~1s head start; SHUTTER)
    eager_eou: bool = False
    # route provisional frames to a separate sketch surface in the UI
    # instead of the main canvas (SCRIM)
    sketch_surface: bool = False


# SCRIM is the shipped architecture (the owner's pick after the v5 trial):
# BATON base (deterministic user-verbatim instructions, strict committed
# queue, conversation cache) + mid-sentence sketches on a separate PiP
# surface — the committed canvas only ever shows completed utterances.
# The other trialed policies (prestige/stet/shutter behaviors) remain
# engine capabilities behind ModePolicy fields if ever wanted again.
MODES: dict[str, ModePolicy] = {
    # conversation=False: the timeline LEDGER is the model's memory now.
    # Feeding it its own prior turns as well made it pattern-complete old
    # answers instead of deciding fresh (observed: echoed reasoning,
    # empty patches, suppressed history_action).
    "scrim": ModePolicy(
        name="scrim", label="SCRIM",
        mid_sentence="verbatim", commit_trigger="pause",
        vision="reactive", conversation=False, idle_critic_s=None,
        instruction_author="user", sketch_surface=True),
}

DEFAULT_MODE = "scrim"

# words that mean the user is talking ABOUT the current image — a frame
# must ride along so the model sees what they see
REACTIVE_WORDS = {"that", "those", "it", "one", "why", "where", "looks",
                  "look", "there", "wrong", "weird", "again", "still",
                  "background", "foreground", "left", "right", "top",
                  "bottom", "behind", "front"}


# directives about orientation/facing need the CURRENT image: authored
# geometry ("rotate 90 degrees clockwise") is a blind guess without it
ORIENTATION_WORDS = frozenset({
    "face", "faces", "facing", "faced", "rotate", "rotated", "turn",
    "turned", "point", "pointing", "toward", "towards", "orient",
    "flip", "mirror", "angle", "angled",
    # removal directives also need eyes: category deletes ("remove all
    # the furniture") are grounded by enumerating the ATTACHED image,
    # not by imagining what the scene probably contains
    "remove", "removes", "delete", "deletes", "erase", "clear",
    "everything", "rid",
    # placement needs eyes too: destination boxes chosen blind kept
    # landing on top of other furniture
    "move", "moves", "moving", "put", "place", "placing", "shift",
    "slide", "against", "beside", "under", "between",
})


def wants_frame(policy: ModePolicy, directive: str, verify_debt: bool,
                calls_used: int) -> bool:
    if policy.vision == "never":
        return False
    if policy.vision == "always":
        return True
    words = {w.strip(".,!?").lower() for w in directive.split()}
    reactive = (bool(words & REACTIVE_WORDS) or verify_debt
                or bool(words & ORIENTATION_WORDS))
    if policy.vision == "reactive":
        return reactive
    if policy.vision == "piggyback":
        return reactive or (calls_used % 3 == 2)
    return False
