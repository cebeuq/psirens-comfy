"""Probe the delta-editor parser directly: feed scripted utterances against
a scripted scene and print the strategy-relevant fields the model returned
(grounding / target / contact / edit_difficulty). Runs in the orchestrator
container:

  docker compose exec orchestrator python /srv/app/tools/parse_probe.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, "/srv/app")

from psirens_voice.editor.client import DeltaEditorClient          # noqa: E402
from psirens_voice.scene.state import Entity, SceneState           # noqa: E402

CASES = [
    ("Add a bright red water slide with a Supreme logo on the right.",
     "add"),
    ("Make the water slide go into the pool.", "relational"),
    ("Make the house bigger.", "attribute"),
    ("Turn the whole scene into a snowy winter wonderland.", "open"),
    ("Okay, now in front of the bedroom, make the entire lawn grass and "
     "add a pool in the middle. And with a hot tub.", "multi-part"),
]


async def main() -> None:
    state = SceneState(
        subject="a modern house",
        setting="in Joshua Tree with a pool that has a rock volcano design",
        entities=[Entity(id="water_slide", name="water slide",
                         description="a bright red water slide with a "
                                     "Supreme logo",
                         box={"x": 0.68, "y": 0.45, "w": 0.25, "h": 0.3})],
    )
    client = DeltaEditorClient(
        base_url=os.environ.get("EDITOR_URL", "http://vllm:8000/v1"),
        model=os.environ.get("EDITOR_MODEL",
                             "nvidia/Qwen3.6-35B-A3B-NVFP4"))
    for utt, label in CASES:
        patch, ms, retries = await client.propose_patch(
            state, "tone: energy=med valence=med tempo=med hesitation=low",
            utt, max_changed_fields=3, below_magnitude=0.5)
        print(f"--- [{label}] {utt}")
        print(f"    difficulty={patch.edit_difficulty}"
              f" target={patch.target and patch.target.model_dump()}"
              f" contact={patch.contact and patch.contact.model_dump()}")
        print(f"    grounding={patch.grounding!r}")
        print(f"    magnitude={patch.edit_magnitude}"
              f" instr={patch.edit_instruction!r} ({ms}ms)")


if __name__ == "__main__":
    asyncio.run(main())
