"""Render a short scripted edit chain through ComfyCanvas and time it.

docker compose run --rm --no-deps -v ./data/bench:/out renderer \
    python -m tools.bench_comfy [--size 1024] [--steps 8] [--no-upsample]
Frames land in /out (data/bench on the host).
"""
import argparse
import base64
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--tau", type=float, default=None)
ap.add_argument("--no-upsample", action="store_true")
ap.add_argument("--out", default="/out")
args = ap.parse_args()
os.environ["KLEIN_STEPS"] = str(args.steps)
if args.tau is not None:
    os.environ["PSIRENS_PASS_TAU"] = str(args.tau)
if args.no_upsample:
    os.environ["PSIRENS_UPSAMPLE"] = "0"

from psirens_renderer.comfy_canvas import ComfyCanvas  # noqa: E402

SCENE = ("a cozy wooden cabin in a snowy pine forest at dusk, warm light in "
         "the windows, photograph")
EDITS = [
    "Make the sky a deep purple with a few stars",
    "Add a red sled in the snow in front of the cabin",
    "Keep everything exactly the same",
    "Make it snow heavily",
]

c = ComfyCanvas()
c.size = args.size
t = time.time()
c.load()
print(f"load+warmup {time.time() - t:.1f}s", flush=True)
tag = f"s{args.size}_k{args.steps}{'_noup' if args.no_upsample else ''}"


def save(name, b64):
    with open(os.path.join(args.out, f"{tag}_{name}.jpg"), "wb") as f:
        f.write(base64.b64decode(b64))


t = time.time()
save("0_root", c.render(SCENE, 0.8, [], lane="committed", source="commit"))
print(f"root {time.time() - t:.2f}s", flush=True)
for i, e in enumerate(EDITS, 1):
    t = time.time()
    save(f"{i}_edit", c.render(SCENE, 0.5, [], instruction=e,
                               lane="committed", source="commit"))
    print(f"edit {i} {time.time() - t:.2f}s  {e!r}", flush=True)
t = time.time()
save("p_provisional", c.render(SCENE, 0.25, [], instruction="make the cabin red",
                               lane="provisional", source="overlay"))
print(f"provisional {time.time() - t:.2f}s", flush=True)
