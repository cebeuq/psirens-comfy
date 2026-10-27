"""Fetch every model the stack needs into ./models (idempotent).

  models/hf/      Hugging Face cache for vLLM + the voice service
  models/comfy/   ComfyUI model folders (diffusion_models, text_encoders, vae)

Usage: docker compose run --rm downloader [group ...]
Groups: llm, voice, comfy (default: all)
"""
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download

HF_CACHE = "/models/hf/hub"   # == HF_HOME=/models/hf for vLLM/NeMo
COMFY = "/models/comfy"

LLM = [os.environ.get("EDITOR_MODEL", "cyankiwi/Qwen3.5-4B-AWQ-4bit")]
VOICE = [
    "nvidia/nemotron-speech-streaming-en-0.6b",
    "nvidia/parakeet_realtime_eou_120m-v1",
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
]
# (repo, file in repo, ComfyUI models subfolder). The FLUX 3 Image Klein
# repo is gated: accept it on Hugging Face and set HF_TOKEN in .env.
KLEIN = "black-forest-labs/flux-3-image-klein-comfy"
COMFY_FILES: list[tuple[str, str, str]] = [
    (KLEIN, "diffusion_models/" + os.environ.get(
        "KLEIN_FILE", "flux3-image-klein-int8-convrot.safetensors"),
     "diffusion_models"),
    (KLEIN, "text_encoders/" + os.environ.get(
        "KLEIN_TE_FILE", "qwen3vl_4b_int8_convrot.safetensors"),
     "text_encoders"),
    (KLEIN, "vae/flux2-vae.safetensors", "vae"),
]


def fetch_snapshots(repos):
    for repo in repos:
        print(f"[hf] {repo}", flush=True)
        snapshot_download(repo, cache_dir=HF_CACHE)


def fetch_comfy():
    for repo, path, sub in COMFY_FILES:
        dest = os.path.join(COMFY, sub)
        target = os.path.join(dest, os.path.basename(path))
        if os.path.exists(target):
            print(f"[comfy] have {target}", flush=True)
            continue
        print(f"[comfy] {repo}/{path} -> {sub}/", flush=True)
        hf_hub_download(repo, path, local_dir=os.path.join(COMFY, ".staging"))
        os.makedirs(dest, exist_ok=True)
        os.replace(os.path.join(COMFY, ".staging", path), target)


def main(argv):
    groups = set(argv) or {"llm", "voice", "comfy"}
    if "llm" in groups:
        fetch_snapshots(LLM)
    if "voice" in groups:
        fetch_snapshots(VOICE)
    if "comfy" in groups:
        fetch_comfy()
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
