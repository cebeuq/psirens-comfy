"""Fetch every model the stack needs into ./models (idempotent).

  models/hf/      Hugging Face cache for vLLM + the voice service
  models/comfy/   ComfyUI model folders (diffusion_models, text_encoders, vae)

Usage: docker compose run --rm downloader [group ...]
Groups: llm, voice, comfy (default: all)
"""
import hashlib
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
# (repo, file in repo, ComfyUI models subfolder). The official FLUX 3 Image
# Klein repo is gated: accept it on Hugging Face and set HF_TOKEN in .env,
# or point KLEIN_REPO at an ungated byte-identical mirror (files are
# checked against SHA256 below either way).
KLEIN = os.environ.get("KLEIN_REPO") or "black-forest-labs/flux-3-image-klein-comfy"
SHA256 = {
    "flux3-image-klein-int8-convrot.safetensors":
        "847d8933548afd93df4af2cac2498e70891bf28d9dae06d695f536992e168fe2",
    "qwen3vl_4b_int8_convrot.safetensors":
        "32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c",
    "flux2-vae.safetensors":
        "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
}
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
        staged = os.path.join(COMFY, ".staging", path)
        want = SHA256.get(os.path.basename(path))
        if want:
            h = hashlib.sha256()
            with open(staged, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != want:
                raise SystemExit(f"SHA256 mismatch for {path}: {h.hexdigest()}")
            print(f"[comfy] sha256 ok {os.path.basename(path)}", flush=True)
        os.makedirs(dest, exist_ok=True)
        os.replace(staged, target)


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
