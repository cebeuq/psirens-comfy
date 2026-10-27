# GPU renderer image. Base: NGC vLLM image (cached on spark1) — ships
# arm64 CUDA torch + transformers known-good on GB10. We add diffusers for
# ZImagePipeline. The mock mode also runs fine in this image.
FROM nvcr.io/nvidia/vllm:26.05-py3

# --no-deps on diffusers would lose its real deps; a plain install is safe
# here because its requirements (huggingface-hub, safetensors, Pillow,
# numpy, regex) are already satisfied and it does NOT pin torch.
RUN pip install --no-cache-dir "diffusers>=0.36" "websockets>=12" "pillow>=10" accelerate

WORKDIR /srv/renderer
ENV PYTHONPATH=/srv/renderer

CMD ["python", "-m", "psirens_renderer.main"]
