# ComfyUI pinned to a release tag; the comfyui-psirens node pack is
# bind-mounted (see compose.yaml), its python deps are baked in here.
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      git curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ARG COMFY_REF=v0.42.5
RUN git clone --depth 1 --branch ${COMFY_REF} \
      https://github.com/Comfy-Org/ComfyUI /opt/comfyui \
 && pip install --no-cache-dir --break-system-packages -r /opt/comfyui/requirements.txt
COPY custom_nodes/comfyui-psirens/requirements.txt /tmp/req-psirens.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/req-psirens.txt
WORKDIR /opt/comfyui
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
