# Orchestrator: FastAPI voice service (speech-to-text, prosody, commit
# detector, LLM scene editor) + the browser UI. CUDA for NeMo STT and the
# prosody model; the editor LLM itself runs in the vllm service.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir \
    "fastapi>=0.115" "uvicorn[standard]>=0.30" "websockets>=12" httpx \
    "pydantic>=2" pyyaml numpy soundfile "pytest>=8"

# NeMo ASR: Nemotron streaming STT + Parakeet end-of-utterance model
RUN pip install --no-cache-dir "nemo_toolkit[asr]==3.0.0"

WORKDIR /srv/app
ENV PYTHONPATH=/srv/app
CMD ["uvicorn", "psirens_voice.main:app", "--host", "0.0.0.0", "--port", "8000"]
