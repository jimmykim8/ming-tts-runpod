# Runpod serverless worker: Ming-3.0-tts-0.5b (weights baked into the image).
# Based on the upstream docker/Dockerfile (torch 2.6 / CUDA 12.4, transformers 4.46.3), trimmed to what the 0.5B model imports.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torchaudio==2.6.0 \
    transformers==4.46.3 tokenizers accelerate==1.3.0 huggingface_hub \
    numpy soundfile==0.12.1 tqdm loguru \
    x_transformers==2.9.2 diffusers==0.33.0 onnxruntime inflect einops packaging \
    runpod

# Ming code pinned to a known commit
ARG MING_COMMIT=028c1eace193e7acdb256453da89b6e9207e052b
RUN git clone https://github.com/inclusionAI/Ming-3.0-tts.git /app/ming && cd /app/ming && git checkout ${MING_COMMIT}

# Model weights (about 3 GB) baked in so cold starts do not download
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('inclusionAI/Ming-3.0-tts-0.5b', local_dir='/models/Ming-3.0-tts-0.5b', allow_patterns=['*.json','*.txt','*.onnx','*.safetensors'])"

COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
