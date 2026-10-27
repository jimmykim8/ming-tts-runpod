# -*- coding: utf-8 -*-
"""Runpod serverless worker for Ming-3.0-tts-0.5b (inclusionAI). 2026-10-27

Input (job["input"]):
  task             "VOICE_DESIGN" | "MULTILINGUAL" | "TTS" | "IP" | "FREEFORM" | "BASIC" | "EMOTION" ...  (Ming task names)
  text             text to speak
  language_id      e.g. "Korean" / "ko" (required for MULTILINGUAL)
  controls         dict, e.g. {"style": "..."} for VOICE_DESIGN, {"ip": "US_James"} for IP
  instruction      free-form instruction (FREEFORM)
  prompt_wav_b64   reference audio (wav/mp3 bytes, base64) for cloning / MULTILINGUAL
  prompt_text      transcript of the reference audio
  use_spk_emb / use_zero_spk_emb, cfg, sigma, temperature, max_length, raw_text, seed
Output: {"wav_b64": ..., "sample_rate": int, "seconds": float}
"""
import base64
import io
import os
import random
import sys
import tempfile
import time

import numpy as np
import runpod
import soundfile as sf
import torch

sys.path.insert(0, "/app/ming")
from ming_tts import MingTTS  # noqa: E402

MODEL_PATH = os.environ.get("MING_MODEL_PATH", "/models/Ming-3.0-tts-0.5b")
t0 = time.time()
MODEL = MingTTS(MODEL_PATH, device="cuda:0", text_normalization=False)
print("Ming loaded in %.1fs, sample_rate=%s" % (time.time() - t0, MODEL.sample_rate), flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def handler(job):
    inp = job.get("input") or {}
    seed_everything(int(inp.get("seed", 1895)))
    tmp = None
    try:
        kwargs = dict(task=inp.get("task", "TTS"), text=inp["text"], language_id=inp.get("language_id"),
                      instruction=inp.get("instruction"), controls=inp.get("controls"),
                      prompt_text=inp.get("prompt_text"),
                      use_spk_emb=bool(inp.get("use_spk_emb", False)), use_zero_spk_emb=bool(inp.get("use_zero_spk_emb", False)),
                      cfg=float(inp.get("cfg", 2.0)), sigma=float(inp.get("sigma", 0.25)), temperature=float(inp.get("temperature", 0.0)),
                      max_length=int(inp.get("max_length", 24)), raw_text=bool(inp.get("raw_text", False)))
        if inp.get("prompt_wav_b64"):
            raw = base64.b64decode(inp["prompt_wav_b64"])
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, data, sr)
            tmp.close()
            kwargs["prompt_wav_path"] = tmp.name
        t = time.time()
        wav = MODEL.synthesize(**kwargs)
        audio = wav.squeeze().float().cpu().numpy() if hasattr(wav, "cpu") else np.asarray(wav).squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio, MODEL.sample_rate, format="WAV", subtype="PCM_16")
        return {"wav_b64": base64.b64encode(buf.getvalue()).decode(), "sample_rate": int(MODEL.sample_rate),
                "seconds": round(len(audio) / MODEL.sample_rate, 3), "gen_seconds": round(time.time() - t, 2)}
    except Exception as e:  # report instead of crashing the worker
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-3000:]}
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
