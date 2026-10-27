# ming-tts-runpod

Runpod serverless worker for [Ming-3.0-tts-0.5b](https://github.com/inclusionAI/Ming-3.0-tts) (MIT).
The image is built by GitHub Actions and pushed to `ghcr.io/<owner>/ming-tts-runpod`.

Request `input`: `task`, `text`, `language_id`, `controls`, `instruction`, `prompt_wav_b64`, `prompt_text`,
`use_spk_emb`, `use_zero_spk_emb`, `cfg`, `sigma`, `temperature`, `max_length`, `raw_text`, `seed`.
Response: `wav_b64`, `sample_rate`, `seconds`.

Korean: design a voice with `VOICE_DESIGN` (style text), then use that wav as `prompt_wav_b64` with
`task: MULTILINGUAL`, `language_id: Korean`, `use_spk_emb: true`.
