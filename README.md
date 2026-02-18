# Chutes Storyteller

A collaborative storytelling web app built around Chutes models. Users provide a single prompt and the app generates:

- A short atmospheric story text.
- Multiple scene illustrations.
- A complete narration audio track.

Music generation is intentionally omitted for now.

## Features

- One input prompt to start.
- Visual + progress updates while generation runs.
- Gallery of generated illustrations.
- Playable audio narration once generation completes.

## Tech Stack

- FastAPI backend.
- Vanilla JS frontend.
- Chutes endpoints:
  - `moonshotai/Kimi-K2.5` for story writing.
  - `chutes-z-image-turbo.chutes.ai` for text-to-image.
  - `chutes-kokoro.chutes.ai` for text-to-speech.

## Run locally

```bash
cd chutes-storyteller
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in CHUTES_API_KEY
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Deployment (Render)

Use Render with:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Set `CHUTES_API_KEY` as an environment variable in Render.
