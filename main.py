import asyncio
import base64
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Chutes Storyteller")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "").strip()
LLM_ENDPOINT = os.getenv("STORY_LLM_ENDPOINT", "https://llm.chutes.ai/v1/chat/completions")
LLM_MODEL = os.getenv("STORY_LLM_MODEL", "moonshotai/Kimi-K2.5-TEE")
IMAGE_ENDPOINT = os.getenv("STORY_IMAGE_ENDPOINT", "https://chutes-z-image-turbo.chutes.ai/generate")
TTS_ENDPOINT = os.getenv("STORY_TTS_ENDPOINT", "https://chutes-kokoro.chutes.ai/speak")
TTS_VOICE = os.getenv("STORY_TTS_VOICE", "af_heart")
IMAGE_COUNT = int(os.getenv("STORY_IMAGE_COUNT", "3"))
IMAGE_WIDTH = int(os.getenv("STORY_IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.getenv("STORY_IMAGE_HEIGHT", "576"))
IMAGE_STEPS = int(os.getenv("STORY_IMAGE_STEPS", "10"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("STORY_HTTP_TIMEOUT", "600"))

# Keep prompt + result states in memory only. Enough for a lightweight demo.
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = asyncio.Lock()


class StoryRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=2000)


@dataclass
class ScenePlan:
    text: str
    image_prompt: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_payload(job_id: str) -> Dict[str, Any]:
    return jobs[job_id]


async def _set_job_state(job_id: str, **updates: Any) -> None:
    async with jobs_lock:
        state = jobs[job_id]
        state.update(updates)
        state["updatedAt"] = _now_iso()


def _strip_json_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return text.strip()


def _find_json_blob(text: str) -> str | None:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def _extract_stages(text: str) -> tuple[str, str, List[ScenePlan]]:
    """Turn LLM output into (title, story, scenes).

    This parser intentionally handles both structured JSON and free-form output.
    """
    clean = _strip_json_fences(text)
    payload = None

    for body in (clean, _find_json_blob(clean) or ""):
        if not body:
            continue
        try:
            payload = json.loads(body)
            break
        except Exception:
            continue

    if not isinstance(payload, dict):
        scene_texts = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
        parsed = [ScenePlan(text=segment, image_prompt=segment[:120]) for segment in scene_texts[:IMAGE_COUNT]]
        title = parsed[0].text[:80] if parsed else "Untitled Story"
        story = "\n\n".join(scene.text for scene in parsed)
        return title, story.strip() or text.strip(), parsed

    title = (payload.get("title") or "Untitled Story").strip()
    summary = (payload.get("summary") or "").strip()

    scenes_raw = payload.get("scenes")
    parsed: List[ScenePlan] = []
    if isinstance(scenes_raw, list):
        for scene in scenes_raw[:IMAGE_COUNT]:
            if not isinstance(scene, dict):
                continue
            scene_text = str(scene.get("text", "")).strip()
            if not scene_text:
                continue
            scene_prompt = str(
                scene.get("image_prompt")
                or scene.get("visual_prompt")
                or scene.get("prompt")
                or scene_text[:200]
            ).strip()
            parsed.append(ScenePlan(text=scene_text, image_prompt=scene_prompt))

    if not parsed:
        if summary:
            parsed.append(ScenePlan(text=summary, image_prompt=summary))
        if title:
            title_hint = f"{title}: {summary[:120]}" if summary else f"A story called {title}"
            if not parsed:
                parsed.append(ScenePlan(text=title_hint, image_prompt=title_hint))

    story_chunks: List[str] = []
    if summary:
        story_chunks.append(summary)
    story_chunks.extend(scene.text for scene in parsed)
    combined = "\n\n".join([chunk for chunk in story_chunks if chunk]).strip()
    if not combined:
        combined = text.strip()

    return title, combined, parsed


def _safe_b64(data: bytes, mime_type: str) -> Dict[str, str]:
    return {
        "b64": base64.b64encode(data).decode("ascii"),
        "mimeType": mime_type,
    }


def _extract_b64_value(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, str):
        return value.strip().split(",")[-1]
    return None


def _decode_binary_response(response: httpx.Response, *, fallback_mime: str) -> Dict[str, str]:
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type.startswith("application/json"):
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("JSON response did not contain a dictionary payload")

        for key in ("mimeType", "mime_type", "mime"):
            mime = data.get(key)
            if isinstance(mime, str):
                fallback_mime = mime

        for key in ("image_b64", "image_base64", "image", "audio", "audio_base64", "audio_data", "data"):
            if isinstance(data, dict) and data.get(key):
                value = _extract_b64_value(data[key])
                if not value:
                    continue
                return {"b64": value, "mimeType": str(data.get("mime_type", fallback_mime))}
        images = data.get("images")
        if isinstance(images, (list, tuple)):
            value = _extract_b64_value(images[0])
            if value:
                return {"b64": value, "mimeType": fallback_mime}

        raise ValueError("JSON response did not contain binary payload")

    if not response.content:
        raise ValueError("Empty response payload")
    return _safe_b64(response.content, content_type or fallback_mime)


async def _call_endpoint(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    payload: Dict[str, Any],
    method: str = "POST",
) -> httpx.Response:
    response = await client.request(
        method,
        endpoint,
        headers={
            "Authorization": f"Bearer {CHUTES_API_KEY}",
            "Content-Type": "application/json",
            "X-Client": "chutes-storyteller",
        },
        json=payload,
    )
    if response.status_code != 200:
        raise RuntimeError(f"{endpoint} responded with {response.status_code}: {response.text[:500]}")
    return response


async def _run_story_pipeline(job_id: str, prompt: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            if not CHUTES_API_KEY:
                raise RuntimeError("Missing CHUTES_API_KEY environment variable")

            await _set_job_state(
                job_id,
                status="running",
                progress=2,
                stage="Planning",
                message="Story prompt received, generating narrative structure.",
            )

            llm_payload = {
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a creative storyteller. Return strict JSON only."
                    },
                    {
                        "role": "user",
                        "content": (
                            "Create a short atmospheric story from this idea: "
                            f"{prompt}\n\n"
                            "Return JSON only with this exact structure:\n"
                            '{\"title\": \"string\", \"summary\": \"string\", \"scenes\": [\n'
                            '{\"text\": \"string\", \"image_prompt\": \"string\"}, ...] }. '
                            "Use 2-5 scenes and keep every scene vivid and readable. "
                            "Do not include markdown."
                        ),
                    },
                ],
                "max_tokens": 1800,
                "temperature": 0.85,
            }

            llm_response = await _call_endpoint(client, endpoint=LLM_ENDPOINT, payload=llm_payload)
            llm_body = llm_response.json()
            llm_text = (
                llm_body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            title, full_story_text, scenes = _extract_stages(llm_text)

            if not scenes:
                scenes = [ScenePlan(text=full_story_text[:1200], image_prompt=full_story_text[:160])]

            await _set_job_state(
                job_id,
                status="running",
                progress=28,
                stage="Scenes",
                message=f"Story assembled into {len(scenes)} scene(s). Preparing image prompts.",
            )

            produced_images: List[Dict[str, str]] = []
            image_limit = min(len(scenes), IMAGE_COUNT)
            for idx, scene in enumerate(scenes[:image_limit], start=1):
                await _set_job_state(
                    job_id,
                    stage="Illustrations",
                    progress=min(75, 28 + idx * 12),
                    message=f"Rendering image {idx} of {image_limit}.",
                )
                image_payload = {
                    "prompt": scene.image_prompt,
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                    "num_inference_steps": IMAGE_STEPS,
                    "guidance_scale": 2.0,
                }
                image_response = await _call_endpoint(client, endpoint=IMAGE_ENDPOINT, payload=image_payload)
                image_data = _decode_binary_response(image_response, fallback_mime="image/png")
                produced_images.append(
                    {
                        "caption": scene.image_prompt,
                        "b64": image_data["b64"],
                        "mimeType": image_data["mimeType"],
                    }
                )

            await _set_job_state(
                job_id,
                status="running",
                progress=82,
                stage="Narration",
                message="All images ready. Creating full narration audio.",
            )

            tts_payload = {
                "text": full_story_text,
                "voice": TTS_VOICE,
            }
            tts_response = await _call_endpoint(client, endpoint=TTS_ENDPOINT, payload=tts_payload)
            tts_data = _decode_binary_response(
                tts_response,
                fallback_mime="audio/wav",
            )

            await _set_job_state(
                job_id,
                status="completed",
                progress=100,
                stage="Done",
                message="Story generation complete.",
                result={
                    "title": title,
                    "story": full_story_text,
                    "scenes": [
                        {
                            "text": scene.text,
                            "imagePrompt": scene.image_prompt,
                        }
                        for scene in scenes[:image_limit]
                    ],
                    "images": produced_images,
                    "audio": {
                        "b64": tts_data["b64"],
                        "mimeType": tts_data["mimeType"],
                    },
                    "prompt": prompt,
                },
            )
    except Exception as exc:
        await _set_job_state(
            job_id,
            status="failed",
            progress=100,
            stage="Error",
            message=f"Generation failed: {str(exc)}",
            result={"error": str(exc)},
        )


@app.post("/api/story")
async def start_story(payload: StoryRequest) -> JSONResponse:
    job_id = uuid.uuid4().hex
    state: Dict[str, Any] = {
        "jobId": job_id,
        "status": "queued",
        "progress": 0,
        "stage": "Queued",
        "message": "Queued for generation.",
        "result": None,
        "createdAt": _now_iso(),
        "updatedAt": _now_iso(),
    }
    async with jobs_lock:
        jobs[job_id] = state

    asyncio.create_task(_run_story_pipeline(job_id, payload.prompt))
    return JSONResponse(_status_payload(job_id))


@app.get("/api/story/{job_id}")
async def story_status(job_id: str) -> JSONResponse:
    async with jobs_lock:
        state = jobs.get(job_id)
    if state is None:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return JSONResponse(state)


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
