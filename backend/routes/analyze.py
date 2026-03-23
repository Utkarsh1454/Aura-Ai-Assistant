# ─────────────────────────────────────────────────────────────
# Route: /analyze  – Main multimodal emotion analysis endpoint
# ─────────────────────────────────────────────────────────────
import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ai.face.predict import predict_face
from ai.voice.predict import predict_voice
from ai.fusion.fusion import fuse
from llm.prompt_engine import build_prompt
from llm.ollama_client import query_ollama, stream_ollama
from backend.config import EMOTION_LOG_PATH

router = APIRouter(prefix="/api", tags=["Analyze"])


def _log_emotion(face: str, voice: str, final: str) -> None:
    """Append emotion event to a JSON Lines log file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "face": face,
        "voice": voice,
        "final": final,
    }
    os.makedirs(os.path.dirname(EMOTION_LOG_PATH), exist_ok=True)
    with open(EMOTION_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


@router.post("/analyze")
async def analyze(
    frame: UploadFile = File(..., description="JPEG/PNG face image"),
    audio: UploadFile = File(..., description="WAV audio clip (16 kHz, mono)"),
):
    """
    Accept a face image + audio clip, return detected emotions and an
    AI-generated wellness response from the local LLaMA 3 model.
    """
    # ── Face ────────────────────────────────────────────────
    img_bytes = await frame.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Send JPEG or PNG.")

    # ── Audio ───────────────────────────────────────────────
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    # ── Inference ───────────────────────────────────────────
    face_emotion = predict_face(image)
    voice_emotion = predict_voice(audio_bytes)
    final_emotion = fuse(face_emotion, voice_emotion)

    # ── LLM Response ────────────────────────────────────────
    prompt = build_prompt(final_emotion)
    response = query_ollama(prompt)

    # ── Logging ─────────────────────────────────────────────
    _log_emotion(face_emotion, voice_emotion, final_emotion)

    return {
        "face_emotion": face_emotion,
        "voice_emotion": voice_emotion,
        "final_emotion": final_emotion,
        "response": response,
    }


@router.get("/history")
def history():
    """Return the last 20 emotion log entries."""
    if not os.path.exists(EMOTION_LOG_PATH):
        return {"entries": []}
    entries = []
    with open(EMOTION_LOG_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return {"entries": entries[-20:]}


@router.post("/analyze/stream")
async def analyze_stream(
    frame: UploadFile = File(..., description="JPEG/PNG face image"),
    audio: UploadFile = File(..., description="WAV audio clip (16 kHz, mono)"),
    mock_emotion: str | None = None,
):
    """
    Streaming version of /analyze.
    If mock_emotion is provided, skips model inference and uses the mock value.
    """
    # ── Inference (fast, local) ──────────────────────────────
    if mock_emotion and mock_emotion in ["happy", "sad", "stress", "neutral", "fatigue"]:
        face_emotion  = mock_emotion
        voice_emotion = mock_emotion
        final_emotion = mock_emotion
        # Still read the files to clear the buffer
        await frame.read()
        await audio.read()
    else:
        # ── Face ────────────────────────────────────────────
        img_bytes = await frame.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # ── Audio ───────────────────────────────────────────
        audio_bytes = await audio.read()
        
        try:
            face_emotion  = predict_face(image) if image is not None else "neutral"
        except Exception:
            face_emotion = "neutral"

        try:
            voice_emotion = predict_voice(audio_bytes) if audio_bytes else "neutral"
        except Exception:
            voice_emotion = "neutral"

        final_emotion = fuse(face_emotion, voice_emotion)
    
    _log_emotion(face_emotion, voice_emotion, final_emotion)
    prompt = build_prompt(final_emotion)

    def _event_stream():
        # 1 – send emotion metadata immediately (no LLM wait)
        meta = json.dumps({
            "type":  "meta",
            "face":  face_emotion,
            "voice": voice_emotion,
            "final": final_emotion,
        })
        yield meta + "\n"

        # 2 – stream LLM tokens
        for token in stream_ollama(prompt):
            yield json.dumps({"type": "token", "text": token}) + "\n"

        # 3 – signal done
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        _event_stream(),
        media_type="application/x-ndjson",
    )
