# ─────────────────────────────────────────────────────────────
# Ollama Client – Offline LLM Integration
# Uses the Ollama REST API with streaming for low-latency.
# ─────────────────────────────────────────────────────────────
import json
import logging
from typing import Generator, Optional

import requests

from backend.config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)

_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"

# ── Speed tuning ─────────────────────────────────────────────
# num_predict caps output tokens → stops after ~80 tokens (2-3 sentences)
# num_ctx shrinks context window  → faster prefill phase
_OLLAMA_OPTIONS = {
    "num_predict": 80,
    "num_ctx":     512,
    "temperature": 0.7,
    "top_p":       0.9,
    "stop":        ["\n\n"],
}


def query_ollama(prompt: str, model: Optional[str] = None) -> str:
    """
    Blocking call to Ollama. Joins streaming tokens into a full string.
    Using stream=True under the hood avoids one long HTTP wait.
    """
    return "".join(stream_ollama(prompt, model))


def stream_ollama(
    prompt: str, model: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Yield response tokens one by one as the LLM generates them.
    Use in FastAPI StreamingResponse or Streamlit st.write_stream.
    """
    payload = {
        "model":   model or OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  True,
        "options": _OLLAMA_OPTIONS,
    }

    try:
        with requests.post(
            _GENERATE_URL, json=payload, stream=True, timeout=60
        ) as resp:
            if resp.status_code != 200:
                error_body = resp.text
                logger.error(f"Ollama error {resp.status_code}: {error_body}")
                yield f"AI Engine Error ({resp.status_code}). Ensure '{OLLAMA_MODEL}' is pulled: `ollama pull {OLLAMA_MODEL}`."
                return

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError:
        logger.error("Ollama is not running. Start with: ollama serve")
        yield (
            "I cannot reach the AI engine right now. "
            "Please run `ollama serve` and try again."
        )
    except requests.exceptions.Timeout:
        logger.error("Ollama timed out.")
        yield "The AI took too long to respond. Please try again."
    except Exception as exc:
        logger.exception("Unexpected Ollama error: %s", exc)
        yield f"Error: {exc}"
