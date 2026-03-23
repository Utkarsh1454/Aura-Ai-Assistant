# ─────────────────────────────────────────────────────────────
# Route: /  – Health check
# ─────────────────────────────────────────────────────────────
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/")
def health():
    return {"status": "MAITRI running 🚀", "version": "1.0.0"}


@router.get("/health")
def health_detail():
    return {
        "status": "ok",
        "services": {
            "backend": "running",
            "description": "FastAPI multimodal emotion analysis",
        },
    }
