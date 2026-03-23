# ─────────────────────────────────────────────────────────────
# MAITRI Backend – FastAPI Entry Point
# ─────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.analyze import router as analyze_router
from backend.routes.health import router as health_router

app = FastAPI(
    title="MAITRI AI",
    description="Mental AI for Total Real-time Intelligence – Offline Wellness Assistant",
    version="1.0.0",
)

# Allow Streamlit and any localhost client to reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analyze_router)
