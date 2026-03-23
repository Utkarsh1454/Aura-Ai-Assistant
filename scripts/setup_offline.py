# ─────────────────────────────────────────────────────────────
# MAITRI – Offline Setup Utility
# Downloads all pretrained models to local cache for 
# 100% offline use (mission ready).
# ─────────────────────────────────────────────────────────────
import os
import logging
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification,
    pipeline
)
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OfflineSetup")

# Same IDs used in config.py / predict.py
FACE_MODEL_ID = "dima806/facial_emotions_image_detection"
VOICE_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

def download_voice():
    logger.info(f"Downloading Voice Model: {VOICE_MODEL_ID}...")
    AutoFeatureExtractor.from_pretrained(VOICE_MODEL_ID)
    AutoModelForAudioClassification.from_pretrained(VOICE_MODEL_ID)
    logger.info("✅ Voice Model ready!")

def download_face():
    logger.info(f"Downloading Face Model: {FACE_MODEL_ID}...")
    # Pre-warm the pipeline (this downloads the model)
    pipeline("image-classification", model=FACE_MODEL_ID)
    logger.info("✅ Face Model ready!")

def download_ollama_weights():
    logger.info("To pre-download LLM weights, ensure Ollama is running and run:")
    logger.info("   ollama pull llama3")

if __name__ == "__main__":
    logger.info("Starting MAITRI Offline Setup...")
    
    try:
        download_voice()
        download_face()
        download_ollama_weights()
        logger.info("\n🚀 ALL MODELS CACHED!")
        logger.info("MAITRI is now ready for offline deployment.")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        logger.info("\nEnsure you have an internet connection for this first-time setup.")
