# ─────────────────────────────────────────────────────────────
# Face Emotion Prediction (Authentic / Pretrained)
# Uses a pretrained MobileNet-v2 fine-tuned on FER-2013.
# ─────────────────────────────────────────────────────────────
import os
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from transformers import pipeline

from backend.config import EMOTION_LABELS

logger = logging.getLogger(__name__)

# --- Model Config ---
# High-accuracy MobileNet-V2 model for facial emotion recognition
MODEL_ID = "dima806/facial_emotions_image_detection"
_DEVICE = 0 if torch.cuda.is_available() else -1 # 0 for GPU, -1 for CPU

_classifier = None

# Mapping from external model labels to MAITRI EMOTION_LABELS
# dima806 model labels: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
_LABEL_MAPPING = {
    "angry": "stress",
    "fear": "stress",
    "disgust": "stress",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "surprise": "fatigue", # best fit
}

def _load_model():
    global _classifier
    if _classifier is None:
        logger.info(f"Loading pretrained face model: {MODEL_ID}...")
        _classifier = pipeline(
            "image-classification", 
            model=MODEL_ID, 
            device=_DEVICE
        )
    return _classifier

def predict_face(frame: np.ndarray) -> str:
    """
    Predict facial emotion from a BGR frame (OpenCV format)
    using an authentic pretrained vision model.
    """
    try:
        classifier = _load_model()

        # 1. Convert BGR to RGB and PIL Image (required by transformers pipeline)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 2. Inference
        results = classifier(pil_img)
        
        # 3. Process top result
        # results format: [{'label': 'happy', 'score': 0.99}, ...]
        top_result = results[0]
        ext_label = top_result['label'].lower()
        score     = top_result['score']
        
        final_label = _LABEL_MAPPING.get(ext_label, "neutral")
        
        logger.info(f"Face Inference: {ext_label} ({score:.2f}) -> {final_label}")
        return final_label

    except Exception as exc:
        logger.error(f"predict_face (pretrained) failed: {exc}")
        return "neutral"
