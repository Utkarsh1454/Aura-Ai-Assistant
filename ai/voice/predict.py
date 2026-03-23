# ─────────────────────────────────────────────────────────────
# Voice Emotion Prediction (Authentic / Pretrained)
# Uses Wav2Vec2-base-960h fine-tuned for emotion detection.
# ─────────────────────────────────────────────────────────────
import os
import torch
import logging
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from utils.audio_utils import normalize_audio, trim_silence
from backend.config import EMOTION_LABELS

logger = logging.getLogger(__name__)

# --- Model Config ---
# We use a robust XLS-R based model for multilingual/English emotion support
MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_processor = None
_model = None

# Mapping from external model labels to MAITRI EMOTION_LABELS
# ehcalabres model labels: ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
_LABEL_MAPPING = {
    "angry": "stress",
    "fearful": "stress",
    "disgust": "stress",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "calm": "neutral",
    "surprised": "fatigue", # best fit for high-arousal outlier
}

def _load_model():
    global _processor, _model
    if _model is None:
        logger.info(f"Loading pretrained voice model: {MODEL_ID}...")
        _processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(_DEVICE)
        _model.eval()
    return _processor, _model

def predict_voice(audio_bytes: bytes) -> str:
    """
    Predict voice emotion using an authentic pretrained Wav2Vec2 model.
    """
    try:
        extractor, model = _load_model()
        
        # 1. Decode and basic cleaning (normalization + trim)
        from ai.voice.predict import _decode_audio 
        y = _decode_audio(audio_bytes)
        y = trim_silence(y, sr=16000)
        y = normalize_audio(y)

        # 2. Extract features
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

        # 3. Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
            
        # 4. Map to MAITRI labels
        labels = model.config.id2label
        ext_label = labels[pred_id].lower()
        final_label = _LABEL_MAPPING.get(ext_label, "neutral")
        
        logger.info(f"Voice Inference: {ext_label} -> {final_label}")
        return final_label

    except Exception as exc:
        logger.error(f"predict_voice (pretrained) failed: {exc}")
        return "neutral"

# --- Re-export decoder from previous version if needed ---
import io
import librosa
import tempfile

def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Fallback decoder (copy-pasted from previous version for self-containment)"""
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        return y
    except Exception:
        pass
    try:
        # Simple magic byte check
        if audio_bytes[:4] == b"\x1a\x45\xdf\xa3": # WebM
            suffix = ".webm"
        else:
            suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        os.unlink(tmp_path)
        return y
    except Exception:
        return np.zeros(16000, dtype=np.float32)
