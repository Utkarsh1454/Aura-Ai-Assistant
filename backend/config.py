# Aura AI Config
APP_NAME = "Aura AI"
APP_TAGLINE = "Mind-Body Connection Assistant"

# Ollama
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_HOST = "http://localhost:11434"   # default Ollama API endpoint

# AI Models (HuggingFace Hub)
FACE_MODEL_ID = "dima806/facial_emotions_image_detection"
VOICE_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Emotion labels (same order as training)
EMOTION_LABELS = ["happy", "sad", "stress", "neutral", "fatigue"]

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Data
EMOTION_LOG_PATH = "data/emotion_log.json"
