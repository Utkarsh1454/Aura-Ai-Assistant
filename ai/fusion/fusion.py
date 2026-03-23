# ─────────────────────────────────────────────────────────────
# Multi-modal Emotion Fusion
# Combines face and voice emotion predictions into one result.
# ─────────────────────────────────────────────────────────────

# Priority ranking: higher index → more urgent / overrides others
_PRIORITY = {
    "happy":   0,
    "neutral": 1,
    "fatigue": 2,
    "sad":     3,
    "stress":  4,
}


def fuse(face: str, voice: str) -> str:
    """
    Decide the final emotion from face and voice predictions.

    Fusion rules (in order):
    1. If both agree → return that emotion.
    2. Prioritise the higher-urgency signal (stress > sad > fatigue > …).
    3. If equal priority → prefer voice (often more reliable for mental state).

    Args:
        face:  Emotion label from face model.
        voice: Emotion label from voice model.

    Returns:
        Final emotion label string.
    """
    if face == voice:
        return face

    face_priority  = _PRIORITY.get(face,  0)
    voice_priority = _PRIORITY.get(voice, 0)

    if face_priority >= voice_priority:
        return face
    return voice


# ── Confidence-aware fusion (optional, for future model upgrades) ─
def fuse_with_confidence(
    face: str,  face_conf: float,
    voice: str, voice_conf: float,
    threshold: float = 0.6,
) -> str:
    """
    Weighted fusion when model confidence scores are available.

    Falls back to priority-based fuse() if neither exceeds threshold.
    """
    if face == voice:
        return face
    if voice_conf >= threshold and voice_conf > face_conf:
        return voice
    if face_conf >= threshold and face_conf > voice_conf:
        return face
    return fuse(face, voice)
