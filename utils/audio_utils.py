import io
import wave
import librosa
import numpy as np


def wav_bytes_to_numpy(audio_bytes: bytes, expected_sr: int = 16_000) -> np.ndarray:
    """
    Convert raw WAV bytes to a float32 numpy array.

    Args:
        audio_bytes:  Raw bytes of a WAV file.
        expected_sr:  Expected sample rate (for logging only).

    Returns:
        1D float32 numpy array normalised to [-1, 1].
    """
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw         = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    # Convert stereo → mono
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    audio /= 32768.0  # normalise to [-1, 1]

    if sample_rate != expected_sr:
        print(
            f"[audio_utils] Warning: sample rate {sample_rate} Hz, "
            f"expected {expected_sr} Hz. Consider resampling."
        )

    return audio


def audio_duration(audio_bytes: bytes) -> float:
    """Return duration of a WAV clip in seconds."""
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def is_silent(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True if the audio clip is effectively silent (no speech)."""
    return float(np.abs(audio).mean()) < threshold


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalise audio to [-1, 1]."""
    peak = np.abs(audio).max()
    if peak > 0:
        return audio / peak
    return audio


def trim_silence(audio: np.ndarray, sr: int = 16000, top_db: int = 20) -> np.ndarray:
    """Remove leading and trailing silence from an audio signal."""
    yt, _ = librosa.effects.trim(audio, top_db=top_db)
    return yt

