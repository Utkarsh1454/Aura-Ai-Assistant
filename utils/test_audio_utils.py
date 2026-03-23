import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.audio_utils import normalize_audio, trim_silence

def test_normalize():
    print("Testing normalize_audio...")
    # Low volume signal
    audio = np.array([0.1, 0.2, -0.1], dtype=np.float32)
    normalized = normalize_audio(audio)
    assert np.isclose(np.abs(normalized).max(), 1.0), f"Peak should be 1.0, got {np.abs(normalized).max()}"
    print("[OK] normalize_audio passed")

def test_trim_silence():
    print("Testing trim_silence...")
    sr = 16000
    # 1s silence, 1s signal, 1s silence
    silence = np.zeros(sr, dtype=np.float32)
    signal = np.ones(sr, dtype=np.float32)
    audio = np.concatenate([silence, signal, silence])
    
    trimmed = trim_silence(audio, sr=sr)
    
    orig_len = len(audio)
    trimmed_len = len(trimmed)
    print(f"Original length: {orig_len}, Trimmed length: {trimmed_len}")
    
    assert trimmed_len < orig_len, "Trimmed length should be less than original"
    # Allow more tolerance due to librosa's windowing (1s to 1.5s is fine)
    assert 15000 < trimmed_len < 24000, f"Expected between 15000-24000 samples, got {trimmed_len}"
    print("[OK] trim_silence passed")

if __name__ == "__main__":
    try:
        test_normalize()
        test_trim_silence()
        print("\n[SUCCESS] All audio utility tests PASSED!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        sys.exit(1)
