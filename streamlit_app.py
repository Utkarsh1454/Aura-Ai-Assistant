# ─────────────────────────────────────────────────────────────
# ✨ Aura AI – Mind-Body Connection Assistant (SINGLE-FILE CLOUD ENTRYPOINT)
# ─────────────────────────────────────────────────────────────
import streamlit as st

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Aura AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
import os
import sys
import requests
from datetime import datetime

# ── Configuration & Metadata ─────────────────────────────────
APP_NAME = "Aura AI"
APP_TAGLINE = "Mind-Body Connection Assistant"

# Check for environment variable or Streamlit secret for public deployment
try:
    API_BASE = st.secrets.get("API_BASE", os.environ.get("ST_API_BASE", "http://localhost:8000"))
except:
    API_BASE = os.environ.get("ST_API_BASE", "http://localhost:8000")

# ── Imports ──────────────────────────────────────────────────
# Using try-except for the recorder as it can be tricky on some Cloud OS
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    def audio_recorder(): return None

# ─────────────────────────────────────────────────────────────
# 🌌 ULTRA MODERN UI CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* Header */
.aura-header {
    text-align: center;
    padding: 2rem 0;
}
.aura-title {
    font-size: 3.5rem;
    font-weight: 600;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.aura-tagline {
    color: #94a3b8;
    letter-spacing: 2px;
    font-size: 0.85rem;
}

/* Glass Cards */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: 0.3s ease;
}
.glass:hover {
    transform: translateY(-4px);
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    border-radius: 12px;
    height: 3.5rem;
    color: white;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}
.stButton button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px #6366f1;
}

/* Emotion Pills */
.emotion-pill {
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.happy { background:#22c55e; color:white;}
.sad { background:#64748b; color:white;}
.stress { background:#ef4444; color:white;}
.neutral { background:#94a3b8; color:white;}
.fatigue { background:#0ea5e9; color:white;}

/* Response Box */
.response-box {
    background: rgba(255,255,255,0.03);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #6366f1;
    line-height: 1.7;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown(f"""
<div class="aura-header">
    <div class="aura-title">{APP_NAME}</div>
    <div class="aura-tagline">{APP_TAGLINE}</div>
</div>
""", unsafe_allow_html=True)

# ── Backend Check & Sandbox Fallsback ────────────────────────
@st.cache_data(ttl=60)
def check_backend():
    try:
        return requests.get(f"{API_BASE}/", timeout=1.5).status_code == 200
    except:
        return False

backend_up = check_backend()
if not backend_up:
    st.warning("⚠️ Aura Core Offline. Entering **Demo Sandbox Mode** for preview.")
    st.session_state.sandbox_mode = True
else:
    st.session_state.sandbox_mode = False

# ── Layout ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📸 Presence Input")
    img_input = st.camera_input("Capture")
    if img_input is None:
        img_input = st.file_uploader("Upload Image", type=["jpg","png"])

    st.markdown("### 🎤 Voice")
    audio = audio_recorder() if 'audio_recorder' in globals() else None
    if audio:
        st.audio(audio)

    btn = st.button("Analyze", disabled=not(img_input and audio))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🔮 Aura Insight")

    if btn:
        status_msg = st.info("🧬 Calibrating Aura...")
        img_bytes = img_input.getvalue() if hasattr(img_input, "getvalue") else img_input.read()
        
        if st.session_state.get("sandbox_mode"):
            import time
            time.sleep(1.5)
            status_msg.empty()
            sim_e = st.session_state.get("mock_choice", "happy")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Visual", sim_e.capitalize())
            c2.metric("Vocal", sim_e.capitalize())
            c3.metric("State", sim_e.capitalize())
            
            st.markdown(f'<div style="margin:1rem 0"><span class="emotion-pill {sim_e}">{sim_e.upper()} (SANDBOX)</span></div>', unsafe_allow_html=True)
            
            res_box = st.empty()
            full_res = f"The Aura Field perceives a {sim_e} resonance. Cloud Sandbox active."
            res_box.markdown(f'<div class="response-box">{full_res}</div>', unsafe_allow_html=True)
            
            if "history" not in st.session_state: st.session_state.history = []
            st.session_state.history.append({
                "face": sim_e, "voice": sim_e, "final": sim_e, 
                "response": full_res, "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.stop()

        files = {"frame": ("frame.jpg", img_bytes, "image/jpeg"), "audio": ("audio.wav", audio, "audio.wav")}
        params = {"mock_emotion": st.session_state.get("mock_choice")} if st.session_state.get("mock_enable") else {}

        try:
            with requests.post(f"{API_BASE}/api/analyze/stream", params=params, files=files, stream=True, timeout=90) as resp:
                resp.raise_for_status()
                status_msg.empty()
                metrics_ph, badge_ph, response_ph = st.empty(), st.empty(), st.empty()
                face_e, voice_e, final_e, response_text = "...", "...", "...", ""

                for raw in resp.iter_lines():
                    if not raw: continue
                    chunk = json.loads(raw)
                    if chunk.get("type") == "meta":
                        face_e, voice_e, final_e = chunk.get("face"), chunk.get("voice"), chunk.get("final")
                        with metrics_ph.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Visual", face_e.capitalize())
                            m2.metric("Vocal", voice_e.capitalize())
                            m3.metric("State", final_e.capitalize())
                        badge_ph.markdown(f'<div style="margin:1rem 0"><span class="emotion-pill {final_e}">{final_e.upper()}</span></div>', unsafe_allow_html=True)
                    elif chunk.get("type") == "token":
                        response_text += chunk.get("text", "")
                        response_ph.markdown(f'<div class="response-box">{response_text}</div>', unsafe_allow_html=True)

                if "history" not in st.session_state: st.session_state.history = []
                st.session_state.history.append({"face": face_e, "voice": voice_e, "final": final_e, "response": response_text, "timestamp": datetime.now().strftime("%H:%M:%S")})
        except Exception as e:
            st.error(f"Insight calibration failed: {e}")
    else:
        st.info("Align your visual and vocal presence to begin.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Controls")
    st.toggle("Simulation Mode", key="mock_enable")
    st.selectbox("Emotion", ["happy","sad","stress","neutral","fatigue"], key="mock_choice")
    st.markdown("---")
    st.subheader("📜 Journey History")
    if "history" in st.session_state and st.session_state.history:
        for entry in reversed(st.session_state.history[-5:]):
            with st.expander(f"{entry['timestamp']} — {entry['final'].capitalize()}"):
                st.caption(f"Visual: {entry['face']} | Vocal: {entry['voice']}")
                st.write(entry['response'])
    else:
        st.caption("No entries found yet.")
