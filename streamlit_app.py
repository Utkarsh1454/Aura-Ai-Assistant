# ─────────────────────────────────────────────────────────────
# Aura AI – Streamlit Cloud Entrypoint
# ─────────────────────────────────────────────────────────────
import os
import sys

# Ensure the root directory is in the path for internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Execute the main frontend application
from frontend.app import st

# Use the existing frontend logic
if __name__ == "__main__":
    pass # Frontend logic is already invoked on import in 'app.py'
