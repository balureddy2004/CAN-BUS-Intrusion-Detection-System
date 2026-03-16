from dataclasses import dataclass
import os, glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Settings:
    host: str = os.getenv("FLASK_HOST", "0.0.0.0")
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    email_sender: str = os.getenv("EMAIL_SENDER", "")
    email_receiver: str = os.getenv("EMAIL_RECEIVER", "")
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_pass: str = os.getenv("SMTP_PASS", "")
    alert_threshold: float = float(os.getenv("ALERT_THRESHOLD", "0.9"))
    current_model: str = None

settings = Settings()

# 🔍 Auto-detect model file if not set
if not settings.current_model or settings.current_model == "N/A":
    model_files = glob.glob("models/*.pkl")
    if model_files:
        # ✅ Store only the filename (e.g. "model.pkl")
        settings.current_model = os.path.basename(model_files[0])
    else:
        settings.current_model = None

print(f"⚙️ Using model: {settings.current_model or 'No model found'}")
