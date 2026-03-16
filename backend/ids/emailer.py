from email.message import EmailMessage
import smtplib
from core.settings import settings
from core.logger import get_logger

log = get_logger(__name__)

def send_email_alert(packet: dict):
    if not settings.email_sender or not settings.email_receiver:
        log.warning("Email not configured; skipping send")
        return
    msg = EmailMessage()
    msg["Subject"] = "🚨 CAN-Bus Intrusion Detected"
    msg["From"] = settings.email_sender
    msg["To"] = settings.email_receiver
    msg.set_content(
        f"Timestamp: {packet['ts']}\n"
        f"CAN ID: {packet['can_id']}\n"
        f"Score: {packet['score']}\n"
        f"Label: {packet['pred']}\n"
        f"Data: {packet['data']}\n"
    )
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
        smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_pass)
        smtp.send_message(msg)
        log.info("Alert email sent → %s", settings.email_receiver)
