import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load env variables from backend/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
print(f"Loading env from: {env_path}")
load_dotenv(env_path)

smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
smtp_port = int(os.getenv("SMTP_PORT", 587))
smtp_user = os.getenv("SMTP_USER")
smtp_password = os.getenv("SMTP_PASSWORD")

print(f"SMTP Server: {smtp_server}")
print(f"SMTP Port: {smtp_port}")
print(f"SMTP User: {smtp_user}")
print(f"SMTP Password length: {len(smtp_password) if smtp_password else 0}")

if not smtp_user or not smtp_password:
    print("SMTP credentials are not fully configured in .env!")
    exit(1)

msg = MIMEMultipart("alternative")
msg["Subject"] = "[HealthPolicyLens] SMTP Connection Test"
msg["From"] = f"HealthPolicyLens <{smtp_user}>"
msg["To"] = smtp_user

body = "This is a test email to verify that HealthPolicyLens SMTP configurations are fully working."
msg.attach(MIMEText(body, "plain"))

try:
    print("Connecting to SMTP server...")
    with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
        print("Starting TLS...")
        server.starttls()
        print("Logging in...")
        server.login(smtp_user, smtp_password)
        print("Sending mail...")
        server.sendmail(smtp_user, [smtp_user], msg.as_string())
    print("SMTP verification successful! Email sent to yourself.")
except Exception as e:
    print(f"SMTP verification failed: {e}")
