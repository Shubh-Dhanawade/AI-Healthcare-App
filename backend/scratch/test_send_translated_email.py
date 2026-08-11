import asyncio
import os
import sys
from sqlalchemy.orm import selectinload
from sqlalchemy import select
from dotenv import load_dotenv

# Ensure backend folder is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env variables from backend/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.models.user import User
from app.api.v1.documents import generate_html_report

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

async def test_email_send(lang: str):
    print(f"\n--- Testing Email Sending for Language: {lang} ---")
    async with AsyncSessionLocal() as db:
        # Fetch the latest document
        stmt = (
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(1)
            .options(
                selectinload(Document.summary),
                selectinload(Document.extracted_fields),
                selectinload(Document.risk_analyses),
            )
        )
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
        if not doc:
            print("No documents found in database.")
            return
            
        # Fetch a user to act as sender
        stmt_user = select(User).limit(1)
        res_user = await db.execute(stmt_user)
        current_user = res_user.scalar_one_or_none()
        
        if not current_user:
            print("No user found in database.")
            return

        # Explicitly load relationships
        await db.refresh(doc, ["summary", "extracted_fields", "risk_analyses"])
        
        print("Generating translated HTML report...")
        html_content = await generate_html_report(doc, lang)
        
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if not smtp_user or not smtp_password:
            print(f"SMTP credentials are not configured! user: {smtp_user}, pass length: {len(smtp_password) if smtp_password else 0}")
            return
            
        user_sender = current_user.email
        user_name = current_user.full_name or "User"
        sender_email = smtp_user
        
        lang_lower = lang.lower()
        if lang_lower == "hindi":
            subject = f"[HealthPolicyLens] नीति विश्लेषण ऑडिट रिपोर्ट: {doc.original_filename}"
            text_fallback = f"प्रिय उपयोगकर्ता,\n\nकृपया {doc.original_filename} के लिए HealthPolicyLens नीति विश्लेषण रिपोर्ट संलग्न पाएं।"
        elif lang_lower == "marathi":
            subject = f"[HealthPolicyLens] पॉलिसी विश्लेषण ऑडिट अहवाल: {doc.original_filename}"
            text_fallback = f"प्रिय वापरकर्ता,\n\nकृपया {doc.original_filename} साठी HealthPolicyLens पॉलिसी विश्लेषण अहवाल सोबत जोडलेला पहा."
        else:
            subject = f"[HealthPolicyLens] Policy Analysis Audit Report: {doc.original_filename}"
            text_fallback = f"Dear User,\n\nPlease find attached the HealthPolicyLens policy analysis report for {doc.original_filename}."

        msg = MIMEMultipart("alternative")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = f"{user_name} via HealthPolicyLens <{sender_email}>"
        msg["To"] = smtp_user  # Send to ourselves for verification
        msg["Reply-To"] = user_sender
        
        part1 = MIMEText(text_fallback, "plain", "utf-8")
        part2 = MIMEText(html_content, "html", "utf-8")
        msg.attach(part1)
        msg.attach(part2)
        
        print(f"Connecting to SMTP server {smtp_server}:{smtp_port}...")
        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [smtp_user], msg.as_string())
            print(f"SUCCESS: Email in {lang} sent to {smtp_user}.")
        except Exception as e:
            print(f"FAILED: Failed to send email via SMTP: {e}")

async def main():
    await test_email_send("Hindi")
    await test_email_send("Marathi")

if __name__ == "__main__":
    asyncio.run(main())
