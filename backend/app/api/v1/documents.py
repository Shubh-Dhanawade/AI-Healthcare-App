"""Document upload and management API endpoints."""

import os
import uuid
from pathlib import Path as FilePath
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from loguru import logger

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.document import Document, ExtractedField
from app.models.risk_analysis import Summary, RiskAnalysis
from app.models.reminder import PolicyReminder
from app.schemas.schemas import DocumentResponse, DocumentDetailResponse, CompareRequest, CompareResponse, ComparisonSynthesisSchema
from app.services.ocr_service import extract_document_text
from app.services.ai_service import generate_comparison_synthesis
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# In-progress tracker — prevents duplicate background analysis jobs
# ─────────────────────────────────────────────────────────────────────────────
_analysis_in_progress: set = set()


def _parse_date_string(date_str: str) -> Optional[datetime]:
    import re
    if not date_str:
        return None
    date_str = date_str.strip()
    
    # 1. DD/MM/YYYY or DD-MM-YYYY
    m1 = re.search(r'(\d{1,2})[\-/](\d{1,2})[\-/](\d{4})', date_str)
    if m1:
        d, m, y = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        # Handle cases where year is first (YYYY-MM-DD)
        if d > 1900:
            y, m, d = d, m, int(m1.group(3))
        try:
            return datetime(y, m, d)
        except ValueError:
            pass

    # 2. YYYY-MM-DD
    m2 = re.search(r'(\d{4})[\-/](\d{1,2})[\-/](\d{1,2})', date_str)
    if m2:
        y, m, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        try:
            return datetime(y, m, d)
        except ValueError:
            pass

    # 3. DD-MMM-YYYY (e.g. 15-Jul-2024 or 15 Jul 2024)
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    m3 = re.search(r'(\d{1,2})\s*[\-\s/]?\s*([A-Za-z]{3})[a-z]*\s*[\-\s/]?\s*(\d{4})', date_str, re.IGNORECASE)
    if m3:
        d = int(m3.group(1))
        mon_str = m3.group(2).lower()
        y = int(m3.group(3))
        if mon_str in months:
            m = months[mon_str]
            try:
                return datetime(y, m, d)
            except ValueError:
                pass
    return None


def _parse_premium_amount(prem_str: str) -> Optional[float]:
    import re
    if not prem_str:
        return None
    # Remove currency symbols, commas, and spaces
    cleaned = re.sub(r'[^\d.]', '', prem_str)
    try:
        return float(cleaned)
    except ValueError:
        return None


async def _auto_schedule_policy_alerts(db, doc, fields_data) -> None:
    from sqlalchemy import delete
    from app.models.reminder import PolicyReminder
    from app.models.document import ExtractedField
    from datetime import datetime, timedelta
    
    extracted_renewal = None
    extracted_premium_due = None
    extracted_premium_val = None

    for field in fields_data:
        name_lower = field["field_name"].lower()
        val = str(field["field_value"]).strip()
        if not val or val.lower() in ("not found in document", "not specified", "null", "none"):
            continue
        if any(x in name_lower for x in ["renewal date", "expiry date", "valid to", "policy end date", "period of insurance to", "policy term"]):
            parsed_d = _parse_date_string(val)
            if parsed_d:
                extracted_renewal = parsed_d
        if any(x in name_lower for x in ["premium due", "payment due"]):
            parsed_pd = _parse_date_string(val)
            if parsed_pd:
                extracted_premium_due = parsed_pd
        if any(x in name_lower for x in ["premium amount", "gross premium", "net premium", "total premium", "premium"]):
            if val and val.lower() not in ("not mentioned in policy", "not specified"):
                extracted_premium_val = val

    # Direct document text fallback if dates or premium amounts were missed
    if (not extracted_renewal or not extracted_premium_val) and doc.extracted_text:
        import re
        if not extracted_renewal:
            # Find date range like 05-05-2026 to 04-05-2029 or 04-05-2029
            date_matches = re.findall(r'([0-3]?\d[\-/][0-1]?\d[\-/]\d{4})', doc.extracted_text)
            if len(date_matches) >= 2:
                candidate_d = _parse_date_string(date_matches[1])
                if candidate_d:
                    extracted_renewal = candidate_d
            elif len(date_matches) == 1:
                candidate_d = _parse_date_string(date_matches[0])
                if candidate_d:
                    extracted_renewal = candidate_d

        if not extracted_premium_val:
            prem_match = re.search(
                r'[\u20b9Rs.INR]*\s*([\d,]{4,}(?:\.\d{1,2})?)\s*(?:towards\s+premium|towards\s+the\s+premium|towards\s+insurance|premium)',
                doc.extracted_text, re.IGNORECASE
            )
            if not prem_match:
                prem_match = re.search(
                    r'(?:received\s+an\s+amount\s+of|premium\s+paid|total\s+premium)[:\s\u20b9Rs.INR]*\s*([\d,]{4,}(?:\.\d{1,2})?)',
                    doc.extracted_text, re.IGNORECASE
                )
            if prem_match:
                extracted_premium_val = f"₹{prem_match.group(1).strip()}"

    # Coerce extracted datetimes to naive to match db layout
    if extracted_renewal:
        extracted_renewal = extracted_renewal.replace(tzinfo=None)
    if extracted_premium_due:
        extracted_premium_due = extracted_premium_due.replace(tzinfo=None)

    # Check and assign defaults if missing
    has_renewal = (extracted_renewal is not None)
    has_premium_due = (extracted_premium_due is not None)
    has_premium_val = (extracted_premium_val is not None)

    if not has_renewal:
        # Default to 1 year from now
        extracted_renewal = datetime.utcnow() + timedelta(days=365)
        db.add(ExtractedField(
            document_id=doc.id,
            field_name="Renewal Date",
            field_value=f"{extracted_renewal.strftime('%Y-%m-%d')} (Not Mentioned, Defaulted)",
            field_category="policy_period"
        ))
        logger.info(f"[AUTO-ALERT] Renewal date not found in document. Defaulting to 1 year: {extracted_renewal}")
    else:
        # Save actual renewal date to ExtractedFields if not already present
        existing_renewal = any(f["field_name"].lower() == "renewal date" for f in fields_data)
        if not existing_renewal:
            db.add(ExtractedField(
                document_id=doc.id,
                field_name="Renewal Date",
                field_value=extracted_renewal.strftime('%Y-%m-%d'),
                field_category="policy_period"
            ))

    if not has_premium_due:
        # Default to 11 months from now
        extracted_premium_due = datetime.utcnow() + timedelta(days=330)
        db.add(ExtractedField(
            document_id=doc.id,
            field_name="Premium Due Date",
            field_value=f"{extracted_premium_due.strftime('%Y-%m-%d')} (Not Mentioned, Defaulted)",
            field_category="premium"
        ))
        logger.info(f"[AUTO-ALERT] Premium due date not found in document. Defaulting to 11 months: {extracted_premium_due}")
    else:
        existing_prem_due = any(f["field_name"].lower() == "premium due date" for f in fields_data)
        if not existing_prem_due:
            db.add(ExtractedField(
                document_id=doc.id,
                field_name="Premium Due Date",
                field_value=extracted_premium_due.strftime('%Y-%m-%d'),
                field_category="premium"
            ))

    if not has_premium_val:
        extracted_premium_val = "Not Mentioned in Policy"
        db.add(ExtractedField(
            document_id=doc.id,
            field_name="Premium Amount",
            field_value="Not Mentioned in Policy",
            field_category="premium"
        ))
        logger.info(f"[AUTO-ALERT] Premium amount not found in document. Defaulting placeholder indicator.")
    else:
        existing_prem_val = any(f["field_name"].lower() == "premium amount" for f in fields_data)
        if not existing_prem_val:
            db.add(ExtractedField(
                document_id=doc.id,
                field_name="Premium Amount",
                field_value=str(extracted_premium_val),
                field_category="premium"
            ))

    target_prem_due = extracted_premium_due or extracted_renewal

    doc.renewal_date = extracted_renewal
    
    # Clear existing renewal alerts
    await db.execute(delete(PolicyReminder).where(
        PolicyReminder.document_id == doc.id,
        PolicyReminder.reminder_type == "renewal"
    ))
    
    # Create renewal reminder (7 days prior)
    trigger_date = extracted_renewal - timedelta(days=7)
    r1 = PolicyReminder(
        user_id=doc.user_id,
        document_id=doc.id,
        title=f"Policy Renewal Approaching: {doc.original_filename}",
        reminder_type="renewal",
        reminder_date=trigger_date,
        is_dismissed=False
    )
    db.add(r1)
    logger.info(f"[AUTO-ALERT] Scheduled renewal reminder for doc {doc.id} on {trigger_date}")

    doc.premium_due_date = target_prem_due
    
    # Clear existing premium alerts
    await db.execute(delete(PolicyReminder).where(
        PolicyReminder.document_id == doc.id,
        PolicyReminder.reminder_type == "premium"
    ))
    
    # Create premium reminder (5 days prior)
    trigger_date = target_prem_due - timedelta(days=5)
    r2 = PolicyReminder(
        user_id=doc.user_id,
        document_id=doc.id,
        title=f"Premium Payment Approaching: {doc.original_filename}",
        reminder_type="premium",
        reminder_date=trigger_date,
        premium_amount=str(extracted_premium_val),
        is_dismissed=False
    )
    db.add(r2)
    logger.info(f"[AUTO-ALERT] Scheduled premium reminder for doc {doc.id} on {trigger_date}")

    # Trigger email notification to user
    try:
        user = doc.user
        if user:
            send_alert_email_notification(
                user_name=user.full_name or user.email,
                user_email=user.email,
                policy_name=doc.original_filename,
                renewal_date=doc.renewal_date,
                premium_due_date=doc.premium_due_date,
                premium_amount=str(extracted_premium_val)
            )
    except Exception as email_err:
        logger.error(f"Failed to trigger auto email notification for reminder: {email_err}")


def send_alert_email_notification(
    user_name: str,
    user_email: str,
    policy_name: str,
    renewal_date: Optional[datetime],
    premium_due_date: Optional[datetime],
    premium_amount: Optional[str]
) -> None:
    """Send or log a localized policy alert configuration email notification to the registered user."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib
    import os
    import uuid
    from loguru import logger

    # 1. Format dates
    renewal_str = renewal_date.strftime('%Y-%m-%d') if renewal_date else "Not set"
    premium_due_str = premium_due_date.strftime('%Y-%m-%d') if premium_due_date else "Not set"
    premium_val_str = premium_amount if premium_amount else "Not set"

    # 2. Build email body HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #f8fafc;
                margin: 0;
                padding: 30px;
                color: #1e293b;
            }}
            .card {{
                max-width: 580px;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #1e3a8a, #3b82f6);
                padding: 30px 20px;
                text-align: center;
                color: #ffffff;
            }}
            .header h2 {{ margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.025em; }}
            .body {{ padding: 30px 25px; }}
            .greeting {{ font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 12px; }}
            .text {{ font-size: 14px; line-height: 1.6; color: #475569; margin-bottom: 25px; }}
            .details-box {{
                background-color: #f1f5f9;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 25px;
                border: 1px solid #e2e8f0;
            }}
            .detail-row {{
                display: flex;
                justify-content: space-between;
                font-size: 13px;
                padding: 8px 0;
                border-bottom: 1px dashed #cbd5e1;
            }}
            .detail-row:last-child {{ border-bottom: none; }}
            .detail-label {{ color: #64748b; font-weight: 500; }}
            .detail-value {{ color: #0f172a; font-weight: 600; }}
            .footer {{
                background-color: #f8fafc;
                padding: 20px;
                text-align: center;
                font-size: 11px;
                color: #94a3b8;
                border-top: 1px solid #e2e8f0;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <h2>Policy Reminder Configured ⏰</h2>
            </div>
            <div class="body">
                <div class="greeting">Dear {user_name},</div>
                <div class="text">
                    This is an automated notification to confirm that smart renewal and premium alerts have been successfully set up for your insurance policy. The system will monitor deadlines and alert you prior to the due dates.
                </div>
                <div class="details-box">
                    <div class="detail-row">
                        <span class="detail-label">Policy Name:</span>
                        <span class="detail-value">{policy_name}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Renewal Date:</span>
                        <span class="detail-value">{renewal_str}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Premium Due Date:</span>
                        <span class="detail-value">{premium_due_str}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Premium Amount:</span>
                        <span class="detail-value">{premium_val_str}</span>
                    </div>
                </div>
                <div class="text" style="font-size: 12px; color: #64748b; background-color: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; border-radius: 8px;">
                    ℹ️ <strong>Reminder triggers:</strong> Renewal notifications will trigger 7 days prior, and premium notifications will trigger 5 days prior.
                </div>
            </div>
            <div class="footer">
                This is an auto-generated notification. Please do not reply to this email.<br>
                &copy; HealthPolicyLens Corp.
            </div>
        </div>
    </body>
    </html>
    """

    # 3. Setup email structure
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[HealthPolicyLens] Active Alert Configured: {policy_name}"
    msg["From"] = "noreply@healthpolicylens.local"
    msg["To"] = user_email
    
    text_fallback = (
        f"Dear {user_name},\n\n"
        f"Your policy alert has been configured for {policy_name}.\n"
        f"Renewal Date: {renewal_str}\n"
        f"Premium Due Date: {premium_due_str}\n"
        f"Premium Amount: {premium_val_str}\n\n"
        f"Best regards,\nHealthPolicyLens Team"
    )
    part1 = MIMEText(text_fallback, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)
    
    # 4. Try sending SMTP or write to local debug folder
    sent_successfully = False
    try:
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if smtp_server and smtp_user and smtp_password:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(msg["From"], msg["To"], msg.as_string())
            sent_successfully = True
            logger.info(f"📧 Notification email successfully sent via SMTP to {user_email}")
    except Exception as e:
        logger.error(f"Failed to send alert notification email via SMTP: {e}")
        
    if not sent_successfully:
        debug_dir = "./logs/sent_emails"
        os.makedirs(debug_dir, exist_ok=True)
        debug_filepath = f"{debug_dir}/alert_notif_{uuid.uuid4().hex[:6]}.html"
        try:
            with open(debug_filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"💾 Logged outgoing alert email locally to: {debug_filepath}")
        except Exception as io_err:
            logger.error(f"Failed to write alert email debug log: {io_err}")


async def _run_fields_background(doc_id: str, force_regenerate: bool = False) -> None:
    """Server-side asyncio task: Extract Fields only. Launched via create_task()."""
    from app.core.database import AsyncSessionLocal
    from app.services.ai_service import extract_policy_fields

    tracker_key = f"fields:{doc_id}"
    logger.info(f"[BG-FIELDS] Starting field extraction for {doc_id} (force={force_regenerate})")

    # 1. Fetch extracted text in a quick read-only block to release locks immediately
    extracted_text = None
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc and doc.extracted_text:
            extracted_text = doc.extracted_text

    if not extracted_text:
        logger.warning(f"[BG-FIELDS] Document {doc_id} not found or has no text. Aborting.")
        _analysis_in_progress.discard(tracker_key)
        return

    try:
        # 2. Call Ollama FIRST (takes 10-30s, NO database session/lock is held!)
        fields_data = await extract_policy_fields(extracted_text, force_regenerate=force_regenerate)

        # 3. Save to database in a new quick write transaction
        async with AsyncSessionLocal() as db:
            # Clear old fields first
            existing = await db.execute(select(ExtractedField).where(ExtractedField.document_id == doc_id))
            for f in existing.scalars().all():
                await db.delete(f)
            await db.flush()

            for field in fields_data:
                db.add(ExtractedField(
                    document_id=doc_id,
                    field_name=field["field_name"],
                    field_value=field["field_value"],
                    field_category=field.get("field_category"),
                ))

            # Update document status & auto alerts
            result = await db.execute(
                select(Document)
                .options(selectinload(Document.user))
                .where(Document.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if doc:
                if doc.status not in ("completed",):
                    doc.status = "completed"
                
                # Auto schedule alerts based on extracted fields
                await _auto_schedule_policy_alerts(db, doc, fields_data)

            await db.commit()
            logger.info(f"[BG-FIELDS] Done for {doc_id} — {len(fields_data)} fields saved")
    except Exception as e:
        logger.error(f"[BG-FIELDS] Failed for {doc_id}: {e}")

    _analysis_in_progress.discard(tracker_key)


async def _run_risks_background(doc_id: str, force_regenerate: bool = False) -> None:
    """Server-side asyncio task: Risk Analysis only. Launched via create_task()."""
    from app.core.database import AsyncSessionLocal
    from app.services.ai_service import analyze_risks

    tracker_key = f"risks:{doc_id}"
    logger.info(f"[BG-RISKS] Starting risk analysis for {doc_id} (force={force_regenerate})")

    # 1. Fetch extracted text in a quick read-only block to release locks immediately
    extracted_text = None
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc and doc.extracted_text:
            extracted_text = doc.extracted_text

    if not extracted_text:
        logger.warning(f"[BG-RISKS] Document {doc_id} not found or has no text. Aborting.")
        _analysis_in_progress.discard(tracker_key)
        return

    try:
        # 2. Call Ollama FIRST (takes 10-30s, NO database session/lock is held!)
        risk_data = await analyze_risks(extracted_text, force_regenerate=force_regenerate)

        # 3. Save to database in a new quick write transaction
        async with AsyncSessionLocal() as db:
            # Clear old risks first
            existing = await db.execute(select(RiskAnalysis).where(RiskAnalysis.document_id == doc_id))
            for r in existing.scalars().all():
                await db.delete(r)
            await db.flush()

            for risk in risk_data.get("risks", []):
                db.add(RiskAnalysis(
                    document_id=doc_id,
                    clause_text=risk["clause_text"],
                    risk_type=risk["risk_type"],
                    severity=risk.get("severity", "medium"),
                    explanation=risk.get("explanation"),
                    recommendation=risk.get("recommendation"),
                ))

            await db.commit()
            logger.info(f"[BG-RISKS] Done for {doc_id} — {len(risk_data.get('risks', []))} risks saved")
    except Exception as e:
        logger.error(f"[BG-RISKS] Failed for {doc_id}: {e}")

    _analysis_in_progress.discard(tracker_key)


async def _run_summary_background(doc_id: str, force_regenerate: bool = False) -> None:
    """Server-side asyncio task: Summarization only. Launched via create_task()."""
    from app.core.database import AsyncSessionLocal
    from app.services.summary_service import generate_and_store_summary

    tracker_key = f"summary:{doc_id}"
    logger.info(f"[BG-SUMMARY] Starting summarization for {doc_id} (force={force_regenerate})")

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc or not doc.extracted_text:
                logger.warning(f"[BG-SUMMARY] Document {doc_id} not found or has no text. Aborting.")
                return

            extracted_text = doc.extracted_text

            # Call summary service — this calls Ollama then saves to DB
            await generate_and_store_summary(db, doc_id, extracted_text, force_regenerate=force_regenerate)
            await db.commit()
            logger.info(f"[BG-SUMMARY] Done for {doc_id}")
        except Exception as e:
            logger.error(f"[BG-SUMMARY] Failed for {doc_id}: {e}")

    _analysis_in_progress.discard(tracker_key)


async def run_full_analysis_background(doc_id: str) -> None:
    """Combined wrapper: runs summary, fields, and risks CONCURRENTLY for maximum speed."""
    import asyncio
    await asyncio.gather(
        _run_summary_background(doc_id, force_regenerate=False),
        _run_fields_background(doc_id),
        _run_risks_background(doc_id),
    )



# Allowed MIME types
ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/png": "image",
    "image/tiff": "image",
    "image/webp": "image",
}

MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes


async def process_document_background(doc_id: str, file_path: str, file_type: str):
    """Background task to extract text from uploaded document and perform auto-analysis.
    
    Phase 1 (FAST): Extract text → commit to DB → mark as 'text_extracted' so UI unblocks immediately.
    Phase 2 (BACKGROUND): Run chunking+embedding and AI summary CONCURRENTLY, then mark 'completed'.
    """
    from app.core.database import AsyncSessionLocal
    import asyncio
    
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc:
                return
            
            # ── PHASE 1: Fast text extraction (unblocks UI quickly) ──
            doc.status = "processing"
            await db.commit()
            
            text, method, page_count = await extract_document_text(file_path, file_type)
            
            doc.extracted_text = text
            doc.extraction_method = method
            doc.page_count = page_count
            doc.status = "text_extracted"  # ← UI can now display the document
            await db.commit()
            logger.info(f"Phase 1 done for {doc_id}: text extracted ({len(text)} chars), status=text_extracted")

        except Exception as e:
            logger.error(f"Phase 1 text extraction failed for {doc_id}: {e}")
            async with AsyncSessionLocal() as err_db:
                result = await err_db.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = "failed"
                    await err_db.commit()
            return

    # ── PHASE 2: PARALLEL LLM Analysis — all three tasks run concurrently ──
    # Summary, field extraction, and risk analysis are launched simultaneously via asyncio.gather().
    # This reduces total time from ~120s (sequential) to ~60s (time of the slowest single task).
    # The model stays resident in VRAM throughout (keep_alive=-1) so no cold-start between tasks.
    async def _run_summary_task():
        try:
            async with AsyncSessionLocal() as db_sum:
                from app.services.summary_service import generate_and_store_summary
                await generate_and_store_summary(db_sum, doc_id, text)
                await db_sum.commit()
            logger.info(f"⚡ [PARALLEL] Summary complete for {doc_id}")
        except Exception as e:
            logger.error(f"[PARALLEL-SUMMARY] Failed for {doc_id}: {e}")

    try:
        # Run summary, fields, risks, AND vector embedding all at the same time
        await asyncio.gather(
            _run_summary_task(),
            _run_fields_background(doc_id),
            _run_risks_background(doc_id),
            return_exceptions=True,  # Don't let one failure cancel others
        )
        logger.info(f"⚡ All parallel AI tasks complete for {doc_id}")
    except Exception as llm_err:
        logger.error(f"Parallel LLM analysis phase failed for {doc_id}: {llm_err}")

    # ── PHASE 3: Fast Single-Batch Vector Embedding Indexing (Nomic) ──
    async with AsyncSessionLocal() as db_emb:
        try:
            from app.services.rag_service import generate_document_chunks
            await generate_document_chunks(doc_id, text, db_emb)
            await db_emb.commit()
            logger.info(f"⚡ Vector chunking & embedding indexing complete for {doc_id}")
        except Exception as e:
            logger.error(f"Chunking/embedding failed for {doc_id}: {e}")

    # Final status update — mark as completed
    async with AsyncSessionLocal() as db_final:
        try:
            result = await db_final.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "completed"
                await db_final.commit()
            logger.info(f"✅ Document {doc_id} fully processed and marked completed!")
        except Exception as e:
            logger.error(f"Failed to update final status for {doc_id}: {e}")


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a healthcare insurance document (PDF or image)."""
    # Validate file type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Supported: PDF, JPG, PNG, TIFF, WEBP",
        )
    
    file_type = ALLOWED_MIME_TYPES[content_type]
    
    # Read file and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB",
        )
    
    # Calculate SHA-256 hash of the file content
    import hashlib
    file_hash = hashlib.sha256(content).hexdigest()
    
    # Check if a document with this hash has already been uploaded by this user
    result = await db.execute(
        select(Document).where(
            Document.user_id == current_user.id,
            Document.file_hash == file_hash
        )
    )
    existing_doc = result.scalar_one_or_none()
    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This document has already been uploaded. Duplicate uploads of the same report are not allowed.",
        )
    
    # Generate unique filename
    extension = FilePath(file.filename).suffix.lower() or (".pdf" if file_type == "pdf" else ".jpg")
    stored_filename = f"{uuid.uuid4()}{extension}"
    
    # Create user-specific upload directory
    user_upload_dir = FilePath(settings.UPLOAD_DIR) / str(current_user.id)
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = str(user_upload_dir / stored_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(content)
    
    logger.info(f"File saved: {file_path}")
    
    # Create database record
    doc = Document(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_filename=stored_filename,
        file_path=file_path,
        file_type=file_type,
        file_size_bytes=len(content),
        mime_type=content_type,
        status="uploaded",
        file_hash=file_hash,
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)
    
    # Start background text extraction
    background_tasks.add_task(
        process_document_background, doc.id, file_path, file_type
    )
    
    return DocumentResponse.model_validate(doc)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-image upload endpoint
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/tiff", "image/webp"
}


async def process_multi_image_background(doc_id: str, image_paths: list[str]) -> None:
    """
    Background task for multi-image bundle documents.

    Phase 1: Extract text from each image concurrently via OCR,
             concatenate in page order → store in Document.extracted_text.
    Phase 2: Full parallel analysis (chunking + embeddings + summary + fields + risks).
    """
    import asyncio
    from app.core.database import AsyncSessionLocal
    from app.services.ocr_service import extract_document_text, clean_extracted_text

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc:
                return

            doc.status = "processing"
            await db.commit()

            # --- Phase 1: Concurrent OCR across all pages -----------------
            async def _ocr_one(path: str, page_num: int):
                loop = asyncio.get_event_loop()
                from app.services.ocr_service import extract_text_from_image
                raw, method, _ = await loop.run_in_executor(None, extract_text_from_image, path)
                cleaned = clean_extracted_text(raw)
                return page_num, cleaned, method

            tasks = [_ocr_one(p, i + 1) for i, p in enumerate(image_paths)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            page_texts = []
            methods_used = []
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"[MULTI-IMG] OCR failed for one page: {r}")
                    continue
                page_num, text, method = r
                page_texts.append((page_num, text))
                methods_used.append(method)

            # Sort by original page order and concatenate
            page_texts.sort(key=lambda x: x[0])
            combined_text = "\n\n".join(
                f"[Page {pn}]\n{txt}" for pn, txt in page_texts if txt.strip()
            )

            if not combined_text.strip():
                combined_text = (
                    "Could not extract text from the uploaded images. "
                    "Please ensure images are clear and well-lit."
                )

            extraction_method = methods_used[0] if methods_used else "unavailable"
            doc.extracted_text = combined_text
            doc.extraction_method = extraction_method
            doc.page_count = len(image_paths)
            doc.status = "text_extracted"
            await db.commit()
            logger.info(
                f"[MULTI-IMG] Phase 1 done for {doc_id}: "
                f"{len(combined_text)} chars from {len(image_paths)} images"
            )

        except Exception as e:
            logger.error(f"[MULTI-IMG] Phase 1 failed for {doc_id}: {e}")
            async with AsyncSessionLocal() as err_db:
                res = await err_db.execute(select(Document).where(Document.id == doc_id))
                d = res.scalar_one_or_none()
                if d:
                    d.status = "failed"
                    await err_db.commit()
            return

    # --- Phase 2: Concurrent LLM Analysis + Embedding Pipeline -------------
    text = combined_text

    async def _run_summary_task():
        try:
            async with AsyncSessionLocal() as db_sum:
                from app.services.summary_service import generate_and_store_summary
                await generate_and_store_summary(db_sum, doc_id, text)
                await db_sum.commit()
            logger.info(f"[MULTI-IMG] ⚡ Auto-summary done for {doc_id}")
        except Exception as e:
            logger.error(f"[MULTI-IMG] Summary failed for {doc_id}: {e}")

    async def _run_embedding_task():
        try:
            async with AsyncSessionLocal() as db_emb:
                from app.services.rag_service import generate_document_chunks
                await generate_document_chunks(doc_id, text, db_emb)
                await db_emb.commit()
            logger.info(f"[MULTI-IMG] ⚡ Chunking & embedding done for {doc_id}")
        except Exception as e:
            logger.error(f"[MULTI-IMG] Embedding failed for {doc_id}: {e}")

    try:
        await asyncio.gather(
            _run_summary_task(),
            _run_fields_background(doc_id),
            _run_risks_background(doc_id),
            _run_embedding_task(),
            return_exceptions=True,
        )
        logger.info(f"[MULTI-IMG] ⚡ All parallel tasks complete for {doc_id}")
    except Exception as llm_err:
        logger.error(f"[MULTI-IMG] Analysis phase failed for {doc_id}: {llm_err}")

    async with AsyncSessionLocal() as db_final:
        try:
            res = await db_final.execute(select(Document).where(Document.id == doc_id))
            d = res.scalar_one_or_none()
            if d:
                d.status = "completed"
                await db_final.commit()
            logger.info(f"[MULTI-IMG] ✅ Document {doc_id} fully processed and marked completed!")
        except Exception as e:
            logger.error(f"[MULTI-IMG] Failed to update final status for {doc_id}: {e}")


@router.post(
    "/upload-images",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_multiple_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload multiple images of a single insurance report as one unified document.

    All images are OCR-processed concurrently and their text is concatenated
    in submission order (Page 1, Page 2, …).  The combined document then goes
    through the full auto-analysis pipeline (chunking, summary, fields, risks).
    """
    import hashlib

    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many images. Maximum 50 images per bundle.",
        )

    # Validate all files are images
    for f in files:
        ct = f.content_type or ""
        if ct not in ALLOWED_IMAGE_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"File '{f.filename}' is not a supported image type. "
                    "Allowed: JPG, PNG, TIFF, WEBP. Use the standard upload for PDFs."
                ),
            )

    # Read all files and validate total size
    file_contents: list[tuple[str, str, bytes]] = []  # (filename, content_type, data)
    total_size = 0
    for f in files:
        content = await f.read()
        total_size += len(content)
        if total_size > MAX_FILE_SIZE * 50:  # generous limit: 50 × single limit
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Total upload size exceeds the allowed limit.",
            )
        file_contents.append((f.filename or "image.jpg", f.content_type or "image/jpeg", content))

    # Create user upload directory
    user_upload_dir = FilePath(settings.UPLOAD_DIR) / str(current_user.id)
    user_upload_dir.mkdir(parents=True, exist_ok=True)

    # Save each image to disk, collect paths
    saved_paths: list[str] = []
    primary_filename = file_contents[0][0]  # Use first image's name as document name
    combined_hash_input = b""

    for orig_name, mime_type, data in file_contents:
        ext = FilePath(orig_name).suffix.lower() or ".jpg"
        stored_name = f"{uuid.uuid4()}{ext}"
        fpath = str(user_upload_dir / stored_name)
        with open(fpath, "wb") as fh:
            fh.write(data)
        saved_paths.append(fpath)
        combined_hash_input += data

    # Combined hash = hash of all image bytes concatenated
    combined_hash = hashlib.sha256(combined_hash_input).hexdigest()

    # Check for duplicate bundle
    existing = await db.execute(
        select(Document).where(
            Document.user_id == current_user.id,
            Document.file_hash == combined_hash,
        )
    )
    existing_doc = existing.scalar_one_or_none()
    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This exact set of images has already been uploaded as a document.",
        )

    # Derive a friendly document name:
    # If original filename has an extension, strip it; append page count info
    base_name = FilePath(primary_filename).stem
    bundle_name = f"{base_name} ({len(files)}-page bundle)"

    # Create a single Document record (status=uploaded; background task fills text)
    doc = Document(
        user_id=current_user.id,
        original_filename=bundle_name,
        stored_filename=FilePath(saved_paths[0]).name,  # primary image filename
        file_path=saved_paths[0],                       # primary image path
        file_type="image",
        file_size_bytes=total_size,
        mime_type="image/bundle",
        status="uploaded",
        file_hash=combined_hash,
        page_count=len(files),
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Kick off background OCR + full analysis pipeline
    background_tasks.add_task(process_multi_image_background, doc.id, saved_paths)

    logger.info(
        f"Multi-image bundle created: doc_id={doc.id}, pages={len(files)}, "
        f"user={current_user.id}"
    )
    return DocumentResponse.model_validate(doc)


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all documents for the current user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(desc(Document.created_at))
    )
    docs = result.scalars().all()
    return [DocumentResponse.model_validate(d) for d in docs]


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: All static sub-path routes (e.g. /reminders, /compare) MUST be
# registered BEFORE the /{document_id} wildcard to avoid being caught by it.
# ─────────────────────────────────────────────────────────────────────────────

class ScheduleReminderRequest(BaseModel):
    document_id: str
    renewal_date: Optional[datetime] = None
    premium_due_date: Optional[datetime] = None
    premium_amount: Optional[str] = None


class EmailReportRequest(BaseModel):
    email: EmailStr


@router.get("/reminders")
async def get_reminders(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List scheduled premium/renewal reminders for the user."""
    result = await db.execute(
        select(PolicyReminder)
        .where(PolicyReminder.user_id == current_user.id, PolicyReminder.is_dismissed == False)
        .order_by(PolicyReminder.reminder_date)
    )
    reminders = result.scalars().all()
    
    # Format responses dynamically
    return [
        {
            "id": r.id,
            "document_id": r.document_id,
            "title": r.title,
            "reminder_type": r.reminder_type,
            "reminder_date": r.reminder_date,
            "premium_amount": r.premium_amount,
            "is_dismissed": r.is_dismissed
        }
        for r in reminders
    ]
@router.post("/reminders")
async def schedule_reminder(
    request: ScheduleReminderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Schedule policy premium and renewal notifications."""
    result = await db.execute(
        select(Document).where(
            Document.id == request.document_id,
            Document.user_id == current_user.id
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Update document dates and ensure timezone-naive datetimes for PostgreSQL TIMESTAMP WITHOUT TIME ZONE columns
    renewal_naive = request.renewal_date.replace(tzinfo=None) if request.renewal_date else None
    premium_due_naive = request.premium_due_date.replace(tzinfo=None) if request.premium_due_date else None

    if renewal_naive:
        doc.renewal_date = renewal_naive
        # Create reminder alert
        # Trigger 7 days prior
        trigger_date = renewal_naive - timedelta(days=7)
        r1 = PolicyReminder(
            user_id=current_user.id,
            document_id=doc.id,
            title=f"Policy Renewal Approaching: {doc.original_filename}",
            reminder_type="renewal",
            reminder_date=trigger_date
        )
        db.add(r1)
        
    if premium_due_naive:
        doc.premium_due_date = premium_due_naive
        trigger_date = premium_due_naive - timedelta(days=5)
        r2 = PolicyReminder(
            user_id=current_user.id,
            document_id=doc.id,
            title=f"Premium Payment Approaching: {doc.original_filename}",
            reminder_type="premium",
            reminder_date=trigger_date,
            premium_amount=request.premium_amount
        )
        db.add(r2)
        
    await db.commit()

    # Trigger email notification to user
    try:
        send_alert_email_notification(
            user_name=current_user.full_name or current_user.email,
            user_email=current_user.email,
            policy_name=doc.original_filename,
            renewal_date=renewal_naive,
            premium_due_date=premium_due_naive,
            premium_amount=request.premium_amount
        )
    except Exception as email_err:
        logger.error(f"Failed to trigger email notification for reminder: {email_err}")

    return {"message": "Policy dates and reminders successfully scheduled"}


@router.patch("/reminders/{reminder_id}/dismiss")
async def dismiss_reminder(
    reminder_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Dismiss/acknowledge an active notification alert."""
    result = await db.execute(
        select(PolicyReminder).where(
            PolicyReminder.id == reminder_id,
            PolicyReminder.user_id == current_user.id
        )
    )
    reminder = result.scalar_one_or_none()
    if not reminder:
        raise HTTPException(status_code=404, detail="Notification alert not found")
        
    reminder.is_dismissed = True
    await db.commit()
    return {"message": "Notification successfully dismissed"}


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get detailed document info including AI results."""
    from sqlalchemy.orm import selectinload
    
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id, Document.user_id == current_user.id)
        .options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields),
            selectinload(Document.risk_analyses),
        )
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    return DocumentDetailResponse.model_validate(doc)


@router.post("/{document_id}/run-summary", status_code=status.HTTP_202_ACCEPTED)
async def trigger_background_summary(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Force-regenerate the AI Summary from the actual document text (deletes stale/mock data first).
    Returns 202 immediately. Poll GET /documents/{id} for results.
    """
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if not doc.extracted_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text not yet extracted. Please wait for processing.",
        )

    tracker_key = f"summary:{document_id}"
    if tracker_key in _analysis_in_progress:
        return {"status": "already_running", "message": "Summarization is already running for this document."}

    _analysis_in_progress.add(tracker_key)
    import asyncio
    asyncio.create_task(_run_summary_background(document_id, force_regenerate=True))

    logger.info(f"[API] Background re-summarization launched for {document_id}")
    return {
        "status": "started",
        "message": "Summary regeneration queued on server. Results will appear automatically.",
        "document_id": document_id,
        "job": "summary",
    }


@router.post("/{document_id}/run-fields", status_code=status.HTTP_202_ACCEPTED)
async def trigger_background_fields(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Kick off Extract Fields as a server-side asyncio background task.
    Returns 202 immediately. Poll GET /documents/{id} for results.
    """
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if not doc.extracted_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text not yet extracted. Please wait for processing.",
        )

    tracker_key = f"fields:{document_id}"
    if tracker_key in _analysis_in_progress:
        return {"status": "already_running", "message": "Field extraction is already running for this document."}

    _analysis_in_progress.add(tracker_key)
    import asyncio
    asyncio.create_task(_run_fields_background(document_id, force_regenerate=True))

    logger.info(f"[API] Background field extraction launched for {document_id}")
    return {
        "status": "started",
        "message": "Field extraction queued on server. Results will appear automatically — safe to navigate away.",
        "document_id": document_id,
        "job": "fields",
    }


@router.post("/{document_id}/run-risks", status_code=status.HTTP_202_ACCEPTED)
async def trigger_background_risks(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Kick off Risk Analysis as a server-side asyncio background task.
    Returns 202 immediately. Poll GET /documents/{id} for results.
    """
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if not doc.extracted_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text not yet extracted. Please wait for processing.",
        )

    tracker_key = f"risks:{document_id}"
    if tracker_key in _analysis_in_progress:
        return {"status": "already_running", "message": "Risk analysis is already running for this document."}

    _analysis_in_progress.add(tracker_key)
    import asyncio
    asyncio.create_task(_run_risks_background(document_id, force_regenerate=True))

    logger.info(f"[API] Background risk analysis launched for {document_id}")
    return {
        "status": "started",
        "message": "Risk analysis queued on server. Results will appear automatically — safe to navigate away.",
        "document_id": document_id,
        "job": "risks",
    }


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a document and its associated data."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id, Document.user_id == current_user.id
        )
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    # Delete physical file
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    
    await db.delete(doc)
    logger.info(f"Document {document_id} deleted")


@router.post("/compare", response_model=CompareResponse)
async def compare_documents(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compare 2 or 3 documents side-by-side."""
    from sqlalchemy.orm import selectinload
    
    # Check length
    if len(request.document_ids) < 2 or len(request.document_ids) > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must select between 2 and 3 documents to compare.",
        )
    
    # Fetch documents with their summary, extracted fields, and risks
    result = await db.execute(
        select(Document)
        .where(
            Document.id.in_(request.document_ids),
            Document.user_id == current_user.id
        )
        .options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields),
            selectinload(Document.risk_analyses),
        )
    )
    docs = result.scalars().all()
    
    # Verify all exist and belong to user
    if len(docs) != len(request.document_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more documents not found or access denied.",
        )
        
    # Check status of documents - they must have been processed (completed/summarized/text_extracted)
    for doc in docs:
        if doc.status in ("uploaded", "processing"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{doc.original_filename}' is still processing. Please wait.",
            )
        if doc.status == "failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{doc.original_filename}' failed to process and cannot be compared.",
            )

    # Format documents data for the AI synthesis service call
    policies_data = []
    for doc in docs:
        doc_dict = {
            "id": doc.id,
            "original_filename": doc.original_filename,
            "status": doc.status,
            "summary": {
                "summary_text": doc.summary.summary_text if doc.summary else "",
                "coverage_summary": doc.summary.coverage_summary if doc.summary else "",
                "exclusions_summary": doc.summary.exclusions_summary if doc.summary else "",
                "waiting_period_summary": doc.summary.waiting_period_summary if doc.summary else "",
                "premium_summary": doc.summary.premium_summary if doc.summary else "",
            } if doc.summary else None,
            "extracted_fields": [
                {
                    "field_name": f.field_name,
                    "field_value": f.field_value,
                    "field_category": f.field_category,
                }
                for f in doc.extracted_fields
            ],
            "risk_analyses": [
                {
                    "clause_text": r.clause_text,
                    "risk_type": r.risk_type,
                    "severity": r.severity,
                    "explanation": r.explanation,
                    "recommendation": r.recommendation,
                }
                for r in doc.risk_analyses
            ]
        }
        
        # Calculate dynamic overall risk level
        high_count = sum(1 for r in doc.risk_analyses if r.severity == "high")
        med_count = sum(1 for r in doc.risk_analyses if r.severity == "medium")
        if high_count > 0:
            doc_dict["overall_risk_level"] = "high"
        elif med_count > 0:
            doc_dict["overall_risk_level"] = "medium"
        else:
            doc_dict["overall_risk_level"] = "low"
            
        policies_data.append(doc_dict)

    # Generate comparison synthesis using Ollama or fallback mock
    synthesis_report = await generate_comparison_synthesis(policies_data)
    
    # Return documents details + comparative synthesis
    return CompareResponse(
        documents=[DocumentDetailResponse.model_validate(d) for d in docs],
        comparison_synthesis=ComparisonSynthesisSchema(
            synthesis=synthesis_report.get("synthesis", ""),
            best_for=synthesis_report.get("best_for", ""),
            verdict=synthesis_report.get("verdict", ""),
            feature_winners=synthesis_report.get("feature_winners", [])
        )
    )


# ─────────────────────────────────────────
# Reminders and Exporter Endpoints
# ─────────────────────────────────────────

class ScheduleReminderRequest(BaseModel):
    document_id: str
    renewal_date: Optional[datetime] = None
    premium_due_date: Optional[datetime] = None
    premium_amount: Optional[str] = None


class EmailReportRequest(BaseModel):
    email: EmailStr


@router.get("/reminders")
async def get_reminders(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List scheduled premium/renewal reminders for the user."""
    result = await db.execute(
        select(PolicyReminder)
        .where(PolicyReminder.user_id == current_user.id, PolicyReminder.is_dismissed == False)
        .order_by(PolicyReminder.reminder_date)
    )
    reminders = result.scalars().all()
    
    # Format responses dynamically
    return [
        {
            "id": r.id,
            "document_id": r.document_id,
            "title": r.title,
            "reminder_type": r.reminder_type,
            "reminder_date": r.reminder_date,
            "premium_amount": r.premium_amount,
            "is_dismissed": r.is_dismissed
        }
        for r in reminders
    ]





@router.patch("/reminders/{reminder_id}/dismiss")
async def dismiss_reminder(
    reminder_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Dismiss/acknowledge an active notification alert."""
    result = await db.execute(
        select(PolicyReminder).where(
            PolicyReminder.id == reminder_id,
            PolicyReminder.user_id == current_user.id
        )
    )
    reminder = result.scalar_one_or_none()
    if not reminder:
        raise HTTPException(status_code=404, detail="Notification alert not found")
        
    reminder.is_dismissed = True
    await db.commit()
    return {"message": "Notification successfully dismissed"}


def generate_html_report(doc: Document) -> str:
    """Helper to generate a clean, responsive HTML print template for policy reports."""
    fields_list = ""
    for f in doc.extracted_fields:
        fields_list += f"""
        <div class="field-item">
            <span class="field-label">{f.field_name}</span>
            <span class="field-value">{f.field_value or "—"}</span>
        </div>
        """
        
    risks_list = ""
    if not doc.risk_analyses:
        risks_list = "<p style='color: #10b981; font-weight: 500;'>No critical risk clauses detected.</p>"
    else:
        for r in doc.risk_analyses:
            color = "#f87171" if r.severity == "high" else ("#fbbf24" if r.severity == "medium" else "#34d399")
            risks_list += f"""
            <div class="risk-card" style="border-left: 4px solid {color}">
                <div class="risk-header">
                    <span class="risk-type">{r.risk_type.replace('_', ' ').upper()}</span>
                    <span class="risk-severity" style="color: {color}; font-weight: bold;">{r.severity.upper()}</span>
                </div>
                <p class="risk-clause"><strong>Clause:</strong> <em>"{r.clause_text}"</em></p>
                <p class="risk-explanation"><strong>Explanation:</strong> {r.explanation or "—"}</p>
                <p class="risk-rec"><strong>Recommendation:</strong> {r.recommendation or "—"}</p>
            </div>
            """
            
    summary_text = doc.summary.summary_text if doc.summary else "No summary available."
    coverage = doc.summary.coverage_summary if doc.summary else "—"
    exclusions = doc.summary.exclusions_summary if doc.summary else "—"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>HealthPolicyLens Document Report - {doc.original_filename}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                color: #0f172a;
                line-height: 1.5;
                margin: 0;
                padding: 40px;
                background: #f8fafc;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: #ffffff;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
                border: 1px solid #e2e8f0;
            }}
            .header {{
                border-bottom: 2px solid #3b82f6;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
                color: #1e3a8a;
            }}
            .metadata {{
                font-size: 12px;
                color: #64748b;
                margin-top: 5px;
            }}
            .section {{
                margin-bottom: 35px;
            }}
            .section h2 {{
                font-size: 16px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #475569;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 8px;
                margin-bottom: 15px;
            }}
            .field-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }}
            .field-item {{
                background: #f8fafc;
                padding: 12px 15px;
                border-radius: 8px;
                border: 1px solid #f1f5f9;
            }}
            .field-label {{
                display: block;
                font-size: 10px;
                text-transform: uppercase;
                color: #64748b;
                font-weight: bold;
            }}
            .field-value {{
                font-size: 14px;
                font-weight: 500;
                color: #1e293b;
                margin-top: 2px;
            }}
            .risk-card {{
                background: #fff8f8;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            }}
            .risk-header {{
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                font-weight: bold;
            }}
            .risk-clause {{
                font-size: 13px;
                color: #334155;
            }}
            .risk-explanation {{
                font-size: 13px;
                color: #475569;
            }}
            .risk-rec {{
                font-size: 12px;
                color: #2563eb;
                font-weight: 500;
            }}
            @media print {{
                body {{ background: none; padding: 0; }}
                .container {{ box-shadow: none; border: none; padding: 0; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Healthcare Policy Analysis Report</h1>
                <div class="metadata">
                    <strong>Document:</strong> {doc.original_filename} &nbsp;|&nbsp;
                    <strong>Processed:</strong> {doc.created_at.strftime('%Y-%m-%d')} &nbsp;|&nbsp;
                    <strong>Pages:</strong> {doc.page_count}
                </div>
            </div>
            
            <div class="section">
                <h2>AI Executive Summary</h2>
                <p style="font-size: 14px; line-height: 1.6; color: #334155;">{summary_text}</p>
                <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <strong style="font-size: 13px; color: #1e293b;">Top Coverages:</strong>
                        <pre style="font-family: inherit; font-size: 12px; color: #475569; white-space: pre-wrap; margin-top: 5px;">{coverage}</pre>
                    </div>
                    <div>
                        <strong style="font-size: 13px; color: #1e293b;">Top Exclusions:</strong>
                        <pre style="font-family: inherit; font-size: 12px; color: #475569; white-space: pre-wrap; margin-top: 5px;">{exclusions}</pre>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Extracted Policy Parameters</h2>
                <div class="field-grid">
                    {fields_list}
                </div>
            </div>
            
            <div class="section">
                <h2>Critical Risk Audit</h2>
                {risks_list}
            </div>
        </div>
    </body>
    </html>
    """
    return html


@router.get("/{id}/export", response_class=HTMLResponse)
async def export_report(
    id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export policy analysis report as a formatted printable HTML/PDF attachment."""
    result = await db.execute(
        select(Document)
        .where(Document.id == id, Document.user_id == current_user.id)
        .options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields),
            selectinload(Document.risk_analyses),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Explicitly load relationships to avoid lazy loading MissingGreenlet errors in sync generator
    await db.refresh(doc, ["summary", "extracted_fields", "risk_analyses"])
    
    html_content = generate_html_report(doc)
    headers = {"Content-Disposition": f"attachment; filename=HealthPolicyLens_Report_{doc.id}.html"}
    return HTMLResponse(content=html_content, headers=headers)


@router.post("/{id}/email")
async def email_report(
    id: str,
    request: EmailReportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Email the formatted HTML policy audit report directly to the user."""
    result = await db.execute(
        select(Document)
        .where(Document.id == id, Document.user_id == current_user.id)
        .options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields),
            selectinload(Document.risk_analyses),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Explicitly load relationships to avoid lazy loading MissingGreenlet errors in sync generator
    await db.refresh(doc, ["summary", "extracted_fields", "risk_analyses"])
    
    html_content = generate_html_report(doc)
    
    # 1. Setup email structure
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[HealthPolicyLens] Policy Analysis Audit Report: {doc.original_filename}"
    msg["From"] = "noreply@healthpolicylens.local"
    msg["To"] = request.email
    
    # Plaintext fallback
    text_fallback = f"Dear User,\n\nPlease find attached the HealthPolicyLens policy analysis report for {doc.original_filename}."
    part1 = MIMEText(text_fallback, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)
    
    # 2. Try sending SMTP or write to local debug folder
    sent_successfully = False
    error_msg = ""
    try:
        # Check if email configs are set up in environment, otherwise log
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if smtp_server and smtp_user and smtp_password:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(msg["From"], msg["To"], msg.as_string())
            sent_successfully = True
            logger.info(f"📧 Email report successfully sent via SMTP to {request.email}")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to send email via SMTP: {e}")
        
    # Write to local debug logs folder
    debug_dir = "./logs/sent_emails"
    os.makedirs(debug_dir, exist_ok=True)
    debug_filepath = f"{debug_dir}/email_{doc.id}_{uuid.uuid4().hex[:6]}.html"
    try:
        with open(debug_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"💾 Logged outgoing email report locally to: {debug_filepath}")
    except Exception as io_err:
        logger.error(f"Failed to write email debug log: {io_err}")
        
    if sent_successfully:
        return {"status": "sent", "message": f"Report successfully emailed to {request.email}."}
    else:
        return {
            "status": "logged",
            "message": f"SMTP is not configured in local development environment. Outgoing report logged locally to: {debug_filepath}.",
            "details": error_msg
        }


