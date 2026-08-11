import asyncio
import os
import sys
from sqlalchemy.orm import selectinload
from sqlalchemy import select

# Ensure backend folder is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.api.v1.documents import generate_html_report

async def main():
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
            
        print(f"Using document: {doc.original_filename} (ID: {doc.id})")
        
        # Explicitly load relationships to avoid lazy loading MissingGreenlet errors
        await db.refresh(doc, ["summary", "extracted_fields", "risk_analyses"])
        
        print("\nGenerating report in Hindi...")
        try:
            hindi_html = await generate_html_report(doc, "Hindi")
            hindi_path = "scratch/hindi_report.html"
            with open(hindi_path, "w", encoding="utf-8") as f:
                f.write(hindi_html)
            print(f"Saved Hindi report to: {hindi_path}")
        except Exception as e:
            print(f"Failed to generate Hindi report: {e}")
            
        print("\nGenerating report in Marathi...")
        try:
            marathi_html = await generate_html_report(doc, "Marathi")
            marathi_path = "scratch/marathi_report.html"
            with open(marathi_path, "w", encoding="utf-8") as f:
                f.write(marathi_html)
            print(f"Saved Marathi report to: {marathi_path}")
        except Exception as e:
            print(f"Failed to generate Marathi report: {e}")

if __name__ == "__main__":
    asyncio.run(main())
