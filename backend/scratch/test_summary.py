import asyncio
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from sqlalchemy.orm import selectinload

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16').options(
            selectinload(Document.summary)
        )
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc or not doc.summary:
        print("Document or summary not found.")
        return
        
    print("\n--- POLICY SUMMARY NARRATIVE ---")
    safe_text = doc.summary.summary_text.encode('ascii', errors='replace').decode('ascii')
    print(safe_text)
    
    print("\n--- WAITING PERIOD SUMMARY BULLETS ---")
    safe_wait = doc.summary.waiting_period_summary.encode('ascii', errors='replace').decode('ascii')
    print(safe_wait)

if __name__ == "__main__":
    asyncio.run(main())
