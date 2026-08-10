import asyncio
import re
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from app.services.ai_service import _regex_find_any

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    text_clean = re.sub(r'\s+', ' ', doc.extracted_text)
    
    # Test modified Pre-existing disease waiting period regex
    patterns = [
        r'[Pp]re-?existing\s+diseases?\s+waiting\s+period[^\n]{0,50}([\d\s/]+\s*(?:month|year|day)[s]?)',
        r'PED\s+wait\s+period[^\n]{0,60}([\d\s/]+\s*(?:Year|Month|year|month)[s]?)',
        r'[Pp]re-?existing\s+[Dd]isease[s]?\s+[Ww]aiting\s+[Pp]eriod[:\s]+([\d\s/]+\s*(?:month|year)[s]?)',
        r'[Pp]re-existing[^.\n]{0,30}([\d\s/]+\s*(?:month|year)[s]?[^.\n]{0,40})',
    ]
    
    val = _regex_find_any(patterns, text_clean)
    print(f"Extracted Pre-existing wait period: '{val}'")

if __name__ == "__main__":
    asyncio.run(main())
