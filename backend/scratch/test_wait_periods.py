import asyncio
import re
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    text_clean = re.sub(r'\s+', ' ', doc.extracted_text)
    
    # 1. Pre-existing disease waiting period
    m1 = re.search(r'Pre-existing diseases waiting period.*?(?:Code-Excl01)?[:\s\-]*([\d\s/]+months)', text_clean, re.IGNORECASE)
    ped_wait = m1.group(1).strip() if m1 else "None"
    
    # 2. Specified Disease/Procedure waiting period
    m2 = re.search(r'Specified Disease/Procedure waiting period.*?[:\s\-]*(\d+\s*months)', text_clean, re.IGNORECASE)
    spec_wait = m2.group(1).strip() if m2 else "None"
    
    # 3. Initial waiting Period
    m3 = re.search(r'Initial waiting Period.*?[:\s\-]*(\d+\s*days)', text_clean, re.IGNORECASE)
    init_wait = m3.group(1).strip() if m3 else "None"
    
    print(f"Pre-existing disease waiting period: '{ped_wait}'")
    print(f"Specified disease waiting period: '{spec_wait}'")
    print(f"Initial waiting period: '{init_wait}'")

if __name__ == "__main__":
    asyncio.run(main())
