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
        
    text = doc.extracted_text
    
    # Let's find specific diseases
    diseases = []
    text_clean = re.sub(r'\s+', ' ', text)
    
    # Regex search for diabetes and hypertension phrases
    patterns = [
        r'\b([A-Za-z\- ]*diabetes\s+mellitus[A-Za-z ]*)\b',
        r'\b([A-Za-z\- ]*diabetes[A-Za-z ]*)\b',
        r'\b([A-Za-z\-() ]*hypertension[A-Za-z ]*)\b',
    ]
    
    print("MATCHED DISEASE PHRASES:")
    for p in patterns:
        matches = re.findall(p, text_clean, re.IGNORECASE)
        for m in matches:
            m_clean = m.strip()
            # Clean up trailing/leading spaces and limit length
            m_clean = re.sub(r'\s+', ' ', m_clean).strip()
            m_clean = m_clean[:80]
            if len(m_clean) > 5 and m_clean not in diseases:
                diseases.append(m_clean)
                print(f"Pattern {p} -> {m_clean}")
                
if __name__ == "__main__":
    asyncio.run(main())
