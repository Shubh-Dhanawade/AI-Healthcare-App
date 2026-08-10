import asyncio
import re
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from app.services.ai_service import _extract_insured_persons_validated

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    text = doc.extracted_text
    
    # Run _extract_insured_persons_validated candidates print
    patterns = [
        r'(?:policyholder\s+name|proposer\s+name|proposer\s*/\s*policyholder|insured\s+name|name\s+of\s+insured)[:\s\-/]+([^\n\r]+)',
        r'(?:member\s+name|insured\s+person\(s\)|name\s+of\s+insured\s+person\(s\))[:\s\-/]+([^\n\r]+)',
    ]
    
    candidates = []
    
    # 1. Direct label extraction
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            first_line = m.split('\n')[0].strip()
            first_line = re.sub(r'[^a-zA-Z\s.\-]', '', first_line).strip()
            first_line = re.sub(r'\s+', ' ', first_line)
            candidates.append(("Pattern 1", first_line))
            
    # 2. Salutation based extraction (Mrs? or Ms or Miss)
    salutations = re.findall(r'\b(Mrs?|Ms|Miss)\.?\s+([A-Za-z\s.\-]{3,35})', text, re.IGNORECASE)
    for title, name_part in salutations:
        full_name = f"{title.strip()} {name_part.strip()}"
        first_line = full_name.split('\n')[0].strip()
        first_line = re.sub(r'[^a-zA-Z\s.]', '', first_line).strip()
        first_line = re.sub(r'\s+', ' ', first_line)
        first_line = re.sub(r'\s+(Base|Sum|Insured|Premium|Opted|Variant|Age|DOB|Gender|Relation).*$', '', first_line, flags=re.IGNORECASE).strip()
        candidates.append(("Salutation", first_line))

    # 3. Dear pattern
    dear_match = re.findall(r'(?:Dear|name\s+of\s+(?:insured|policyholder))[:\s,]+([A-Za-z\s.\-]{3,40})', text, re.IGNORECASE)
    for dm in dear_match:
        candidates.append(("Dear", dm.strip()))
        
    print("ALL CANDIDATES:")
    for src, c in candidates:
        print(f"  [{src}] -> {c}")

if __name__ == "__main__":
    asyncio.run(main())
