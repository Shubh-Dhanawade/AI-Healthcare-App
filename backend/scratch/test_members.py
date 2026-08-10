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
    
    # Let's test different patterns for covered members
    print("TESTING MEMBER REGEX PATTERNS:")
    
    # Try pattern 2 with \s+ instead of \s{2,}
    p2_mod = r'\b([A-Z][A-Za-z .]{2,40})\s+(Self|Spouse|Son|Daughter|Father|Mother|Parent|Sibling|Brother|Sister|Child|Dependent)\b'
    matches = re.findall(p2_mod, text)
    print("\n--- Pattern with \\s+ ---")
    for name, rel in matches:
        print(f"Match: name='{name.strip()}', rel='{rel.strip()}'")
        
    # Let's refine the filter: name must not contain certain words, and must be proper capitalized names
    clean_members = []
    noise = ["appointee", "nominee", "proposer", "insured", "holder", "relationship", "relation", "of the", "to the", "details", "policy", "premium"]
    for name, rel in matches:
        name_clean = name.strip()
        rel_clean = rel.strip()
        # Check if first character is capitalized and name doesn't contain noise
        if name_clean and name_clean[0].isupper() and not any(nw in name_clean.lower() for nw in noise):
            clean_members.append(f"{name_clean} ({rel_clean})")
            
    print("\nFiltered members:")
    for m in clean_members:
        print(f"  {m}")

if __name__ == "__main__":
    asyncio.run(main())
