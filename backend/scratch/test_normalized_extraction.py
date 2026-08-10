import asyncio
import re
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from app.services.ai_service import _regex_find_any, _extract_premium_validated, _extract_insured_persons_validated

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    text = doc.extracted_text
    
    # Normalize whitespaces to single space
    norm_text = re.sub(r'\s+', ' ', text)
    
    print("TESTING EXTRACTION ON NORMALIZED TEXT:")
    
    # 1. Sum Insured
    sum_ins = _regex_find_any([
        r'sum\s+insured\s*(?:\(₹\))?\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:base\s+)?sum\s+insured\s*(?:opted)?\s*[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:sum\s+insured|sum\s+assured|si)[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:total\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
        r'(?:basic\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
        r'₹\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,})\s*(?:Lakh|Lakhs|lakh)?',
    ], norm_text)
    print(f"Extracted Sum Insured: {sum_ins}")
    
    # 2. Premium Amount
    prem = _extract_premium_validated(norm_text)
    print(f"Extracted Premium: {prem}")
    
    # 3. Insured Person
    insured = _extract_insured_persons_validated(norm_text)
    print(f"Extracted Insured Person: {insured}")
    
    # 4. Covered Members
    # Let's run a search for name + relationship in normalized text
    # Relationship list: Self, Spouse, Son, Daughter, etc.
    p_mem = r'\b([A-Z][A-Za-z .]{2,40})\s+(Self|Spouse|Son|Daughter|Father|Mother|Parent|Sibling|Brother|Sister|Child|Dependent)\b'
    members = re.findall(p_mem, norm_text)
    print("\nCovered Members Found:")
    clean_members = []
    noise = ["appointee", "nominee", "proposer", "insured", "holder", "relationship", "relation", "of the", "to the", "details", "policy", "premium"]
    for name, rel in members:
        name_clean = name.strip()
        rel_clean = rel.strip()
        # Remove trailing and leading garbage
        name_clean = re.sub(r'^(?:and|or|for|with|to|of|at|on|in|dear|miss)\s+', '', name_clean, flags=re.IGNORECASE)
        # Check first char is upper, no noise, and length >= 3
        if name_clean and name_clean[0].isupper() and len(name_clean) > 3 and not any(nw in name_clean.lower() for nw in noise):
            entry = f"{name_clean} ({rel_clean})"
            if entry not in clean_members:
                clean_members.append(entry)
                print(f"  - {entry}")
                
    # 5. Pre-existing Conditions (ICD Codes / Health Condition)
    print("\nPre-existing conditions matching:")
    conditions = re.findall(r'(?:Health\s+Condition\s*:\s*Pre\s+Existing\s+Disease|Health\s+Conditions\s+Elaboration|ICD\s+CODE\s+Description)\s+(.*?)(?=\s*(?:Premium\s+Details|Note|Declaration|\b\d{10,20}\b|HDFC\s+ERGO|$))', norm_text, re.IGNORECASE)
    for c in conditions:
        print(f"  Match: {c[:200]}")

if __name__ == "__main__":
    asyncio.run(main())
