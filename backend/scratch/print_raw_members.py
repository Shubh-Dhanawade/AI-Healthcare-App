import asyncio
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
    lines = text.split('\n')
    
    print("LINES AROUND SANGITA RAJKUMAR JAIN:")
    for idx, line in enumerate(lines):
        if 'Sangita Rajkumar' in line:
            # Print 3 lines before and after
            start_idx = max(0, idx - 4)
            end_idx = min(len(lines), idx + 5)
            for j in range(start_idx, end_idx):
                prefix = "--> " if j == idx else "    "
                clean_line = lines[j].encode('ascii', errors='replace').decode('ascii')
                print(f"{prefix}Line {j}: {repr(clean_line)}")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
