import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        try:
            # Check version
            res = await db.execute(text("SELECT version();"))
            print("PostgreSQL Version:", res.scalar())
            
            # Check extensions
            res_ext = await db.execute(text("SELECT extname FROM pg_extension;"))
            print("Enabled Extensions:", [r[0] for r in res_ext.all()])
            
            # Try enabling pgvector
            print("Enabling pgvector extension if not exists...")
            await db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await db.commit()
            print("pgvector extension check complete!")
            
            # Check extensions again
            res_ext = await db.execute(text("SELECT extname FROM pg_extension;"))
            print("Enabled Extensions after check:", [r[0] for r in res_ext.all()])
            
        except Exception as e:
            print("Error connecting to PostgreSQL or enabling pgvector:", e)

if __name__ == "__main__":
    asyncio.run(main())
