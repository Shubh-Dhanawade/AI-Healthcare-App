"""
Script to seed the SQLite database with realistic RAG query evaluation logs.
These logs represent actual user queries and model performance.
"""

import sys
import os
import asyncio

# Add backend to path so app can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import AsyncSessionLocal
from app.services.seed_service import seed_initial_rag_logs

async def main():
    async with AsyncSessionLocal() as session:
        try:
            await seed_initial_rag_logs(session)
        except Exception as e:
            await session.rollback()
            print(f"❌ Error seeding database: {e}")
            raise e

if __name__ == "__main__":
    asyncio.run(main())
