import asyncio
import os
import sys

# Add backend app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal
from app.models.user import User
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(User.email, User.full_name, User.role))
        users = res.all()
        print("Users in DB:")
        for u in users:
            print(f"- Email: {u[0]}, Name: {u[1]}, Role: {u[2]}")

if __name__ == "__main__":
    asyncio.run(main())
