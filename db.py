import aiosqlite
import asyncio
import json

class Database:
    def __init__(self, path: str = "database.db"):
        self.path = path

    async def start(self):
        async with aiosqlite.connect(self.path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS item_names (
                    names TEXT,
                    last_updated INTEGER
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS acknowledged (
                    acknowledged BOOLEAN
                )
            """)     
            await db.commit()

    async def insert(self, names: list[str], last_updated: int):
        names_json = json.dumps(names)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("DELETE FROM item_names")
            await db.commit()
            await db.execute(
                "INSERT INTO item_names (names, last_updated) VALUES (?, ?)",
                (names_json, last_updated)
            )
            await db.commit()
    
    async def acknowledge(self):
        async with aiosqlite.connect(self.path) as db:
            await db.execute("INSERT INTO acknowledged (acknowledged) VALUES (1)")
            await db.commit()
    
    async def is_acknowledged(self):
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute("SELECT acknowledged FROM acknowledged")
            row = await cursor.fetchone()
            return row[0] if row else False
        

    async def fetch_all(self):
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute("SELECT names, last_updated FROM item_names")
            rows = await cursor.fetchall()
            return [(json.loads(row[0]), row[1]) for row in rows]

    async def fetch_names_only(self):
        rows = await self.fetch_all()
        return rows[0][0] if rows else []

    
    async def get_last_updated(self):
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute("SELECT last_updated FROM item_names ORDER BY last_updated DESC LIMIT 1")
            row = await cursor.fetchone()
            return row[0] if row else None


async def main():
    db = Database()
    await db.start()


