import aiosqlite
import asyncio
import json

class Database:
    def __init__(self, path: str = "database.db"):
        self.path = path

    async def _ensure_tables(self, db):
        await db.execute("""
            CREATE TABLE IF NOT EXISTS item_names (
                original_names TEXT,
                stripped_names TEXT,
                last_updated INTEGER
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS acknowledged (
                acknowledged BOOLEAN
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS items (
                name TEXT PRIMARY KEY,
                data TEXT,
                last_updated INTEGER
            )
        """)
        await db.commit()

    async def start(self):
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)

    async def insert(self, original_names: list[str], stripped_names: list[str], last_updated: int):
        original_json = json.dumps(original_names)
        stripped_json = json.dumps(stripped_names)
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)
            await db.execute("DELETE FROM item_names")
            await db.execute(
                "INSERT INTO item_names (original_names, stripped_names, last_updated) VALUES (?, ?, ?)",
                (original_json, stripped_json, last_updated)
            )
            await db.commit()

    async def save_item(self, name: str, data: dict, last_updated: int):
        json_data = json.dumps(data)
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)
            await db.execute("""
                INSERT INTO items (name, data, last_updated) VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET data=excluded.data, last_updated=excluded.last_updated
            """, (name, json_data, last_updated))
            await db.commit()

    async def fetch_item(self, name: str) -> tuple[dict | None, int | None]:
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)
            cursor = await db.execute("SELECT data, last_updated FROM items WHERE name = ?", (name,))
            row = await cursor.fetchone()
            if row:
                return json.loads(row[0]), row[1]
            return None, None

    async def acknowledge(self):
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)
            await db.execute("DELETE FROM acknowledged")
            await db.execute("INSERT INTO acknowledged (acknowledged) VALUES (1)")
            await db.commit()

    async def is_acknowledged(self):
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)
            cursor = await db.execute("SELECT acknowledged FROM acknowledged LIMIT 1")
            row = await cursor.fetchone()
            return row[0] if row else False

    async def fetch_all(self):
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)
            cursor = await db.execute("SELECT original_names, stripped_names, last_updated FROM item_names")
            rows = await cursor.fetchall()
            return [(json.loads(row[0]), json.loads(row[1]), row[2]) for row in rows]
    
    async def fetch_name_map(self) -> dict[str, str]:
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)
            cursor = await db.execute("SELECT original_names, stripped_names FROM item_names LIMIT 1")
            row = await cursor.fetchone()
            if not row:
                return {}
            original_names = json.loads(row[0])
            stripped_names = json.loads(row[1])
            return dict(zip(stripped_names, original_names))

    async def fetch_original_name(self, stripped_name: str) -> str | None:
        name_map = await self.fetch_name_map()
        return name_map.get(stripped_name)

    async def fetch_names_only(self):
        rows = await self.fetch_all()
        return rows[0][1] if rows else []

    async def fetch_original_names_only(self):
        rows = await self.fetch_all()
        return rows[0][0] if rows else []

    async def get_last_updated(self):
        async with aiosqlite.connect(self.path) as db:
            await self._ensure_tables(db)
            cursor = await db.execute("SELECT last_updated FROM item_names ORDER BY last_updated DESC LIMIT 1")
            row = await cursor.fetchone()
            return row[0] if row else None
