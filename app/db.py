import asyncpg
from typing import Optional
from app.config import settings


class Database:
    """Database connection manager."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = (
            None  # None until connect() runs on app startup
        )

    async def connect(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            settings.database_url, min_size=5, max_size=20
        )

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the status."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """Fetch rows from a query."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row from a query."""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def is_healthy(self) -> bool:
        """Probe the pool with a trivial query; used by /health to report real status."""
        if self.pool is None:
            return False  # connect() never ran or failed silently
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False  # any connectivity/auth error means "not connected" for health purposes


# Global database instance (mutable singleton — fine for a single-process dev server,
# not safe to reassign across threads/workers without external coordination)
db = Database()
