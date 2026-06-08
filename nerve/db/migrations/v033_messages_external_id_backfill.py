"""V33: Ensure messages.external_id on DBs that skipped the v028 codex migration.

The ``external_id`` column + partial unique index were introduced by
``v028_codex_message_external_id``. On installs that had already applied an
*earlier* local v028 (the notification approval-kind migration, since
renumbered to ``v029``), ``schema_version`` sat at 28, so the runner skipped
the upstream v028 codex migration on upgrade and ``external_id`` was never
created.

``MessageStore.add_message`` writes the ``external_id`` column on *every*
insert, so a missing column breaks all message writes. This migration adds it
idempotently. On fresh installs — where v028 already created the column — the
guards below make it a harmless no-op.
"""

from __future__ import annotations

import logging

import aiosqlite

logger = logging.getLogger(__name__)


async def up(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(messages)")
    cols = {row[1] for row in await cursor.fetchall()}
    if "external_id" not in cols:
        await db.execute("ALTER TABLE messages ADD COLUMN external_id TEXT")
    await db.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_external_id
           ON messages(session_id, external_id)
           WHERE external_id IS NOT NULL"""
    )
    logger.info("v033: ensured messages.external_id column + partial unique index")
