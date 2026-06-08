"""V29: Add target_kind / target_id columns to notifications.

Extends the notification table to support the ``approval`` notification
kind: notifications that route to a server-side dispatcher when the user
answers them (e.g. approve / decline / snooze a queued mechanical
action). The existing ``type`` column gains a third valid value
(``approval``); the column itself stays TEXT so no schema change is
needed there.

The two new columns:

- ``target_kind`` TEXT NULL: dispatcher key (e.g. ``mechanical-action``,
  ``plan``). NULL for legacy ``notify`` / ``question`` rows, which means
  "no dispatch; fall through to the existing answer-injection path."
- ``target_id``   TEXT NULL: dispatcher-specific identifier (e.g. the
  mechanical-action proposal id). Read by the handler registry.

Existing rows are left untouched (target_kind = NULL), so the answer
path stays identical for every notification created before v29.
"""

from __future__ import annotations

import logging

import aiosqlite

logger = logging.getLogger(__name__)


async def up(db: aiosqlite.Connection) -> None:
    # Idempotent guard. This migration shipped as v028 in an earlier local
    # build, then was renumbered to v029 upstream (a new v028 — the codex
    # external_id migration — took slot 28). On a DB that already applied
    # the old v028, schema_version is 28 and these columns already exist, so
    # a bare ADD COLUMN raises "duplicate column name" and aborts startup.
    # Check before adding so re-running on such a DB is a no-op.
    cursor = await db.execute("PRAGMA table_info(notifications)")
    cols = {row[1] for row in await cursor.fetchall()}
    if "target_kind" not in cols:
        await db.execute("ALTER TABLE notifications ADD COLUMN target_kind TEXT")
    if "target_id" not in cols:
        await db.execute("ALTER TABLE notifications ADD COLUMN target_id TEXT")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_notifications_target "
        "ON notifications(target_kind, target_id)"
    )
    logger.info(
        "v029: ensured target_kind/target_id on notifications + index"
    )
