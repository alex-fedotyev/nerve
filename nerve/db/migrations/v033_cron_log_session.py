"""V33: Link cron run logs to their agent sessions.

Adds ``cron_logs.session_id`` so each run can be traced to the session it
executed in (``cron:<job>`` for persistent jobs, ``cron:<job>:<run>`` for
isolated per-run sessions). The web UI uses this to deep-link from a cron
run row to its chat page. Source-runner logs keep NULL — they have no
agent session.
"""

from __future__ import annotations

import aiosqlite


async def up(db: aiosqlite.Connection) -> None:
    # Idempotent guard. A DB upgraded from an earlier local build can sit at
    # schema_version 33 with a different v033 already recorded (a local
    # messages.external_id backfill, since renumbered to v035). The runner is
    # keyed on MAX(version), so this migration can be skipped on such a DB; the
    # companion v034 backfill ensures the column there. Guarding the ADD COLUMN
    # keeps this safe to run anywhere without raising "duplicate column name".
    cursor = await db.execute("PRAGMA table_info(cron_logs)")
    cols = {row[1] for row in await cursor.fetchall()}
    if "session_id" not in cols:
        await db.execute("ALTER TABLE cron_logs ADD COLUMN session_id TEXT")
