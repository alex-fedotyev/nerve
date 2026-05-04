"""Service registry + port allocator for ephemeral dev servers.

Lets skills and the agent reserve a free port from a predictable range,
register what's running, and look up the URL to hand to Playwright or to
notify the user with.

Two zones of services:

- **in-agent**: started inside the Nerve agent container. Reachable from
  the host only if the chosen port falls inside one of the published
  ranges in ``docker-compose.yml``. Reachable from Playwright (when
  Playwright runs inside the agent) at ``http://localhost:<port>``.
- **sibling**: started as a separate container (e.g. via the docker MCP
  sidecar). The host daemon publishes the port directly. Reachable from
  the Mac browser at ``http://localhost:<port>`` and from inside the
  agent container at ``http://host.docker.internal:<port>``.

Each service kind has a non-overlapping range so the two zones never
fight for the same port. The in-agent ranges are the ones published in
the compose file; sibling ranges are owned by the host daemon.

Concurrency: writes are serialised by an ``fcntl.flock`` on the
registry file. Single-host, single-agent assumption, sufficient for
the spike. Multi-agent coordination is future work.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# Default registry file location. Overridden by ``NERVE_SERVICES_FILE``
# (handy for tests).
DEFAULT_REGISTRY_PATH = Path(
    os.environ.get(
        "NERVE_SERVICES_FILE",
        os.path.expanduser("~/.nerve/services.json"),
    )
)


# Service kinds and their non-overlapping port ranges.
#
# in-agent ranges MUST be published in docker-compose.yml so the host
# can reach them. sibling ranges are claimed directly on the host by
# the sibling container's ``-p`` mapping.
KINDS: dict[str, dict] = {
    "docs":        {"range": (3000, 3019), "zone": "in-agent"},
    "vite":        {"range": (5173, 5189), "zone": "in-agent"},
    "storybook":   {"range": (6006, 6019), "zone": "in-agent"},
    "hyperdx-app": {"range": (30200, 30219), "zone": "sibling"},
    "hyperdx-api": {"range": (30100, 30119), "zone": "sibling"},
    "grafana":     {"range": (3030, 3049), "zone": "sibling"},
    "general":     {"range": (8080, 8099), "zone": "either"},
}


VALID_ZONES = {"in-agent", "sibling", "either"}
VALID_STATUSES = {"allocated", "running", "stopped"}


# --- Errors -------------------------------------------------------------


class ServiceError(Exception):
    """Base class for service registry errors."""


class UnknownKind(ServiceError):
    pass


class NoFreePort(ServiceError):
    pass


class LabelInUse(ServiceError):
    pass


class UnknownLabel(ServiceError):
    pass


# --- Data model ---------------------------------------------------------


@dataclass
class ServiceEntry:
    """One row in the service registry."""

    label: str
    kind: str
    zone: str
    host_port: int
    status: str = "allocated"
    pid_or_container: str = ""
    cwd: str = ""
    command: str = ""
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Convenience: the URLs are computed, not stored, so they always
    # reflect the current ``NERVE_PLAYWRIGHT_LOCATION`` setting.

    @property
    def host_url(self) -> str:
        return f"http://localhost:{self.host_port}"

    @property
    def agent_url(self) -> str:
        if self.zone == "sibling":
            return f"http://host.docker.internal:{self.host_port}"
        return f"http://localhost:{self.host_port}"

    def playwright_url(self, location: str | None = None) -> str:
        """URL Playwright should dial, given where Playwright runs.

        ``location`` is one of ``agent`` (default), ``host``, or
        ``sidecar``. Sidecar and host both reach sibling-zone services
        on ``localhost``; agent reaches them via
        ``host.docker.internal``.
        """
        loc = location or os.environ.get("NERVE_PLAYWRIGHT_LOCATION", "agent")
        if loc == "agent":
            return self.agent_url
        return self.host_url

    def to_dict(self, playwright_location: str | None = None) -> dict:
        d = asdict(self)
        d["host_url"] = self.host_url
        d["agent_url"] = self.agent_url
        d["playwright_url"] = self.playwright_url(playwright_location)
        return d


# --- File IO and locking ------------------------------------------------


@contextlib.contextmanager
def _locked_registry(path: Path) -> Iterator[dict]:
    """Open the registry file with an exclusive lock.

    Reads, yields the parsed dict, writes back on close. Creates the
    file (and parent dir) if missing. The lock is released even if the
    caller raises.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open r+ if file exists, w+ otherwise. Use a+ so the file is
    # created on first run and the lock can be acquired immediately.
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with os.fdopen(fd, "r+", closefd=False) as f:
            f.seek(0)
            raw = f.read().strip()
            data = json.loads(raw) if raw else {"entries": []}
            yield data
            # Persist
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _entries(data: dict) -> list[dict]:
    return data.setdefault("entries", [])


def _find(data: dict, label: str) -> dict | None:
    for e in _entries(data):
        if e.get("label") == label:
            return e
    return None


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Best-effort check that ``port`` is not currently bound on host.

    Used as a defence against stale registry state (process gone but
    entry not released, or someone else grabbed the port outside the
    allocator). Returns ``True`` if ``bind`` succeeds.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            return False
        # Anything else (e.g. permission), fall through and treat as
        # free; the actual server start will fail loudly if it's not.
        return True
    finally:
        s.close()
    return True


# --- Public API ---------------------------------------------------------


class ServiceRegistry:
    """High-level wrapper around the registry file."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else DEFAULT_REGISTRY_PATH

    # -- lookups ---------------------------------------------------------

    def list(self, *, include_stopped: bool = False) -> list[dict]:
        with _locked_registry(self.path) as data:
            entries = list(_entries(data))
        if not include_stopped:
            entries = [e for e in entries if e.get("status") != "stopped"]
        return [
            ServiceEntry(**{k: v for k, v in e.items() if k in ServiceEntry.__dataclass_fields__}).to_dict()
            for e in entries
        ]

    def get(self, label: str) -> dict | None:
        with _locked_registry(self.path) as data:
            e = _find(data, label)
            if not e:
                return None
            entry = ServiceEntry(
                **{k: v for k, v in e.items() if k in ServiceEntry.__dataclass_fields__}
            )
            return entry.to_dict()

    # -- mutations -------------------------------------------------------

    def allocate(self, kind: str, label: str) -> int:
        """Reserve a free port for ``label`` from ``kind``'s range.

        Idempotent: re-allocating the same label returns the existing
        port. Raises ``NoFreePort`` if every port in the range is
        claimed (registry-level), or in use on the host (socket-level).
        """
        if kind not in KINDS:
            raise UnknownKind(
                f"Unknown service kind '{kind}'. "
                f"Known kinds: {sorted(KINDS)}"
            )
        lo, hi = KINDS[kind]["range"]
        zone = KINDS[kind]["zone"]

        with _locked_registry(self.path) as data:
            # Idempotent path: same label already has a port.
            existing = _find(data, label)
            if existing:
                if existing.get("kind") != kind:
                    raise LabelInUse(
                        f"Label '{label}' is already registered with "
                        f"kind '{existing.get('kind')}', not '{kind}'."
                    )
                return int(existing["host_port"])

            taken = {
                int(e["host_port"])
                for e in _entries(data)
                if e.get("kind") == kind and e.get("status") != "stopped"
            }
            for port in range(lo, hi + 1):
                if port in taken:
                    continue
                if not _is_port_free(port):
                    continue
                # Reserve it as ``allocated``; the caller fills in pid /
                # command via ``register`` once the process is up.
                entry = ServiceEntry(
                    label=label,
                    kind=kind,
                    zone=zone if zone != "either" else "in-agent",
                    host_port=port,
                    status="allocated",
                )
                _entries(data).append(asdict(entry))
                return port
        raise NoFreePort(
            f"No free port in {lo}-{hi} for kind '{kind}'."
        )

    def register(
        self,
        *,
        label: str,
        kind: str | None = None,
        zone: str | None = None,
        port: int | None = None,
        pid_or_container: str = "",
        cwd: str = "",
        command: str = "",
        status: str = "running",
    ) -> dict:
        """Promote an allocated entry to ``running`` (or create it).

        If ``label`` already exists, fields are merged. Otherwise a new
        entry is created, useful when the caller bypassed ``allocate``
        (e.g. binding to a specific port deliberately).
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'")
        if zone is not None and zone not in VALID_ZONES:
            raise ValueError(f"Invalid zone '{zone}'")

        with _locked_registry(self.path) as data:
            entry = _find(data, label)
            if entry is None:
                if kind is None or port is None:
                    raise UnknownLabel(
                        f"Label '{label}' is not allocated; "
                        "pass --kind and --port to register a fresh entry."
                    )
                if kind not in KINDS:
                    raise UnknownKind(f"Unknown kind '{kind}'")
                resolved_zone = zone or KINDS[kind]["zone"]
                if resolved_zone == "either":
                    resolved_zone = "in-agent"
                entry = asdict(
                    ServiceEntry(
                        label=label,
                        kind=kind,
                        zone=resolved_zone,
                        host_port=int(port),
                        status=status,
                        pid_or_container=pid_or_container,
                        cwd=cwd,
                        command=command,
                    )
                )
                _entries(data).append(entry)
            else:
                if kind and entry.get("kind") != kind:
                    raise LabelInUse(
                        f"Label '{label}' has kind "
                        f"'{entry.get('kind')}', not '{kind}'."
                    )
                if port is not None and int(entry.get("host_port", -1)) != int(port):
                    raise LabelInUse(
                        f"Label '{label}' is allocated to port "
                        f"{entry.get('host_port')}, not {port}."
                    )
                entry["status"] = status
                if zone:
                    entry["zone"] = zone
                if pid_or_container:
                    entry["pid_or_container"] = pid_or_container
                if cwd:
                    entry["cwd"] = cwd
                if command:
                    entry["command"] = command
            return ServiceEntry(
                **{
                    k: v
                    for k, v in entry.items()
                    if k in ServiceEntry.__dataclass_fields__
                }
            ).to_dict()

    def update(
        self,
        label: str,
        *,
        status: str | None = None,
        pid_or_container: str | None = None,
    ) -> dict:
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'")
        with _locked_registry(self.path) as data:
            entry = _find(data, label)
            if entry is None:
                raise UnknownLabel(f"No service with label '{label}'")
            if status is not None:
                entry["status"] = status
            if pid_or_container is not None:
                entry["pid_or_container"] = pid_or_container
            return ServiceEntry(
                **{
                    k: v
                    for k, v in entry.items()
                    if k in ServiceEntry.__dataclass_fields__
                }
            ).to_dict()

    def release(self, label: str) -> dict | None:
        """Mark ``label`` as stopped and free its port.

        The entry stays in the file (so historical lookups work) but is
        excluded from ``list`` by default and its port becomes
        re-allocatable.
        """
        with _locked_registry(self.path) as data:
            entry = _find(data, label)
            if entry is None:
                return None
            entry["status"] = "stopped"
            entry["stopped_at"] = datetime.now(timezone.utc).isoformat()
            return ServiceEntry(
                **{
                    k: v
                    for k, v in entry.items()
                    if k in ServiceEntry.__dataclass_fields__
                }
            ).to_dict()

    def reap_orphans(self) -> list[str]:
        """Mark entries whose pid is dead as ``stopped``.

        Only inspects entries with ``pid:NNN`` style identifiers (in-
        agent zone). Sibling-zone entries are left to the caller (we
        don't have docker socket access here by design).
        """
        reaped: list[str] = []
        with _locked_registry(self.path) as data:
            for e in _entries(data):
                if e.get("status") == "stopped":
                    continue
                ident = e.get("pid_or_container", "")
                if not ident.startswith("pid:"):
                    continue
                try:
                    pid = int(ident.split(":", 1)[1])
                except ValueError:
                    continue
                if not _pid_alive(pid):
                    e["status"] = "stopped"
                    e["stopped_at"] = datetime.now(timezone.utc).isoformat()
                    reaped.append(e["label"])
        return reaped


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` is alive (signal 0 doesn't kill it)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but we don't own it; still "alive" for
        # our purposes.
        return True
    return True


# Convenience module-level API --------------------------------------------


def allocate(kind: str, label: str, *, registry: ServiceRegistry | None = None) -> int:
    return (registry or ServiceRegistry()).allocate(kind, label)


def register(*, registry: ServiceRegistry | None = None, **kw) -> dict:
    return (registry or ServiceRegistry()).register(**kw)


def release(label: str, *, registry: ServiceRegistry | None = None) -> dict | None:
    return (registry or ServiceRegistry()).release(label)


def list_services(
    *,
    include_stopped: bool = False,
    registry: ServiceRegistry | None = None,
) -> list[dict]:
    return (registry or ServiceRegistry()).list(include_stopped=include_stopped)


def get(label: str, *, registry: ServiceRegistry | None = None) -> dict | None:
    return (registry or ServiceRegistry()).get(label)


# Re-export the kind registry so callers can introspect ranges.
__all__ = [
    "KINDS",
    "ServiceEntry",
    "ServiceRegistry",
    "ServiceError",
    "UnknownKind",
    "NoFreePort",
    "LabelInUse",
    "UnknownLabel",
    "DEFAULT_REGISTRY_PATH",
    "allocate",
    "register",
    "release",
    "list_services",
    "get",
    "main",
]


# --- CLI ---------------------------------------------------------------


def _cli_emit(obj: object, fmt: str) -> None:
    import json as _json

    if fmt == "json":
        print(_json.dumps(obj, indent=2, sort_keys=True))
    elif isinstance(obj, dict):
        for k in sorted(obj):
            print(f"{k}\t{obj[k]}")
    elif isinstance(obj, list):
        if not obj:
            return
        cols = ["label", "kind", "zone", "host_port", "status", "host_url", "agent_url"]
        print("\t".join(cols))
        for entry in obj:
            print("\t".join(str(entry.get(c, "")) for c in cols))
    else:
        print(obj)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``nerve-services`` console script.

    Kept inside the package so ``[project.scripts]`` can target it
    directly. The standalone ``scripts/nerve-services`` shim defers to
    this function so source-tree usage and pip-installed usage behave
    identically.
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="nerve-services",
        description="Port allocator + service registry.",
    )
    p.add_argument(
        "--file",
        default=os.environ.get("NERVE_SERVICES_FILE"),
        help="Path to the registry JSON (default: ~/.nerve/services.json).",
    )
    p.add_argument("--format", choices=("text", "json"), default="text")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("allocate", help="Reserve a free port for a label.")
    sp.add_argument("kind", help=f"One of: {', '.join(sorted(KINDS))}")
    sp.add_argument("label")

    sp = sub.add_parser("register", help="Mark an entry as running.")
    sp.add_argument("--label", required=True)
    sp.add_argument("--kind")
    sp.add_argument("--zone", choices=sorted(VALID_ZONES))
    sp.add_argument("--port", type=int)
    sp.add_argument("--pid", type=int, dest="pid")
    sp.add_argument("--container", dest="container")
    sp.add_argument("--cwd", default="")
    sp.add_argument("--command", default="")
    sp.add_argument(
        "--status",
        choices=sorted(VALID_STATUSES),
        default="running",
    )

    sp = sub.add_parser("update", help="Update status / pid of an entry.")
    sp.add_argument("label")
    sp.add_argument("--status", choices=sorted(VALID_STATUSES))
    sp.add_argument("--pid", type=int)
    sp.add_argument("--container", dest="container")

    sp = sub.add_parser("release", help="Stop and free a label's port.")
    sp.add_argument("label")

    sp = sub.add_parser("list", help="List active services.")
    sp.add_argument(
        "--all",
        action="store_true",
        help="Include stopped entries.",
    )

    sp = sub.add_parser("get", help="Show one service entry.")
    sp.add_argument("label")

    sp = sub.add_parser("reap", help="Mark entries with dead pids as stopped.")

    sp = sub.add_parser(
        "kinds",
        help="Print known service kinds and their port ranges.",
    )

    args = p.parse_args(argv)

    registry = ServiceRegistry(Path(args.file) if args.file else None)

    try:
        if args.cmd == "allocate":
            print(registry.allocate(args.kind, args.label))
            return 0

        if args.cmd == "register":
            ident = ""
            if args.pid:
                ident = f"pid:{args.pid}"
            elif args.container:
                ident = f"container:{args.container}"
            entry = registry.register(
                label=args.label,
                kind=args.kind,
                zone=args.zone,
                port=args.port,
                pid_or_container=ident,
                cwd=args.cwd,
                command=args.command,
                status=args.status,
            )
            _cli_emit(entry, args.format)
            return 0

        if args.cmd == "update":
            ident = None
            if args.pid:
                ident = f"pid:{args.pid}"
            elif args.container:
                ident = f"container:{args.container}"
            entry = registry.update(
                args.label, status=args.status, pid_or_container=ident,
            )
            _cli_emit(entry, args.format)
            return 0

        if args.cmd == "release":
            entry = registry.release(args.label)
            if entry is None:
                print(f"no such label: {args.label}", file=sys.stderr)
                return 1
            _cli_emit(entry, args.format)
            return 0

        if args.cmd == "list":
            _cli_emit(registry.list(include_stopped=args.all), args.format)
            return 0

        if args.cmd == "get":
            entry = registry.get(args.label)
            if entry is None:
                print(f"no such label: {args.label}", file=sys.stderr)
                return 1
            _cli_emit(entry, args.format)
            return 0

        if args.cmd == "reap":
            for label in registry.reap_orphans():
                print(label)
            return 0

        if args.cmd == "kinds":
            kinds = {
                name: {"range": list(spec["range"]), "zone": spec["zone"]}
                for name, spec in KINDS.items()
            }
            _cli_emit(kinds, args.format)
            return 0
    except ServiceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1
