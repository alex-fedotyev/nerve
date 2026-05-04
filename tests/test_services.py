"""Unit tests for the port allocator + service registry."""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path

import pytest

from nerve import services


@pytest.fixture
def registry(tmp_path: Path) -> services.ServiceRegistry:
    return services.ServiceRegistry(tmp_path / "services.json")


# --- KINDS table sanity ----------------------------------------------------


def test_kinds_have_unique_ranges() -> None:
    """No two kinds may share a port; collisions break sibling launches."""
    seen: dict[int, str] = {}
    for name, spec in services.KINDS.items():
        lo, hi = spec["range"]
        assert lo <= hi, f"{name}: bad range {lo}-{hi}"
        for port in range(lo, hi + 1):
            assert port not in seen, (
                f"port {port} claimed by both {seen[port]} and {name}"
            )
            seen[port] = name


def test_kinds_zones_are_known() -> None:
    for name, spec in services.KINDS.items():
        assert spec["zone"] in services.VALID_ZONES, name


# --- allocate --------------------------------------------------------------


def test_allocate_returns_port_in_range(registry: services.ServiceRegistry) -> None:
    port = registry.allocate("docs", "docs:branch-a")
    lo, hi = services.KINDS["docs"]["range"]
    assert lo <= port <= hi


def test_allocate_is_idempotent(registry: services.ServiceRegistry) -> None:
    p1 = registry.allocate("docs", "docs:branch-a")
    p2 = registry.allocate("docs", "docs:branch-a")
    assert p1 == p2


def test_allocate_two_labels_get_different_ports(
    registry: services.ServiceRegistry,
) -> None:
    p1 = registry.allocate("docs", "docs:branch-a")
    p2 = registry.allocate("docs", "docs:branch-b")
    assert p1 != p2


def test_allocate_unknown_kind_raises(registry: services.ServiceRegistry) -> None:
    with pytest.raises(services.UnknownKind):
        registry.allocate("not-a-kind", "x")


def test_allocate_no_free_port(
    registry: services.ServiceRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every port in the kind's range is claimed, allocate raises.

    Uses a synthetic kind with a tiny range so the test does not depend
    on the real ranges being unused on the test host (CI sometimes has
    other services on 3000+).
    """
    monkeypatch.setitem(
        services.KINDS, "_test_tiny", {"range": (49952, 49953), "zone": "in-agent"},
    )
    registry.allocate("_test_tiny", "tiny:1")
    registry.allocate("_test_tiny", "tiny:2")
    with pytest.raises(services.NoFreePort):
        registry.allocate("_test_tiny", "tiny:overflow")


def test_allocate_label_kind_mismatch(registry: services.ServiceRegistry) -> None:
    registry.allocate("docs", "shared-label")
    with pytest.raises(services.LabelInUse):
        registry.allocate("vite", "shared-label")


def test_allocate_skips_ports_in_use_on_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a port in the range is already bound by something outside the
    registry, the allocator should skip it and return the next one."""

    # Grab any free port via port=0, then build a synthetic kind whose
    # range straddles it. Allocator must skip the bound port and return
    # the neighbour. Avoids depending on a fixed port being free on the
    # test host.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    bound = s.getsockname()[1]
    try:
        monkeypatch.setitem(
            services.KINDS,
            "_test_skip",
            {"range": (bound, bound + 1), "zone": "in-agent"},
        )
        registry = services.ServiceRegistry(tmp_path / "services.json")
        port = registry.allocate("_test_skip", "skip:occupied")
        assert port == bound + 1
    finally:
        s.close()


# --- register / update / release -------------------------------------------


def test_register_promotes_allocated(registry: services.ServiceRegistry) -> None:
    port = registry.allocate("docs", "docs:r1")
    entry = registry.register(
        label="docs:r1",
        kind="docs",
        port=port,
        pid_or_container="pid:12345",
        cwd="/tmp/x",
        command="yarn start",
    )
    assert entry["status"] == "running"
    assert entry["host_port"] == port
    assert entry["host_url"] == f"http://localhost:{port}"
    assert entry["agent_url"] == f"http://localhost:{port}"  # in-agent zone
    assert entry["pid_or_container"] == "pid:12345"


def test_register_sibling_zone_uses_host_docker_internal_for_agent(
    registry: services.ServiceRegistry,
) -> None:
    port = registry.allocate("grafana", "grafana:test")
    entry = registry.register(
        label="grafana:test", kind="grafana", port=port,
    )
    assert entry["zone"] == "sibling"
    assert entry["agent_url"] == f"http://host.docker.internal:{port}"
    assert entry["host_url"] == f"http://localhost:{port}"


def test_register_creates_fresh_entry_without_allocate(
    registry: services.ServiceRegistry,
) -> None:
    entry = registry.register(
        label="manual:hand-rolled", kind="general", port=8090,
    )
    assert entry["host_port"] == 8090
    assert entry["status"] == "running"


def test_register_requires_kind_and_port_for_new_label(
    registry: services.ServiceRegistry,
) -> None:
    with pytest.raises(services.UnknownLabel):
        registry.register(label="ghost")


def test_release_marks_stopped_and_frees_port(
    registry: services.ServiceRegistry,
) -> None:
    p1 = registry.allocate("docs", "docs:short-lived")
    registry.release("docs:short-lived")

    # Port should be reusable.
    p2 = registry.allocate("docs", "docs:second")
    # The new label may get the same port back, or a different one if
    # the OS is holding it briefly. Both are acceptable; what matters is
    # that we did not fail to allocate.
    assert services.KINDS["docs"]["range"][0] <= p2 <= services.KINDS["docs"]["range"][1]
    # Released entry not in default list.
    labels = {e["label"] for e in registry.list()}
    assert "docs:short-lived" not in labels
    # ...but it is still in the file when we ask for stopped entries.
    labels_all = {e["label"] for e in registry.list(include_stopped=True)}
    assert "docs:short-lived" in labels_all


def test_release_unknown_returns_none(registry: services.ServiceRegistry) -> None:
    assert registry.release("never-registered") is None


def test_update_changes_status(registry: services.ServiceRegistry) -> None:
    port = registry.allocate("docs", "docs:u")
    registry.register(label="docs:u", kind="docs", port=port, pid_or_container="pid:1")
    updated = registry.update("docs:u", status="stopped")
    assert updated["status"] == "stopped"


# --- list / get -----------------------------------------------------------


def test_list_excludes_stopped_by_default(registry: services.ServiceRegistry) -> None:
    registry.allocate("docs", "docs:keep")
    registry.allocate("docs", "docs:gone")
    registry.release("docs:gone")
    listed = {e["label"] for e in registry.list()}
    assert listed == {"docs:keep"}


def test_get_returns_none_for_missing(registry: services.ServiceRegistry) -> None:
    assert registry.get("ghost") is None


def test_get_returns_url_fields(registry: services.ServiceRegistry) -> None:
    port = registry.allocate("docs", "docs:g")
    entry = registry.get("docs:g")
    assert entry is not None
    assert entry["host_url"] == f"http://localhost:{port}"
    assert entry["agent_url"] == f"http://localhost:{port}"
    assert entry["playwright_url"] == f"http://localhost:{port}"


def test_playwright_url_respects_location(
    registry: services.ServiceRegistry, monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = registry.allocate("grafana", "grafana:pw")
    monkeypatch.setenv("NERVE_PLAYWRIGHT_LOCATION", "host")
    entry = registry.get("grafana:pw")
    assert entry is not None
    # When Playwright runs on the host, it talks to siblings on
    # localhost (the host daemon owns the port).
    assert entry["playwright_url"] == f"http://localhost:{port}"

    monkeypatch.setenv("NERVE_PLAYWRIGHT_LOCATION", "agent")
    entry = registry.get("grafana:pw")
    assert entry is not None
    # When Playwright runs in the agent, sibling-zone services need the
    # bridge name.
    assert entry["playwright_url"] == f"http://host.docker.internal:{port}"


# --- reap_orphans ---------------------------------------------------------


def test_reap_orphans_marks_dead_pids_stopped(
    registry: services.ServiceRegistry,
) -> None:
    port = registry.allocate("docs", "docs:dead")
    registry.register(
        label="docs:dead", kind="docs", port=port, pid_or_container="pid:999999999",
    )
    reaped = registry.reap_orphans()
    assert reaped == ["docs:dead"]


def test_reap_orphans_skips_live_pids(registry: services.ServiceRegistry) -> None:
    port = registry.allocate("docs", "docs:alive")
    registry.register(
        label="docs:alive", kind="docs", port=port,
        pid_or_container=f"pid:{os.getpid()}",
    )
    reaped = registry.reap_orphans()
    assert reaped == []


def test_reap_orphans_ignores_container_idents(
    registry: services.ServiceRegistry,
) -> None:
    port = registry.allocate("grafana", "grafana:c")
    registry.register(
        label="grafana:c", kind="grafana", port=port,
        pid_or_container="container:grafana-pr-1733",
    )
    # Containers aren't pid-checkable from inside the agent; reap should
    # leave them alone.
    assert registry.reap_orphans() == []
    entry = registry.get("grafana:c")
    assert entry is not None
    assert entry["status"] == "running"


# --- file format / persistence --------------------------------------------


def test_registry_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "services.json"
    r1 = services.ServiceRegistry(path)
    port = r1.allocate("docs", "docs:persist")

    r2 = services.ServiceRegistry(path)
    entry = r2.get("docs:persist")
    assert entry is not None
    assert entry["host_port"] == port


def test_registry_file_is_pretty_printed_json(tmp_path: Path) -> None:
    path = tmp_path / "services.json"
    services.ServiceRegistry(path).allocate("docs", "docs:fmt")
    raw = path.read_text()
    assert raw.startswith("{")
    parsed = json.loads(raw)
    assert "entries" in parsed
    assert isinstance(parsed["entries"], list)


def test_concurrent_allocate_does_not_collide(tmp_path: Path) -> None:
    """Two threads allocating from the same kind must get different ports."""
    path = tmp_path / "services.json"
    results: list[int] = []
    errors: list[Exception] = []

    def worker(label: str) -> None:
        try:
            r = services.ServiceRegistry(path)
            results.append(r.allocate("docs", label))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(f"docs:t-{i}",))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(results) == 8
    assert len(set(results)) == 8


# --- CLI smoke ------------------------------------------------------------


def test_cli_kinds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("NERVE_SERVICES_FILE", str(tmp_path / "services.json"))
    rc = services.main(["--format", "json", "kinds"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "docs" in payload
    assert payload["docs"]["zone"] == "in-agent"


def test_cli_allocate_register_release_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NERVE_SERVICES_FILE", str(tmp_path / "services.json"))

    rc = services.main(["allocate", "docs", "docs:cli-flow"])
    assert rc == 0
    port = int(capsys.readouterr().out.strip())

    rc = services.main([
        "--format", "json",
        "register",
        "--label", "docs:cli-flow",
        "--kind", "docs",
        "--port", str(port),
        "--pid", str(os.getpid()),
        "--command", "yarn start",
    ])
    assert rc == 0
    entry = json.loads(capsys.readouterr().out)
    assert entry["label"] == "docs:cli-flow"
    assert entry["status"] == "running"

    rc = services.main(["--format", "json", "list"])
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    assert any(e["label"] == "docs:cli-flow" for e in listed)

    rc = services.main(["--format", "json", "release", "docs:cli-flow"])
    assert rc == 0


def test_cli_release_unknown_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NERVE_SERVICES_FILE", str(tmp_path / "services.json"))
    rc = services.main(["release", "no-such-label"])
    assert rc != 0
