"""
Run provenance: which code, configs, inputs and library versions produced a set of outputs.

Every stats table carries a `run_id` column. The matching `{ASSAY}_provenance.json`
in the logs folder records what went into that run. `run_id` is a hash of everything
except the timestamp, so identical code + configs + inputs give an identical run_id.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

_PACKAGES = ["behaviour-stat-pipeline", "numpy", "pandas", "scipy", "statsmodels", "patsy", "pyyaml"]


def sha256_file(path: str | Path | None) -> str | None:
    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # Only strip newlines: porcelain status lines start with a significant space
    return out.stdout.strip("\n")


def git_state(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root)
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "dirty_files": [] if not status else [line[3:] for line in status.splitlines()],
    }


def _package_versions() -> dict:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_provenance(*, repo_root: str | Path, files: dict[str, str | Path | None], argv: list[str]) -> dict:
    """
    files: role -> path, e.g. {"stats_config": ..., "processed": ..., "exclusions": ...}.
    Missing or empty paths are recorded as None.
    """
    prov = {
        "git": git_state(repo_root),
        "files": {
            role: {"path": None if p is None else str(p), "sha256": sha256_file(p)}
            for role, p in sorted(files.items())
        },
        "packages": _package_versions(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(argv),
    }
    stable = json.dumps({k: v for k, v in prov.items() if k not in ("argv", "platform")}, sort_keys=True)
    prov["run_id"] = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:12]
    prov["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return prov


def write_provenance(prov: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
