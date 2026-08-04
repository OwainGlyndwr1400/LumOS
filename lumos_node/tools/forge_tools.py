"""Forge verify tool — the ONLY new place a Forge session runs a command.

`forge_verify` executes an allow-listed verification command (pytest / ruff /
compile / build) inside a git workspace and returns its output, so the Forge
loop can run tests, READ the failures, and iterate — the run→observe→fix cycle
that separates a coding agent from a code suggester.

Security boundary (deny-by-default, layered):
  1. Gated on settings.forge_enabled — off = the tool refuses outright.
  2. The command's leading tokens must match an allow-listed PREFIX
     (settings.forge_verify_commands, else the built-in safe set). An unlisted
     command (rm, curl, python -c, git push…) is refused BEFORE any exec.
  3. cwd is validated against git_workspaces via the git tools' own
     _check_repo_path — the command can only run inside an approved repo.
  4. shell=False (shlex-split), scrubbed of shell metacharacters, hard timeout.
No network, no arbitrary shell, no path escape. This tool is intentionally NOT
in TOOL_CATEGORIES or AUTONOMOUS_PASSIVE_CATEGORIES, so normal chat routing and
autonomous wakes can never reach it — only an explicit Forge session passes it
in `allowed_tools`.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
from pathlib import Path

from ..config import get_settings
from ..log import get_logger
from . import register

log = get_logger(__name__)

# Built-in safe verify prefixes. Each is matched token-wise against the start of
# the requested command, so "pytest" allows "pytest tests/ -q" but not
# "pytest; rm -rf" (the ';' is rejected as a metachar before we even get here).
_DEFAULT_VERIFY_PREFIXES: tuple[str, ...] = (
    "pytest",
    "python -m pytest",
    "ruff check",
    "ruff format --check",
    "python -m compileall",
    "npm run build",
    "npm run lint",
    "npm test",
    "npm run test",
)

# Characters that must never appear in a verify command — they enable chaining,
# redirection, subshells, or globbing that would defeat the prefix allow-list.
_FORBIDDEN_CHARS = set(";&|`$><\n\r\\")

_VERIFY_TIMEOUT_S = 300
_MAX_OUTPUT_BYTES = 12_000


def _allowed_prefixes() -> list[str]:
    raw = get_settings().forge_verify_commands.strip()
    if not raw:
        return list(_DEFAULT_VERIFY_PREFIXES)
    return [p.strip() for p in raw.split(",") if p.strip()]


def is_allowed_command(command: str) -> tuple[bool, str]:
    """PURE — does `command` pass the allow-list + metachar checks?
    Returns (ok, reason). Split out so tests exercise it without a subprocess."""
    cmd = command.strip()
    if not cmd:
        return False, "empty command"
    if any(c in _FORBIDDEN_CHARS for c in cmd):
        return False, "command contains a forbidden shell metacharacter"
    # Normalize whitespace for prefix comparison ("pytest   tests" -> "pytest tests").
    norm = " ".join(cmd.split())
    for prefix in _allowed_prefixes():
        pfx = " ".join(prefix.split())
        if norm == pfx or norm.startswith(pfx + " "):
            return True, prefix
    return False, (
        "command not in the verify allow-list "
        f"(allowed prefixes: {', '.join(_allowed_prefixes())})"
    )


def _looks_pathish(frag: str) -> bool:
    """Heuristic: does this token fragment refer to a filesystem path (vs. a
    bare option like -q or a keyword expression)? Only path-ish fragments get
    the workspace-containment check, so `-p no:cacheprovider` isn't misflagged."""
    return (
        frag.startswith(("/", "\\", "~"))
        or (len(frag) >= 2 and frag[1] == ":")  # C:\ / C:/ drive-absolute
        or "/" in frag
        or "\\" in frag
        or ".." in frag
    )


def _path_fragments(token: str) -> list[str]:
    """Path-bearing fragments in a token: the token itself plus any value after
    '=' (e.g. --rootdir=/etc), each stripped of a pytest '::nodeid' suffix and
    surrounding quotes."""
    raw = [token]
    if "=" in token:
        raw.append(token.split("=", 1)[1])
    out: list[str] = []
    for f in raw:
        f = f.split("::", 1)[0].strip().strip('"').strip("'")
        if f and f != "-":
            out.append(f)
    return out


def _args_within_repo(argv: list[str], repo: Path) -> str | None:
    """Return an error string if any path-like argument resolves OUTSIDE `repo`.

    The prefix/metachar allow-list alone is insufficient: pytest (and similar)
    execute code discovered via their path arguments — a `conftest.py` in the
    ancestry of any collected path runs at collection time. So every path
    argument must resolve inside the approved workspace."""
    repo_r = repo.resolve()
    for tok in argv[1:]:
        for frag in _path_fragments(tok):
            if not _looks_pathish(frag):
                continue
            p = Path(frag)
            candidate = p if p.is_absolute() else (repo_r / frag)
            try:
                if not _resolve_within(candidate, repo_r):
                    return f"path argument escapes the workspace: {frag!r}"
            except OSError as e:
                return f"unresolvable path argument {frag!r}: {e}"
    return None


def _resolve_within(candidate: Path, root: Path) -> bool:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


def _truncate(s: str) -> tuple[str, bool]:
    b = s.encode("utf-8", "replace")
    if len(b) <= _MAX_OUTPUT_BYTES:
        return s, False
    return b[:_MAX_OUTPUT_BYTES].decode("utf-8", "replace"), True


@register(
    name="forge_verify",
    description=(
        "Run an allow-listed verification command (pytest / ruff / compile / "
        "npm build) inside the Forge workspace and return its exit code and "
        "output. Use this after making edits to check whether they pass — then "
        "read the failures and fix them. Only whitelisted commands are permitted. "
        "Use FORWARD slashes in any path arguments (e.g. 'pytest tests/test_x.py')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path of the workspace repo to run in (must be a configured git workspace).",
            },
            "command": {
                "type": "string",
                "description": "The verify command, e.g. 'pytest tests/ -q' or 'ruff check .'. Must match an allow-listed prefix.",
            },
        },
        "required": ["repo_path", "command"],
    },
)
async def forge_verify(repo_path: str, command: str) -> dict:
    settings = get_settings()
    if not settings.forge_enabled:
        return {"ok": False, "error": "Forge is disabled (set LUMOS_FORGE_ENABLED=true)."}

    ok, reason = is_allowed_command(command)
    if not ok:
        log.warning("forge.verify_refused", command=command[:120], reason=reason)
        return {"ok": False, "error": f"command refused: {reason}", "command": command}

    # Reuse the git tools' workspace validator — single source of truth for
    # "is this path an approved repo". Raises PermissionError if not.
    from .git_tools import _check_repo_path

    try:
        repo = _check_repo_path(repo_path)
    except PermissionError as e:
        return {"ok": False, "error": str(e)}
    if not repo.is_dir():
        return {"ok": False, "error": f"workspace path is not a directory: {repo}"}

    argv = shlex.split(command)
    # Path-argument containment: the prefix allow-list stops shell injection but
    # NOT `pytest /some/other/repo` (which imports that path's conftest.py at
    # collection → code execution outside the workspace). Refuse any path arg
    # that escapes `repo`.
    escape = _args_within_repo(argv, repo)
    if escape:
        log.warning("forge.verify_path_escape", command=command[:120], reason=escape)
        return {"ok": False, "error": f"command refused: {escape}", "command": command}

    try:
        # In a worker thread — a 5-minute pytest run must not block the event loop.
        proc = await asyncio.to_thread(
            subprocess.run,  # noqa: S603 — argv is allow-list-validated, shell=False
            argv,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=_VERIFY_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "error": f"command not found: {argv[0]}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"command timed out after {_VERIFY_TIMEOUT_S}s", "command": command}
    except OSError as e:
        return {"ok": False, "error": f"exec failed: {e}"}

    stdout, t1 = _truncate(proc.stdout or "")
    stderr, t2 = _truncate(proc.stderr or "")
    return {
        "ok": True,
        "command": command,
        "cwd": str(repo),
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": t1 or t2,
    }
