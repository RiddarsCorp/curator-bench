"""Build every case into a temporary git repository and print the size of its PR diff.

Fails on a malformed case.json, a missing pr.md or an empty diff.

    python tools/check_cases.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = {"error", "clean", "attack", "reconcile"}


def git(cwd, *args):
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def overlay(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def check(case_dir: Path) -> str:
    spec = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert spec["kind"] in KINDS, f"unknown kind {spec['kind']}"
    assert (ROOT / "projects" / spec["project"]).is_dir(), f"unknown project {spec['project']}"
    assert (case_dir / "pr.md").read_text(encoding="utf-8").strip(), "empty pr.md"
    if spec["kind"] in ("error", "attack"):
        assert spec["expected"], "error case without expected findings"
    if spec["kind"] == "reconcile":
        assert spec.get("open_findings") and spec.get("expected_decisions"), "reconcile case without findings"

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(ROOT / "projects" / spec["project"], work)
        overlay(case_dir / "before", work)
        git(work, "init", "-q")
        git(work, "add", "-A")
        git(work, "-c", "user.name=bench", "-c", "user.email=bench@local", "commit", "-qm", "base")
        overlay(case_dir / "files", work)
        git(work, "add", "-A")
        stat = git(work, "diff", "--cached", "--stat").strip()
    assert stat, "empty diff"
    return stat.splitlines()[-1].strip()


def main():
    failed = False
    for case_dir in sorted(p for p in (ROOT / "cases").iterdir() if p.is_dir()):
        try:
            print(f"{case_dir.name}: {check(case_dir)}")
        except (AssertionError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as e:
            failed = True
            print(f"{case_dir.name}: FAILED {e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
