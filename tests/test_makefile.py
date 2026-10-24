"""Contract tests for the Makefile command surface (#428).

Plans come from ``make -n``. Behaviour comes from real ``make`` runs in a scratch project whose
Hatch, pip and PATH tools are stubs that only log their calls: nothing is installed, no test lane
starts and no network is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKE = shutil.which("make") or "make"
PATH_TOOLS = ("uv", "git", "func", "azurite", "bicep")


def _hatch_scripts(env: str) -> dict[str, str]:
    envs: dict[str, dict[str, dict[str, str]]] = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["hatch"]["envs"]
    return {**envs["default"]["scripts"], **envs.get(env, {}).get("scripts", {})}


_HATCH_RUN = re.compile(r"\bhatch run (?:(?P<env>[\w-]+):)?(?P<script>[\w-]+)")


def _expand(plan: str) -> str:
    """Replace every ``hatch run <script>`` in *plan* by the command line that script runs."""

    def command(match: re.Match[str]) -> str:
        return _hatch_scripts(match["env"] or "default").get(match["script"], match[0])

    expanded = _HATCH_RUN.sub(command, plan)
    return expanded if expanded == plan else _expand(expanded)


def _plan(*args: str) -> str:
    result = subprocess.run(
        [MAKE, "-n", "--no-print-directory", "-C", str(REPO_ROOT), *args],
        env={"PATH": os.defpath},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _stub(path: Path, name: str, log: Path, body: str = "") -> Path:
    """An executable that appends ``name|arguments|PYTEST_ADDOPTS`` to *log*, then runs *body*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!/bin/sh\nprintf "%s|%s|%s\\n" "{name}" "$*" "${{PYTEST_ADDOPTS-}}" >> "{log}"\n{body}'
    )
    path.chmod(0o755)
    return path


@dataclass(frozen=True)
class Project:
    """A scratch checkout: the real Makefile, a stub Hatch and pip, stub tools as the only PATH."""

    root: Path
    tools: Path
    log: Path
    pip_template: Path

    def make(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        environment = {"PATH": str(self.tools), "PYTHON": sys.executable, **env}
        return subprocess.run(
            [MAKE, "-C", str(self.root), "--no-print-directory", *args],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def rows(self) -> list[tuple[str, str, str]]:
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return [(tool, args, addopts) for tool, args, addopts in (x.split("|", 2) for x in lines)]

    def calls(self, tool: str) -> list[str]:
        return [args for name, args, _ in self.rows() if name == tool]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    root, tools, log = tmp_path / "project", tmp_path / "tools", tmp_path / "calls.log"
    root.mkdir()
    shutil.copy(REPO_ROOT / "Makefile", root / "Makefile")
    sleep = shutil.which("sleep") or "sleep"
    hatch = _stub(
        tmp_path / "hatch.template",
        "hatch",
        log,
        f'[ -z "${{STUB_SLEEP-}}" ] || {{ {sleep} "$STUB_SLEEP"; printf "end||\\n" >> "{log}"; }}\n'
        'case "$*" in *"${STUB_FAIL:-@none@}"*) exit 1;; esac\n',
    )
    cp = shutil.which("cp") or "cp"
    pip = _stub(
        tmp_path / "pip.template",
        "pip",
        log,
        f'case "$*" in *hatch*) {cp} "{hatch}" "${{0%/*}}/hatch" 2> /dev/null;; esac\n',
    )
    (root / ".venv/bin").mkdir(parents=True)
    shutil.copy(hatch, root / ".venv/bin/hatch")
    shutil.copy(pip, root / ".venv/bin/pip")
    for tool in PATH_TOOLS:
        _stub(tools / tool, tool, log)
    return Project(root, tools, log, pip)


def _missing(result: subprocess.CompletedProcess[str]) -> list[str]:
    prefix = "MISSING  "
    return [
        line.removeprefix(prefix) for line in result.stdout.splitlines() if line.startswith(prefix)
    ]


def test_check_runs_every_check_exactly_once() -> None:
    """Offline: `make check` expands to one ruff check, one format check, one mypy and the lint."""
    plan = _expand(_plan("check"))

    assert plan.count("ruff check src tests") == 1
    assert plan.count("ruff format --check src tests") == 1
    assert plan.count("mypy src tests") == 1
    assert plan.count("tools/lint_hatch_matrix.py") == 1


@pytest.mark.parametrize("target", ["test", "check", "check-all"])
def test_no_routine_target_upgrades_pip(target: str) -> None:
    """Offline: no plan upgrades pip; Hatch is only installed while it is missing."""
    assert "--upgrade" not in _plan(target)


def test_install_ci_installs_into_the_selected_interpreter_and_not_through_hatch() -> None:
    """Offline: `install-ci` is `$(PYTHON) -m pip install -e ".[dev]"` and nothing else."""
    assert _plan("install-ci").strip() == 'python3 -m pip install -e ".[dev]"'
    assert (
        _plan("install-ci", "PYTHON=python3.12").strip() == 'python3.12 -m pip install -e ".[dev]"'
    )


def test_routine_targets_never_call_pip_while_hatch_is_present(project: Project) -> None:
    """Offline: repeated runs neither install nor upgrade anything."""
    for _ in range(3):
        assert project.make("test").returncode == 0

    assert project.calls("pip") == []
    assert project.calls("hatch") == ["run test"] * 3


def test_missing_hatch_is_installed_once_and_pip_is_never_upgraded(project: Project) -> None:
    """Offline: bootstrap runs one `pip install hatch` and later runs find Hatch."""
    (project.root / ".venv/bin/hatch").unlink()

    for _ in range(2):
        assert project.make("test").returncode == 0

    assert project.calls("pip") == ["install --disable-pip-version-check --quiet hatch"]
    assert project.calls("hatch") == ["run test"] * 2


def test_a_directory_at_the_hatch_path_is_not_treated_as_installed(project: Project) -> None:
    """Offline: `-x` is true for a directory, so bootstrap must still install Hatch."""
    hatch = project.root / ".venv/bin/hatch"
    hatch.unlink()
    hatch.mkdir()

    assert project.make("bootstrap").returncode == 0

    assert project.calls("pip") == ["install --disable-pip-version-check --quiet hatch"]


def test_missing_virtual_environment_is_built_by_the_selected_interpreter(
    project: Project, tmp_path: Path
) -> None:
    """Offline: bootstrap creates .venv with PYTHON, installs Hatch, and a second run is silent."""
    shutil.rmtree(project.root / ".venv")
    mkdir = shutil.which("mkdir") or "mkdir"
    cp = shutil.which("cp") or "cp"
    make_venv = f'{mkdir} -p "$3/bin"; {cp} "{project.pip_template}" "$3/bin/pip"'
    python = _stub(
        tmp_path / "python3.11",
        "python",
        project.log,
        f'case "$*" in "-m venv "*) {make_venv};; esac\n',
    )

    first = project.make("bootstrap", PYTHON=str(python))
    second = project.make("bootstrap", PYTHON=str(python))

    assert first.returncode == second.returncode == 0
    assert project.calls("python") == ["-m venv .venv"]
    assert project.calls("pip") == ["install --disable-pip-version-check --quiet hatch"]
    assert second.stdout == ""


@pytest.mark.parametrize("target", ["clean", "clean-all"])
def test_clean_removes_links_without_following_them(
    project: Project, tmp_path: Path, target: str
) -> None:
    """Offline: generated paths that are symlinks are unlinked and their targets are untouched."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    links = ["dist", "build", "__pycache__", ".pytest_cache", "pkg.egg-info"]
    if target == "clean-all":
        shutil.rmtree(project.root / ".venv")
        links += [".venv", ".mypy_cache", ".ruff_cache", "htmlcov", "site"]
    for name in links:
        (project.root / name).symlink_to(outside)

    result = project.make(target, PATH=os.defpath)

    assert result.returncode == 0, result.stderr
    assert (outside / "keep.txt").read_text() == "keep"
    assert not [name for name in links if os.path.lexists(project.root / name)]


def test_doctor_recipe_preserves_failed_check_status() -> None:
    recipe = next(
        line.strip().removeprefix("@").replace("$(MAKE) check-all", "false")
        for line in Path("Makefile").read_text().splitlines()
        if "Some checks failed." in line
    )

    result = subprocess.run(["sh", "-c", recipe], check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert "Some checks failed." in result.stdout
