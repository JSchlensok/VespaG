import os

import nox

package = "vespag"
python_versions = ["3.10"]
nox.options.sessions = ["pre-commit", "tests", "coverage", "mypy", "beartype", "bandit", "safety"]
nox.options.default_venv_backend = "uv"


@nox.session(name="pre-commit", python=python_versions)
def precommit(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "pre-commit", "pre-commit-hooks", "ruff")
    session.run("pre-commit")


@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", ".")
    session.run("uv", "pip", "install", "pytest", "beartype", "coverage[toml]", "pygments")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest")
    finally:
        if session.interactive:
            session.notify("coverage")


@nox.session
def coverage(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "coverage[toml]")
    session.run("coverage", "combine")
    session.run("coverage", "report", "-i")


# TODO fix issues
@nox.session(python=python_versions)
def mypy(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "beartype", "mypy", "pytest")
    session.run("mypy", "vespag", "tests")


@nox.session(python=python_versions)
def beartype(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", ".")
    session.run("uv", "pip", "install", "beartype", "pytest", "pytest-beartype", "pygments")
    session.run("pytest", f"--beartype-packages={package}")


@nox.session(python=python_versions)
def bandit(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "bandit")
    session.run("bandit", "vespag", "tests")


@nox.session(python=python_versions)
def safety(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "safety")
    session.run("safety", "scan", "--policy-file", ".safety-policy.yml")
