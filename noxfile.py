import nox
from nox_poetry import Session, session

package = "vespag"
python_versions = ["3.10"]
nox.options.sessions = [
    "pre-commit",
    "tests",
    "coverage",
    "mypy",
    "typeguard",
    "bandit",
    "safety"
]
nox.options.default_venv_backend = "micromamba"

@session(name="pre-commit", python=python_versions)
def precommit(session: Session) -> None:
    session.install("pre-commit", "pre-commit-hooks", "ruff")
    session.run("pre-commit")

@session(python=python_versions)
def tests(session: Session) -> None:
    session.install(".")
    session.install("pytest", "coverage[toml]", "pygments")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest")
    finally:
        if session.interactive:
            session.notify("coverage")

@session
def coverage(session: Session) -> None:
    session.install("coverage[toml]")
    session.run("coverage", "combine")
    session.run("coverage", "report", "-i")

@session(python=python_versions)
def mypy(session: Session) -> None:
    session.install(".")
    session.install("mypy", "pytest", "typeguard")
    session.run("mypy", "vespag", "tests")

@session(python=python_versions)
def typeguard(session: Session) -> None:
    session.install(".")
    session.install("pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}")

@session(python=python_versions)
def bandit(session: Session) -> None:
    session.install("bandit")
    session.run("bandit", "vespag", "tests")

@session(python=python_versions)
def safety(session: Session) -> None:
    session.install("safety")
    requirements = session.poetry.export_requirements()
    session.run("safety", "scan", "--full-report", f"--file={requirements}")
