import nox

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
nox.options.default_venv_backend = "uv"

@nox.session(name="pre-commit", python=python_versions)
def precommit(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "-r", "pre-commit", "pre-commit-hooks", "ruff")
    session.run("pre-commit")

@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "-e", ".")
    session.run("uv", "pip", "install", "pytest", "coverage[toml]", "pygments")
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

@nox.session(python=python_versions)
def mypy(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "mypy", "pytest", "typeguard")
    session.run("mypy", "vespag", "tests")

@nox.session(python=python_versions)
def typeguard(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}")

@nox.session(python=python_versions)
def bandit(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "bandit")
    session.run("bandit", "vespag", "tests")

@nox.session(python=python_versions)
def safety(session: nox.Session) -> None:
    session.install("uv")
    session.run("uv", "pip", "install", "safety")
    requirements = session.poetry.export_requirements()
    session.run("safety", "scan", "--full-report", f"--file={requirements}")
