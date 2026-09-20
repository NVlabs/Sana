import nox

LOCATIONS = ("src/", "tests/", "noxfile.py", "setup.py")


@nox.session
def lint(session):
    session.install("flake8")
    session.install("flake8-bugbear")
    session.install("flake8-isort")

    args = session.posargs or LOCATIONS
    session.run("flake8", *args)


@nox.session
def tests(session):
    session.install(".")
    session.install("pytest")
    session.install("pytest-mock")
    session.run("pytest", *session.posargs)
