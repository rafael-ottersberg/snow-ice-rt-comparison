"""What produced a result: code version and key library versions, stored in result metadata."""
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_state() -> str:
    """Short commit hash of this repository, ``-dirty`` if there are uncommitted changes, ``unknown`` outside git."""
    try:
        run = lambda *args: subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                                           text=True, check=True).stdout.strip()
        return run("rev-parse", "--short", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def software_versions(packages=("photontracer", "hoomd", "numpy")) -> dict:
    versions = {"icegrain": git_state(), "python": sys.version.split()[0]}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions
