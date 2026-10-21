"""Published notebooks must not leak the machine they were run on.

Stored outputs are published with the book, so an absolute path printed by a
cell (a home directory, a mount point) would reach every reader.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# An absolute local path, not the same segment inside a URL (".../home/terms").
LOCAL_PATH = re.compile(r"(?<![\w.:/-])/(?:mnt|home|Users|root|media|tmp)/[\w.-]+")


def notebooks():
    out = subprocess.run(["git", "ls-files", "*.ipynb"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return sorted(ROOT / p for p in out)


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebook_has_no_local_paths(path):
    hits = sorted(set(LOCAL_PATH.findall(path.read_text(encoding="utf-8"))))
    assert not hits, f"local filesystem paths in {path.name}: {hits}"


def test_pattern_ignores_urls_and_catches_paths():
    assert not LOCAL_PATH.search("https://www.nyc.gov/home/terms-of-use.page")
    assert LOCAL_PATH.search("artifacts dir: /mnt/data/processed")
    assert LOCAL_PATH.search('"/home/someone/project"')
