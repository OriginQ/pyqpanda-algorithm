from pathlib import Path
from pyqpanda_alg import __version__


def test_changelog_mentions_current_version():
    changelog = Path("Tutorials/source/Changelog.rst").read_text(encoding="utf-8")
    assert __version__ in changelog
