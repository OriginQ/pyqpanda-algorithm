from pathlib import Path


def test_test_files_are_not_fully_commented_out():
    ignored = {"test_test_inventory.py"}
    offenders = []
    for path in Path("test").rglob("*.py"):
        if path.name in ignored:
            continue
        executable = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not executable:
            offenders.append(path.as_posix())
    assert offenders == []
