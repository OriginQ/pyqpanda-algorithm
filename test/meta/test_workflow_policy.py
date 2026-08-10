from pathlib import Path


def test_ci_does_not_ignore_pytest_failures():
    workflow = Path(".github/workflows/main.yml").read_text(encoding="utf-8")
    lines = workflow.splitlines()
    pytest_windows = ["\n".join(lines[index:index + 12]) for index, line in enumerate(lines) if "pytest" in line]
    assert pytest_windows
    assert all("|| true" not in window for window in pytest_windows)
    assert 'sed "s/-any-' not in workflow
