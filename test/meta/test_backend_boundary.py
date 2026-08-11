from pathlib import Path


def test_algorithms_do_not_construct_cpuqvm_directly():
    source_root = Path("pyqpanda-algorithm/pyqpanda_alg")
    offenders = []
    for path in source_root.rglob("*.py"):
        if "execution" in path.parts:
            continue
        if "CPUQVM()" in path.read_text(encoding="utf-8"):
            offenders.append(path.as_posix())
    assert offenders == []
