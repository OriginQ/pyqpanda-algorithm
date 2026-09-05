import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DOCS = (
    REPOSITORY_ROOT
    / "pyqpanda-algorithm"
    / "pyqpanda_alg"
    / "QSEncode"
)
NOTEBOOK = (
    REPOSITORY_ROOT
    / "pyqpanda-algorithm"
    / "example"
    / "QAlgBase"
    / "QSEncode_Insight_Demo.ipynb"
)


def test_qseencode_readme_has_required_user_sections_and_valid_links():
    readme = (PACKAGE_DOCS / "README.md").read_text(encoding="utf-8")
    for section in (
        "# QSEncode-Insight",
        "## Why it exists",
        "## Source-checkout setup",
        "## Quick start",
        "## Standard and audit verification",
        "## Evidence scope",
        "## CLI",
        "## Limitations",
    ):
        assert section in readme
    assert (PACKAGE_DOCS / "BENCHMARK_EVIDENCE.md").is_file()
    assert NOTEBOOK.is_file()
    assert "no automatic Walsh/Fourier selector" in readme
    assert "not a claim of quantum speedup" in readme


def test_benchmark_summary_retains_negative_evidence_and_claim_boundary():
    evidence = (PACKAGE_DOCS / "BENCHMARK_EVIDENCE.md").read_text(encoding="utf-8")
    normalized = " ".join(evidence.split())
    assert "480 evaluation cells" in evidence
    assert "45.27%" in evidence and "71.11%" in evidence
    assert "55 `do_not_compress`" in evidence
    assert "Dirichlet family was much weaker" in evidence
    assert "do not establish quantum advantage" in normalized


def test_notebook_is_small_clean_and_all_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert NOTEBOOK.stat().st_size < 100_000
    assert len(notebook["cells"]) == 17
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    for required in (
        "Walsh mode refuses compression",
        "Fourier mode selects sparse preparation",
        "Candidate table",
        "Audit verification",
        "EvidenceScope",
        "## Exercise",
        "## Limitations",
    ):
        assert required in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
