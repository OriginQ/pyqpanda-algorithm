# Release Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current CPU library installable, testable, and release-gated before introducing runtime execution or new algorithms.

**Architecture:** Replace drifting setup scripts with one PEP 621 package definition and one version source. Move pytest discovery to the repository root, repair confirmed public API defects, and make GitHub Actions test the installed wheel without ignoring failures.

**Tech Stack:** Python, setuptools, pyproject.toml, pytest, GitHub Actions, Sphinx.

## Global Constraints

- Target version is exactly `2.1.0`.
- Existing valid CPU calls remain compatible.
- This plan does not introduce runtime execution or algorithm features.
- Build artifacts are pure Python `py3-none-any` wheels; filenames are never manually rewritten.
- Supported Python versions are only those verified against both pyqpanda3 and, later, qpanda3-runtime.
- Each task begins with a failing test or failing release check and ends with a focused commit.

---

### Task 1: Establish truthful pytest discovery

**Files:**
- Create: `pytest.ini`
- Delete: `test/pytest.ini`
- Create: `test/meta/test_test_inventory.py`
- Modify: `.github/workflows/main.yml:819-854`

**Interfaces:**
- Consumes: existing tests under `test/`
- Produces: `python -m pytest test` as the single repository test command

- [ ] **Step 1: Write the failing inventory test**

```python
from pathlib import Path


def test_test_files_are_not_fully_commented_out():
    ignored = {"test_test_inventory.py"}
    offenders = []
    for path in Path("test").rglob("*.py"):
        if path.name in ignored or "legacy_disabled" in path.parts:
            continue
        executable = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not executable:
            offenders.append(path.as_posix())
    assert offenders == []
```

- [ ] **Step 2: Run the inventory test and record the expected failure**

Run: `python -m pytest test/meta/test_test_inventory.py -v -o addopts=`

Expected: FAIL listing the currently fully commented test files.

- [ ] **Step 3: Create root pytest configuration and classify legacy commented files**

Use this exact root configuration:

```ini
[pytest]
testpaths = test
python_files = test_*.py Test_*.py
addopts = -ra --strict-markers
markers =
    runtime_contract: credential-free qpanda3-runtime contract test
    qpu: credentialed release-candidate QPU test
```

Move fully commented legacy tests to `test/legacy_disabled/` with filenames that do not match pytest patterns. Add `test/legacy_disabled/README.md` listing each moved file and the migration plan task that must restore or delete it. The inventory test must scan both active tests and reject newly added fully commented files outside `legacy_disabled`.

- [ ] **Step 4: Run collection and active CPU tests**

Run: `python -m pytest test --collect-only -q`

Expected: collection succeeds with no nonexistent `QRAM` path warning.

Run: `python -m pytest test -q`

Expected: all active tests pass; failures are not suppressed.

- [ ] **Step 5: Commit**

```bash
git add pytest.ini test/pytest.ini test/meta test/legacy_disabled .github/workflows/main.yml
git commit -m "test: establish truthful pytest discovery"
```

### Task 2: Consolidate package metadata and dependencies

**Files:**
- Create: `pyqpanda-algorithm/pyproject.toml`
- Create: `pyqpanda-algorithm/pyqpanda_alg/_version.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/__init__.py:29-53`
- Delete: `pyqpanda-algorithm/setup-cython.py`
- Delete: `pyqpanda-algorithm/setup.py`
- Delete: `pyqpanda-algorithm/requirements.txt`
- Modify: `pyqpanda-algorithm/MANIFEST.in`
- Create: `test/meta/test_package_metadata.py`

**Interfaces:**
- Consumes: package source under `pyqpanda-algorithm/pyqpanda_alg`
- Produces: `pyqpanda_alg.__version__ == "2.1.0"` and optional extra `runtime`

- [ ] **Step 1: Write failing version and metadata tests**

```python
import importlib.metadata
import pyqpanda_alg


def test_runtime_version_matches_distribution():
    assert pyqpanda_alg.__version__ == "2.1.0"
    assert pyqpanda_alg.__version__ == importlib.metadata.version("pyqpanda_alg")
```

- [ ] **Step 2: Run the metadata test and verify failure**

Run from an editable install: `python -m pytest test/meta/test_package_metadata.py -v`

Expected: FAIL because `pyqpanda_alg.__version__` is absent.

- [ ] **Step 3: Add the single version source and PEP 621 metadata**

Create `_version.py`:

```python
__version__ = "2.1.0"
```

Configure setuptools dynamic version from `pyqpanda_alg._version.__version__`. Runtime dependencies must include NumPy, SciPy, SymPy, Matplotlib, pydot, pandas, scikit-learn, and pyqpanda3. Put pytest, pytest-cov, coverage, build, PyYAML, and jsonschema tooling in `test`, Sphinx and sphinx-autoapi tooling in `docs`, and `qpanda3-runtime` in the `runtime` optional dependency. Remove mypy, requests, and pycryptodome from mandatory runtime dependencies unless a source import proves they are required.

- [ ] **Step 4: Build and install the wheel in a clean virtual environment**

Run:

```bash
python -m build --wheel pyqpanda-algorithm
python -m pip install --force-reinstall pyqpanda-algorithm/dist/pyqpanda_alg-2.1.0-py3-none-any.whl
python -c "import pyqpanda_alg; assert pyqpanda_alg.__version__ == '2.1.0'"
```

Expected: wheel uses `py3-none-any`; import succeeds.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm test/meta/test_package_metadata.py
git commit -m "build: consolidate package metadata"
```

### Task 3: Repair confirmed current API defects

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QAOA/qaoa.py:317-331`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QmRMR/__init__.py:6-8`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/__init__.py:29-53`
- Delete: `pyqpanda-algorithm/pyqpanda_alg/extensions/__init__.py`
- Create: `test/QAOA/test_qaoa_hamiltonian_input.py`
- Create: `test/QAlgBase/test_qmrmr_exports.py`
- Create: `test/meta/test_public_imports.py`

**Interfaces:**
- Consumes: `Hamiltonian.pauli_operator().terms()` from pyqpanda3
- Produces: working QAOA Hamiltonian construction and valid string-only `QmRMR.__all__`

- [ ] **Step 1: Write reproducing tests**

```python
from pyqpanda3.hamiltonian import Hamiltonian
from pyqpanda_alg.QAOA.qaoa import QAOA


def test_qaoa_accepts_hamiltonian():
    model = QAOA(Hamiltonian({"Z0": 1.0}))
    assert model.problem_dimension == 1
```

```python
def test_qmrmr_star_exports_feature_selection():
    namespace = {}
    exec("from pyqpanda_alg.QmRMR import *", namespace)
    assert "Feature_Selection" in namespace
```

- [ ] **Step 2: Verify both defects fail before modification**

Run: `python -m pytest test/QAOA/test_qaoa_hamiltonian_input.py test/QAlgBase/test_qmrmr_exports.py -v`

Expected: QAOA fails with missing `Hamiltonian.terms`; QmRMR fails because `__all__` contains a type.

- [ ] **Step 3: Apply minimal fixes and remove the unbuildable extension package**

Use `problem.pauli_operator().terms()` for Hamiltonian traversal. Set:

```python
__all__ = ["Feature_Selection"]
```

Remove the unimportable `extensions` package from the distribution and ensure top-level imports do not reference it. Do not add a replacement binary without source.

- [ ] **Step 4: Run focused and full CPU tests**

Run: `python -m pytest test/QAOA/test_qaoa_hamiltonian_input.py test/QAlgBase/test_qmrmr_exports.py test/meta/test_public_imports.py -v`

Expected: PASS.

Run: `python -m pytest test -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg test
git commit -m "fix: repair current public API defects"
```

### Task 4: Replace permissive CI with installed-wheel gates

**Files:**
- Modify: `.github/workflows/main.yml`
- Delete: `.workflow/ci-cd-pipeline.yml`
- Delete: `.travis.yml`
- Create: `test/meta/test_workflow_policy.py`

**Interfaces:**
- Consumes: root pytest command and PEP 621 build from Tasks 1-2
- Produces: one authoritative CI workflow that fails on test or build errors

- [ ] **Step 1: Write the workflow policy test**

```python
from pathlib import Path


def test_ci_does_not_ignore_pytest_failures():
    workflow = Path(".github/workflows/main.yml").read_text(encoding="utf-8")
    lines = workflow.splitlines()
    pytest_windows = ["\n".join(lines[index:index + 12]) for index, line in enumerate(lines) if "pytest" in line]
    assert pytest_windows
    assert all("|| true" not in window for window in pytest_windows)
    assert 'sed "s/-any-' not in workflow
```

- [ ] **Step 2: Verify it fails against the current workflow**

Run: `python -m pytest test/meta/test_workflow_policy.py -v`

Expected: FAIL because pytest failures are suppressed.

- [ ] **Step 3: Implement a matrix that builds once and tests installed wheels**

The workflow must:

1. Determine the verified Python/platform matrix.
2. Build with `python -m build`.
3. Upload the unrenamed wheel.
4. Install the wheel into a clean environment.
5. Run `python -m pytest test -q` from the checkout against the installed package.
6. Build Sphinx documentation.
7. Create release artifacts only after every required job succeeds.

Do not use `continue-on-error` or shell-level failure suppression around release gates.

- [ ] **Step 4: Validate YAML and policy locally**

Run: `python -m pytest test/meta/test_workflow_policy.py -v`

Expected: PASS.

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/main.yml', encoding='utf-8'))"`

Expected: exits zero.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/main.yml .workflow .travis.yml test/meta/test_workflow_policy.py
git commit -m "ci: enforce installed-wheel release gates"
```

### Task 5: Align documentation and release metadata

**Files:**
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `Tutorials/source/Changelog.rst`
- Modify: `Tutorials/source/conf.py:25-28`
- Modify: `Tutorials/source/index.rst`
- Create: `test/meta/test_documentation_version.py`

**Interfaces:**
- Consumes: `pyqpanda_alg.__version__`
- Produces: documentation that describes the current CPU-only baseline without false runtime claims

- [ ] **Step 1: Write a failing version consistency test**

```python
from pathlib import Path
from pyqpanda_alg import __version__


def test_changelog_mentions_current_version():
    changelog = Path("Tutorials/source/Changelog.rst").read_text(encoding="utf-8")
    assert __version__ in changelog
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/meta/test_documentation_version.py -v`

Expected: FAIL because the documentation has no 2.1.0 entry.

- [ ] **Step 3: Update documentation truthfully**

Set Sphinx `version` and `release` to `2.1.0`. Add a 2.1.0 development entry that distinguishes current CPU behavior from runtime work planned in later plans. Remove claims of complete real-hardware support until the release-qualification plan is finished.

- [ ] **Step 4: Build docs and run metadata tests**

Run: `python -m pytest test/meta/test_documentation_version.py -v`

Expected: PASS.

Run: `python -m sphinx -W -b html Tutorials/source Tutorials/build/html`

Expected: exits zero with warnings treated as errors.

- [ ] **Step 5: Commit**

```bash
git add README.md README_EN.md Tutorials test/meta/test_documentation_version.py
git commit -m "docs: align release metadata with version source"
```

### Task 6: Verify the baseline as a standalone deliverable

**Files:**
- Modify only if verification exposes a baseline defect in a file owned above

**Interfaces:**
- Consumes: Tasks 1-5
- Produces: a clean, CPU-only 2.1.0 development baseline ready for execution-layer work

- [ ] **Step 1: Remove prior local build output using the platform's safe, scoped cleanup command**

Targets are limited to `pyqpanda-algorithm/build`, `pyqpanda-algorithm/dist`, and `pyqpanda-algorithm/*.egg-info` after resolving them under the workspace.

- [ ] **Step 2: Build and inspect package artifacts**

Run: `python -m build pyqpanda-algorithm`

Expected: one sdist and one `py3-none-any` wheel for version 2.1.0.

- [ ] **Step 3: Run all tests against the installed wheel**

Run: `python -m pytest test -q`

Expected: PASS with no ignored collection paths.

- [ ] **Step 4: Build documentation with warnings as errors**

Run: `python -m sphinx -W -b html Tutorials/source Tutorials/build/html`

Expected: PASS.

- [ ] **Step 5: Commit any verification-only corrections**

```bash
git add pytest.ini pyqpanda-algorithm .github README.md README_EN.md Tutorials test
git commit -m "chore: finalize release baseline"
```
