# Runtime Release Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce credential-safe FakeBackend and real-QPU evidence for every public algorithm and prevent release creation until code, package, documentation, and qualification records agree on version 2.1.0.

**Architecture:** Standard CI remains credential-free. A manually dispatched RC workflow installs the candidate wheel, queries an explicitly selected device, runs fixed preflight/QPU cases, writes a sanitized machine-readable manifest, and feeds a deterministic release-policy check.

**Tech Stack:** qpanda3-runtime, pytest qpu markers, JSON Schema, GitHub Actions workflow_dispatch, Sphinx, Python packaging.

## Global Constraints

- RC authentication comes only from `QPANDA3_API_KEY` supplied by workflow secrets or the operator environment.
- No API key, token, RuntimeService object, or raw credential-bearing response is written to artifacts.
- Every executable algorithm has at least one real-QPU task; circuit-only components have a successful device-transpilation record.
- Statistical thresholds, shots, seeds, and maximum attempts are committed before QPU execution.
- Device outage may postpone qualification but cannot be converted into a passing result.
- GitHub Release creation requires a valid qualification manifest for the exact commit and wheel hash.

---

### Task 1: Define the qualification manifest and sanitizer

**Files:**
- Create: `tools/release_qualification/schema.json`
- Create: `tools/release_qualification/manifest.py`
- Create: `test/release/conftest.py`
- Create: `test/release/test_qualification_manifest.py`
- Create: `test/release/test_qualification_security.py`

**Interfaces:**
- Produces: `QualificationManifest`, `AlgorithmQualification`, `sanitize_payload`, schema version `1`

- [ ] **Step 1: Write schema validation tests**

```python
def test_manifest_requires_commit_wheel_and_device(valid_manifest):
    valid_manifest["git_commit"] = "a" * 40
    valid_manifest["wheel_sha256"] = "0" * 64
    valid_manifest["device"]["chip_id"] = "WK_C180"
    validate_manifest(valid_manifest)
```

- [ ] **Step 2: Write recursive secret-redaction tests**

```python
def test_sanitizer_removes_nested_credentials():
    payload = {"api_key": "abc", "nested": {"token": "def", "task_id": "task-1"}}
    clean = sanitize_payload(payload)
    assert "api_key" not in clean
    assert "token" not in clean["nested"]
    assert clean["nested"]["task_id"] == "task-1"
```

- [ ] **Step 3: Implement a closed schema**

In `test/release/conftest.py`, define dictionary-valued `valid_manifest` and string-valued `repo_commit` fixtures, plus `fake_runtime_service` and deterministic good/bad task-result fixtures shared by release tests. Define `fake_service_with_bad_result()` as a plain helper because the runner test constructs it directly. Require version, UTC timestamp, 40-character git commit, wheel filename/SHA-256, Python/pyqpanda3/qpanda3-runtime versions, device summary, calibration timestamp, and per-case algorithm name, execution mode, task IDs, shots, fixed threshold, raw-result digest, parsed result, and verdict. Reject unknown credential-shaped keys recursively. Export `contains_credentials(payload)` for runner tests.

- [ ] **Step 4: Run manifest and security tests**

Run: `python -m pytest test/release/test_qualification_manifest.py test/release/test_qualification_security.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/release_qualification test/release
git commit -m "feat: define runtime qualification manifest"
```

### Task 2: Define the fixed RC case inventory

**Files:**
- Create: `tools/release_qualification/cases.py`
- Create: `test/release/test_case_inventory.py`

**Interfaces:**
- Consumes: every public algorithm API
- Produces: `QUALIFICATION_CASES`, fixed shots/seeds/thresholds, circuit-only transpilation inventory

- [ ] **Step 1: Write completeness tests**

```python
EXECUTABLE_ALGORITHMS = {
    "QAOA", "QARM", "QKmeans", "QPCA", "QSVM", "QUBO_QAOA",
    "QUBO_GAS", "QAE", "QSVD", "QSVR", "Grover",
    "GroverAdaptiveSearch", "QmRMR", "QSEncode", "VQE", "HHL", "Shor",
}


def test_qpu_inventory_covers_every_executable_algorithm():
    assert {case.algorithm for case in QUALIFICATION_CASES} == EXECUTABLE_ALGORITHMS
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/release/test_case_inventory.py -v`

Expected: FAIL because the inventory is absent.

- [ ] **Step 3: Implement small fixed cases**

Each case declares builder, expected domain predicate, shots, confidence/tolerance, seed, maximum algorithm attempts, timeout, required capabilities, and whether FakeBackend execution is required. Shor uses modulus 15 and a base policy that does not encode factors/order. HHL records success probability and an observable rather than full tomography for QPU qualification.

- [ ] **Step 4: Verify cases build without credentials**

Run: `python -m pytest test/release/test_case_inventory.py -v`

Expected: PASS and every case produces a circuit/resource request locally.

- [ ] **Step 5: Commit**

```bash
git add tools/release_qualification/cases.py test/release/test_case_inventory.py
git commit -m "test: define fixed qpu qualification cases"
```

### Task 3: Implement FakeBackend preflight runner

**Files:**
- Create: `tools/release_qualification/run_preflight.py`
- Create: `test/release/test_preflight_runner.py`

**Interfaces:**
- Consumes: RuntimeService, explicit chip ID, `QUALIFICATION_CASES`
- Produces: transpilation/fake-execution records for the manifest

- [ ] **Step 1: Write a fake-service runner test**

```python
def test_preflight_records_transpile_and_fake_results(fake_runtime_service, tmp_path):
    result = run_preflight(fake_runtime_service, "WK_C180", output_dir=tmp_path)
    assert result.device["chip_id"] == "WK_C180"
    assert all(case.transpiled for case in result.cases)
    assert not contains_credentials(result.to_dict())
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/release/test_preflight_runner.py -v`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement main-guarded preflight execution**

The script accepts `--chip-id`, `--wheel`, and `--output-dir`; it obtains the API key only from `QPANDA3_API_KEY`, logs in, queries device configuration, constructs FakeBackend, and runs each case under a real `if __name__ == "__main__":` entry. It writes only sanitized JSON and raw-result SHA-256 digests.

- [ ] **Step 4: Run credential-free runner tests**

Run: `python -m pytest test/release/test_preflight_runner.py -v`

Expected: PASS with the fake service.

- [ ] **Step 5: Commit**

```bash
git add tools/release_qualification/run_preflight.py test/release/test_preflight_runner.py
git commit -m "feat: add fake backend preflight runner"
```

### Task 4: Implement real-QPU runner and verdict calculation

**Files:**
- Create: `tools/release_qualification/run_qpu.py`
- Create: `tools/release_qualification/verdicts.py`
- Create: `test/release/test_qpu_runner.py`
- Create: `test/release/test_verdicts.py`

**Interfaces:**
- Consumes: preflight record, runtime backend, fixed cases
- Produces: task IDs, parsed results, predetermined statistical verdicts

- [ ] **Step 1: Write deterministic verdict tests**

```python
def test_bell_case_passes_fixed_threshold():
    verdict = bell_verdict({"00": 490, "11": 480, "01": 15, "10": 15}, shots=1000)
    assert verdict.passed is True


def test_verdict_does_not_retry_after_threshold_failure():
    runner = QPURunner(service=fake_service_with_bad_result())
    runner.run_case(case_by_name("bell"))
    assert runner.submission_count == 1
```

Import `case_by_name` from `tools.release_qualification.cases`; it returns the already committed case object and must not construct alternative thresholds at test time.

- [ ] **Step 2: Write Shor provenance and HHL-output tests**

Assert the QPU Shor record has `used_quantum=True`, nonempty task IDs, and factors derived after order recovery. Assert HHL qualification records success probability and its requested observable without requiring a state vector.

- [ ] **Step 3: Implement sequential, resumable QPU execution**

Run cases one at a time, checkpoint after each remote task, and resume by querying existing task IDs. A task submission failure is recorded as failure and is not automatically retried. A status-query network failure uses bounded backoff. Verdict code consumes only the committed case threshold.

- [ ] **Step 4: Run runner/verdict tests with stubs**

Run: `python -m pytest test/release/test_qpu_runner.py test/release/test_verdicts.py -v`

Expected: PASS without a real device.

- [ ] **Step 5: Commit**

```bash
git add tools/release_qualification test/release
git commit -m "feat: add real qpu qualification runner"
```

### Task 5: Add manual RC workflow and release policy gate

**Files:**
- Create: `.github/workflows/runtime-rc.yml`
- Create: `tools/release_qualification/check_release.py`
- Modify: `.github/workflows/main.yml`
- Modify: `.gitignore`
- Create: `release/2.1.0/README.md`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `Tutorials/source/Changelog.rst`
- Create: `test/release/test_release_policy.py`

**Interfaces:**
- Consumes: candidate wheel, manifest schema, QPU manifest
- Produces: qualification artifact required by the release job

- [ ] **Step 1: Write release-policy failure tests**

```python
def test_release_rejects_manifest_for_other_commit(valid_manifest, repo_commit):
    valid_manifest["git_commit"] = "0" * 40
    with pytest.raises(ReleasePolicyError, match="commit"):
        check_release(valid_manifest, expected_commit=repo_commit)


def test_release_requires_every_case_to_pass(valid_manifest):
    valid_manifest["cases"][0]["verdict"] = "failed"
    with pytest.raises(ReleasePolicyError, match="failed"):
        check_release(valid_manifest, expected_commit=valid_manifest["git_commit"])
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/release/test_release_policy.py -v`

Expected: FAIL because the policy checker is absent.

- [ ] **Step 3: Implement workflow_dispatch qualification**

Inputs are candidate artifact/run ID and chip ID. The job downloads the exact wheel, verifies SHA-256, installs `[runtime]`, runs preflight, runs QPU cases, validates/sanitizes the manifest, and uploads it for release review. The workflow never echoes `QPANDA3_API_KEY`.

Modify the tag release job to require a manifest whose commit, wheel digest, version, device record, and case verdicts match the tag candidate.

Before running qualification, finalize the root READMEs and Changelog for the candidate. Describe `qpanda3-runtime` installation, explicit service/device injection, small-scale guarantees, capability errors, and the mandatory QPU release gate without claiming a particular device has passed. `release/2.1.0/README.md` documents how to retrieve and verify the immutable workflow/Release manifest. Add `.artifacts/` to `.gitignore`; qualification output is never committed because committing it would change the commit the manifest attests to.

- [ ] **Step 4: Validate workflows and policy**

Run: `python -m pytest test/release -q`

Expected: PASS.

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/runtime-rc.yml', encoding='utf-8'))"`

Expected: exits zero.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows .gitignore tools/release_qualification test/release release/2.1.0 README.md README_EN.md Tutorials/source/Changelog.rst
git commit -m "ci: require runtime qualification for release"
```

### Task 6: Produce and validate the 2.1.0 RC evidence

**Files:**
- Create at execution time, ignored: `.artifacts/2.1.0/runtime-qualification.json`
- Modify: no tracked files

**Interfaces:**
- Consumes: successful manual runtime-rc workflow artifact
- Produces: immutable sanitized workflow/Release evidence for the exact candidate commit and wheel

- [ ] **Step 1: Run the manual RC workflow against the selected device**

Use the GitHub Actions `runtime-rc` workflow with the exact candidate wheel and explicit chip ID. Do not edit the downloaded manifest.

- [ ] **Step 2: Validate the artifact locally**

Run:

```bash
python tools/release_qualification/check_release.py \
  --manifest .artifacts/2.1.0/runtime-qualification.json \
  --commit "$(git rev-parse HEAD)" \
  --wheel pyqpanda-algorithm/dist/pyqpanda_alg-2.1.0-py3-none-any.whl
```

Expected: PASS with every fixed case qualified.

- [ ] **Step 3: Perform a secret scan**

Run: `rg -n -i "api[_-]?key|access[_-]?token|authorization|bearer|password" .artifacts/2.1.0`

Expected: no matches in the JSON artifact.

- [ ] **Step 4: Review the immutable evidence**

Confirm the manifest lists the selected device, calibration timestamp, candidate commit, wheel digest, complete algorithm inventory, known resource limitations, and all task IDs/verdicts. Do not copy raw service responses into documentation.

- [ ] **Step 5: Preserve evidence without changing the candidate**

Retain the manifest in the successful protected workflow run and attach the same SHA-256-verified file to the GitHub Release. Record the workflow run URL in the operator's release checklist. Run `git status --short` and require no tracked changes; do not commit the manifest, because the release policy binds it to the current candidate commit.

### Task 7: Run the final release gate

**Files:**
- Modify only if a gate exposes a defect in the owning plan

**Interfaces:**
- Consumes: all seven implementation plans
- Produces: a releasable 2.1.0 commit; tag creation remains an explicit operator action

- [ ] **Step 1: Build the final wheel and sdist from a clean tree**

Run: `python -m build pyqpanda-algorithm`

Expected: version 2.1.0 sdist and `py3-none-any` wheel.

- [ ] **Step 2: Run the credential-free suite**

Run: `python -m pytest test -q`

Expected: PASS with QPU-marked tests skipped only because standard CI has no credentials; runtime contract tests pass.

- [ ] **Step 3: Build documentation with warnings as errors**

Run: `python -m sphinx -W -b html Tutorials/source Tutorials/build/html`

Expected: PASS.

- [ ] **Step 4: Validate evidence against the final commit and wheel**

Run the exact `check_release.py` command from Task 6.

Expected: PASS. If commit or wheel changed after qualification, create a new candidate and rerun runtime RC; never edit identifiers by hand.

- [ ] **Step 5: Commit gate-only corrections when present**

```bash
git add pyqpanda-algorithm test Tutorials .github tools release
git commit -m "chore: finalize 2.1.0 release candidate"
```
