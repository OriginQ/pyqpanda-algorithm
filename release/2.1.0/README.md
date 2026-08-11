# pyqpanda-algorithm 2.1.0 — Release Qualification

This directory documents how the 2.1.0 release candidate is qualified
and how the immutable qualification evidence is retrieved and
verified.  The qualification gate is mandatory: no GitHub Release can
be created without a valid qualification manifest attesting the exact
candidate commit and wheel.

## The release gate

Release creation (the `release` job in `.github/workflows/main.yml`,
on tag pushes) requires the `runtime-qualification-2.1.0` artifact
produced by the manual `Runtime RC Qualification` workflow
(`.github/workflows/runtime-rc.yml`).  The gate runs
`tools/release_qualification/check_release.py`, which fails unless the
manifest:

* is schema-valid (`tools/release_qualification/schema.json`, closed:
  unknown keys are rejected at every level);
* attests the exact commit the tag points at;
* records a wheel whose filename attests version `2.1.0` and whose
  SHA-256 digest matches the wheel built by CI;
* carries a device record (`chip_id`, `summary`,
  `calibration_timestamp`); and
* records **every** fixed case verdict as `passed` — a device outage
  postpones qualification but can never become a passing result.

The manifest never contains credentials: it is sanitized
(`sanitize_payload` in `tools/release_qualification/manifest.py`)
before it is written, raw service responses are stored only as
SHA-256 digests, and `QPANDA3_API_KEY` exists only in the workflow's
step environment.

## Running the manual RC workflow

1. Push the candidate commit and confirm the standard CI run produced
   the `pyqpanda_alg-wheel` artifact (note the run ID).
2. In Actions, dispatch **Runtime RC Qualification** with:
   * **wheel_artifact_run_id** — the CI run ID from step 1;
   * **chip_id** — the explicitly selected device, e.g. `WK_C180`;
   * **expected_wheel_sha256** (optional) — the wheel's SHA-256, so the
     job fails loudly if the downloaded wheel differs.
3. The workflow downloads the exact wheel, verifies its SHA-256,
   installs `pyqpanda_alg[runtime]`, runs the fixed preflight
   (FakeBackend) and real-QPU qualification cases, validates and
   sanitizes the manifest, and uploads it as the
   `runtime-qualification-2.1.0` artifact.
4. A failed case verdict or a credential-shaped value fails the
   workflow; do not treat a failed run as qualification evidence.

## Retrieving the immutable manifest

The manifest is immutable because the release gate binds it to the
commit that produced it — qualification output is never committed,
since committing it would change the commit the manifest attests to.

1. Actions → **Runtime RC Qualification** → the successful run → the
   `runtime-qualification-2.1.0` artifact (`manifest.json`).
2. Record the workflow run URL in the operator's release checklist;
   the protected run is the authoritative source of the evidence.
3. The same file is attached to the GitHub Release once the tag
   passes the gate, so the release asset and the workflow artifact can
   be cross-checked byte for byte.

## Verifying the manifest locally

```bash
# 1. The manifest is valid for the exact candidate commit and wheel.
python tools/release_qualification/check_release.py \
  --manifest .artifacts/2.1.0/manifest.json \
  --commit "$(git rev-parse HEAD)" \
  --wheel pyqpanda-algorithm/dist/pyqpanda_alg-2.1.0-py3-none-any.whl
# Expected: release policy: PASS

# 2. The artifact is credential-free.
rg -n -i "api[_-]?key|access[_-]?token|authorization|bearer|password" .artifacts/2.1.0
# Expected: no matches

# 3. The downloaded artifact matches the release asset.
sha256sum .artifacts/2.1.0/manifest.json
```

## What the manifest attests

The manifest pins the qualified commit and wheel, the explicitly
selected device and its calibration snapshot, and per-case records:
algorithm, execution mode, task IDs, shots, the fixed threshold, the
raw-result digest, the parsed result, and the verdict.  These records
are the evidence that the fixed small-scale qualification cases passed
on the named device at the recorded time; they do not claim any
particular device has passed more than what the manifest records.
