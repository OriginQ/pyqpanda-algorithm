"""Deterministic release-policy gate for the 2.1.0 candidate.

The release gate treats the sanitized qualification manifest as the
only durable evidence a candidate is releasable.  :func:`check_release`
is the deterministic policy layer that release creation runs: it binds
the manifest to the exact candidate commit, wheel file (by SHA-256),
and release version, and requires every fixed case verdict to be
``passed``.  Schema conformance -- including the device record
(``chip_id``/``summary``/``calibration_timestamp``), identifier shapes,
and the verdict enum -- is enforced first by
:func:`tools.release_qualification.manifest.validate_manifest`, so the
policy layer adds only the release-specific bindings.

Version attestation
-------------------
The manifest's ``version`` field is the *schema* version (``"1"``); the
package release version is single-sourced in the wheel filename, so the
policy verifies the version through the wheel name recorded in the
manifest (``pyqpanda_alg-{expected_version}-...whl``) rather than the
schema-version field.

Credentials never pass through this module: it reads only the sanitized
manifest artifact and raises :class:`ReleasePolicyError` (or
:class:`ManifestValidationError`) on any violation, and the CLI prints
only a verdict -- never a credential or a raw response.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a plain script: make the repo importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.release_qualification.manifest import (  # noqa: E402
    ManifestValidationError,
    validate_manifest,
)


class ReleasePolicyError(ValueError):
    """Raised when a manifest fails a release-policy binding."""


def check_release(
    manifest_dict: dict,
    expected_commit: str,
    wheel_path: str | Path | None = None,
    expected_version: str = "2.1.0",
) -> dict:
    """Verify ``manifest_dict`` authorizes a release of the candidate.

    The manifest must be schema-valid, attest the exact
    ``expected_commit``, carry a wheel whose filename attests
    ``expected_version``, digest-match the ``wheel_path`` file when one
    is given, and record every case verdict as ``passed``.  Returns the
    validated manifest on success; raises :class:`ReleasePolicyError`
    (or :class:`ManifestValidationError` for schema violations) on the
    first violation.
    """
    validate_manifest(manifest_dict)

    if manifest_dict["git_commit"] != expected_commit:
        raise ReleasePolicyError(
            f"release policy violation: manifest git_commit "
            f"{manifest_dict['git_commit']} does not match the expected commit "
            f"{expected_commit}"
        )

    wheel_name = manifest_dict["wheel"]
    version_prefix = f"pyqpanda_alg-{expected_version}-"
    if not wheel_name.startswith(version_prefix):
        raise ReleasePolicyError(
            f"release policy violation: manifest wheel {wheel_name!r} does not "
            f"attest version {expected_version!r}"
        )

    if wheel_path is not None:
        path = Path(wheel_path)
        if wheel_name != path.name:
            raise ReleasePolicyError(
                f"release policy violation: manifest wheel {wheel_name!r} does not "
                f"match the candidate wheel file {path.name!r}"
            )
        actual_digest = _sha256_file(path)
        if manifest_dict["wheel_sha256"] != actual_digest:
            raise ReleasePolicyError(
                f"release policy violation: manifest wheel_sha256 "
                f"{manifest_dict['wheel_sha256']} does not match the candidate "
                f"wheel {path} ({actual_digest})"
            )

    failed = [
        case["algorithm"]
        for case in manifest_dict["cases"]
        if case["verdict"] != "passed"
    ]
    if failed:
        raise ReleasePolicyError(
            f"release policy violation: {len(failed)} case(s) with a failed "
            f"verdict: {', '.join(failed)}; every case must pass"
        )
    return manifest_dict


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """CLI gate: ``--manifest``, ``--commit``, optional ``--wheel``.

    Exits zero with a PASS verdict when the manifest attests the exact
    candidate; exits non-zero with a FAIL verdict on stderr otherwise.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic release-policy gate: verify a qualification manifest "
            "attests the exact candidate commit and wheel with every case passed."
        )
    )
    parser.add_argument(
        "--manifest", required=True, help="path to the sanitized qualification manifest JSON"
    )
    parser.add_argument(
        "--commit", required=True, help="expected 40-character git commit of the candidate"
    )
    parser.add_argument(
        "--wheel", default=None, help="path to the candidate wheel to digest-verify"
    )
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"release policy: FAIL: cannot read manifest {args.manifest}: {exc}")
    try:
        check_release(payload, expected_commit=args.commit, wheel_path=args.wheel)
    except (ManifestValidationError, ReleasePolicyError) as exc:
        print(f"release policy: FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    print("release policy: PASS: manifest attests the candidate commit and wheel")


if __name__ == "__main__":
    main()
