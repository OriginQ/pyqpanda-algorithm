"""Release policy gate: commit binding and all-passed verdicts.

The release-policy checker is the deterministic gate that release
creation runs: a manifest attesting any other commit is rejected, and
a single failed case verdict blocks the release.  Both tests below are
verbatim from the task brief.
"""

import pytest

from tools.release_qualification.check_release import (
    ReleasePolicyError,
    check_release,
)


def test_release_rejects_manifest_for_other_commit(valid_manifest, repo_commit):
    valid_manifest["git_commit"] = "0" * 40
    with pytest.raises(ReleasePolicyError, match="commit"):
        check_release(valid_manifest, expected_commit=repo_commit)


def test_release_requires_every_case_to_pass(valid_manifest):
    valid_manifest["cases"][0]["verdict"] = "failed"
    with pytest.raises(ReleasePolicyError, match="failed"):
        check_release(valid_manifest, expected_commit=valid_manifest["git_commit"])
