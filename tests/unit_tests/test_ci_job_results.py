"""The aggregate check must never hide an unsuccessful prerequisite job."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/check_job_results.py"
RESULT_VARIABLES = (
    "DEPENDENCIES_RESULT",
    "LINT_RESULT",
    "TEST_RESULT",
    "EMBEDDING_RESULT",
)


def run_gate(results: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key not in RESULT_VARIABLES}
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**env, **results},
        capture_output=True,
        text=True,
        check=False,
    )


def test_gate_accepts_all_successful_jobs() -> None:
    result = run_gate(dict.fromkeys(RESULT_VARIABLES, "success"))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("variable", RESULT_VARIABLES)
@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", None])
def test_gate_rejects_each_unsuccessful_job(
    variable: str, outcome: str | None
) -> None:
    results = dict.fromkeys(RESULT_VARIABLES, "success")
    if outcome is None:
        del results[variable]
    else:
        results[variable] = outcome

    result = run_gate(results)

    assert result.returncode == 1
    assert "::error::Required job" in result.stdout
    assert (outcome or "missing") in result.stdout
