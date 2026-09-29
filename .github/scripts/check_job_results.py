"""Make the aggregate CI check reflect failed, cancelled, or skipped jobs."""

import os
import sys


def main() -> int:
    results = {
        name: os.environ.get(variable, "missing")
        for name, variable in (
            ("dependencies", "DEPENDENCIES_RESULT"),
            ("lint", "LINT_RESULT"),
            ("test", "TEST_RESULT"),
            ("embedding", "EMBEDDING_RESULT"),
        )
    }
    failed = {name: result for name, result in results.items() if result != "success"}
    if failed:
        for name, result in failed.items():
            print(f"::error::Required job {name} did not succeed: {result}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
