from __future__ import annotations

import pytest

from tests.integration_tests._backend_utils import skip_embedded_seekdb_capacity_error


def test_skip_embedded_seekdb_capacity_error_skips_embedded_index_exhaustion() -> None:
    with pytest.raises(pytest.skip.Exception):
        skip_embedded_seekdb_capacity_error(
            RuntimeError("execute sql failed 5703 Add index failed"),
            backend="embedded-seekdb",
            operation="creating vector index",
        )
