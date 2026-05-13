from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.integration_tests._backend_utils import (
    ci_db_type,
    embedded_connection_args,
    embedded_seekdb_runtime_available,
    external_db_env_configured,
    server_connection_args_from_env,
    use_embedded_seekdb,
)


@pytest.fixture
def embedded_seekdb_path(tmp_path: Path) -> str:
    if ci_db_type() == "mysql":
        pytest.skip("Embedded SeekDB fixtures are not used in the mysql CI matrix.")
    if not embedded_seekdb_runtime_available():
        pytest.skip(
            "embedded SeekDB requires pylibseekdb (e.g. pip install 'pyseekdb>=1.2' "
            "or pip install 'pyobvector[pyseekdb]')"
        )

    root = tmp_path / f"seekdb_root_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield str(root / "seekdb_data")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def integration_backend_name() -> str:
    if use_embedded_seekdb():
        return "embedded-seekdb"

    db_type = ci_db_type()
    return db_type or "external-server"


@pytest.fixture
def integration_connection_args() -> dict[str, str]:
    if use_embedded_seekdb():
        return embedded_connection_args()
    return server_connection_args_from_env()


@pytest.fixture
def integration_client_kwargs(
    request: pytest.FixtureRequest,
) -> dict[str, Any]:
    if use_embedded_seekdb():
        return {"path": request.getfixturevalue("embedded_seekdb_path")}
    return {}


@pytest.fixture
def default_vector_index_type() -> str:
    return "FLAT" if use_embedded_seekdb() else "HNSW"


@pytest.fixture
def seekdb_capable_backend(integration_backend_name: str) -> str:
    if integration_backend_name == "mysql":
        pytest.skip("This test suite is not compatible with the mysql CI matrix.")
    return integration_backend_name


@pytest.fixture
def ai_functions_connection_args() -> dict[str, str]:
    db_type = ci_db_type()
    if db_type and db_type != "seekdb":
        pytest.skip("AI functions CI currently runs only in the seekdb server matrix.")
    if use_embedded_seekdb():
        pytest.skip("AI functions are not wired for embedded SeekDB in CI.")
    if not db_type and not external_db_env_configured():
        pytest.skip("Configure a server-backed SeekDB or OceanBase environment to run AI function tests.")
    return server_connection_args_from_env()
