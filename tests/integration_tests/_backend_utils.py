from __future__ import annotations

import os
import uuid

import pytest


def embedded_seekdb_runtime_available() -> bool:
    try:
        import pylibseekdb  # noqa: F401
        import pyseekdb  # noqa: F401
    except ImportError:
        return False
    return True


def ci_db_type() -> str:
    return os.getenv("OB_CI_DB_TYPE", "").strip().lower()


def external_db_env_configured() -> bool:
    return any(
        os.getenv(env_var)
        for env_var in (
            "SEEKDB_HOST",
            "SEEKDB_PORT",
            "SEEKDB_USER",
            "SEEKDB_PASSWORD",
            "SEEKDB_DB",
            "OB_HOST",
            "OB_PORT",
            "OB_USER",
            "OB_PASSWORD",
            "OB_DB",
        )
    )


def use_embedded_seekdb() -> bool:
    db_type = ci_db_type()
    if db_type:
        return db_type == "embedded-seekdb"
    if external_db_env_configured():
        return False
    return embedded_seekdb_runtime_available()


def server_connection_args_from_env() -> dict[str, str]:
    return {
        "host": os.getenv("SEEKDB_HOST") or os.getenv("OB_HOST", "127.0.0.1"),
        "port": os.getenv("SEEKDB_PORT") or os.getenv("OB_PORT", "2881"),
        "user": os.getenv("SEEKDB_USER") or os.getenv("OB_USER", "root@test"),
        "password": os.getenv("SEEKDB_PASSWORD") or os.getenv("OB_PASSWORD", ""),
        "db_name": os.getenv("SEEKDB_DB") or os.getenv("OB_DB", "test"),
    }


def embedded_connection_args() -> dict[str, str]:
    return {
        "db_name": os.getenv("SEEKDB_DB") or os.getenv("OB_DB", "test"),
    }


def unique_table_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def is_embedded_seekdb_capacity_error(exc: Exception) -> bool:
    message = str(exc)
    return "execute sql failed 5703 Add index failed" in message or (
        "execute sql failed 4184 Server out of disk space" in message
    )


def skip_embedded_seekdb_capacity_error(
    exc: Exception,
    *,
    backend: str,
    operation: str,
) -> None:
    if backend == "embedded-seekdb" and is_embedded_seekdb_capacity_error(exc):
        pytest.skip(f"embedded SeekDB capacity exceeded while {operation}: {exc}")
