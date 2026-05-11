"""Tests for the legacy checkpoint saver surface."""

from __future__ import annotations

import pytest

from langchain_oceanbase.checkpoint.saver import OceanBaseSaver


def test_legacy_oceanbase_saver_emits_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OceanBaseSaver should warn users to migrate to OceanBaseCheckpointSaver."""
    monkeypatch.setattr(OceanBaseSaver, "_create_client", lambda self, **_: None)
    monkeypatch.setattr(
        OceanBaseSaver, "_create_tables_if_not_exists", lambda self: None
    )

    with pytest.warns(DeprecationWarning, match="OceanBaseCheckpointSaver"):
        OceanBaseSaver(connection_args={})
