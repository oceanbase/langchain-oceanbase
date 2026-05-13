from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from langchain_oceanbase.ai_functions import OceanBaseAIFunctions


def _model_names(prefix: str) -> dict[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return {
        "embed_model": f"{prefix}_embed_{suffix}",
        "complete_model": f"{prefix}_complete_{suffix}",
        "embed_endpoint": f"{prefix}_embed_endpoint_{suffix}",
        "complete_endpoint": f"{prefix}_complete_endpoint_{suffix}",
    }


def _provider_config_from_env() -> dict[str, str] | None:
    required = {
        "url": os.getenv("OB_AI_TEST_URL", "").strip(),
        "access_key": os.getenv("OB_AI_TEST_ACCESS_KEY", "").strip(),
        "embed_model": os.getenv("OB_AI_TEST_EMBED_MODEL", "").strip(),
        "complete_model": os.getenv("OB_AI_TEST_COMPLETE_MODEL", "").strip(),
    }
    if not all(required.values()):
        return None

    return {
        **required,
        "provider": os.getenv("OB_AI_TEST_PROVIDER", "openai").strip() or "openai",
    }


def _row_value(row: dict, *keys: str):
    normalized = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        lowered = key.lower()
        if lowered in normalized:
            return normalized[lowered]
    return None


def _drop_endpoint_if_exists(
    ai_functions: OceanBaseAIFunctions,
    endpoint_name: str,
) -> None:
    try:
        ai_functions.drop_ai_model_endpoint(endpoint_name)
    except Exception:
        pass


def _drop_model_if_exists(
    ai_functions: OceanBaseAIFunctions,
    model_name: str,
) -> None:
    try:
        ai_functions.drop_ai_model(model_name)
    except Exception:
        pass


@pytest.fixture
def ai_functions(
    ai_functions_connection_args: dict[str, str],
) -> OceanBaseAIFunctions:
    return OceanBaseAIFunctions(connection_args=ai_functions_connection_args)


@pytest.fixture
def created_ai_models(ai_functions: OceanBaseAIFunctions) -> Iterator[dict[str, str]]:
    names = _model_names("ci_lifecycle")
    _drop_endpoint_if_exists(ai_functions, names["embed_endpoint"])
    _drop_endpoint_if_exists(ai_functions, names["complete_endpoint"])
    _drop_model_if_exists(ai_functions, names["embed_model"])
    _drop_model_if_exists(ai_functions, names["complete_model"])

    ai_functions.create_ai_model(
        model_name=names["embed_model"],
        model_type="dense_embedding",
    )
    ai_functions.create_ai_model(
        model_name=names["complete_model"],
        model_type="completion",
    )

    try:
        yield names
    finally:
        _drop_endpoint_if_exists(ai_functions, names["embed_endpoint"])
        _drop_endpoint_if_exists(ai_functions, names["complete_endpoint"])
        _drop_model_if_exists(ai_functions, names["embed_model"])
        _drop_model_if_exists(ai_functions, names["complete_model"])


@pytest.fixture
def lifecycle_ai_endpoints(
    ai_functions: OceanBaseAIFunctions,
    created_ai_models: dict[str, str],
) -> Iterator[dict[str, str]]:
    ai_functions.create_ai_model_endpoint(
        endpoint_name=created_ai_models["embed_endpoint"],
        ai_model_name=created_ai_models["embed_model"],
        url="https://api.example.com/v1",
        access_key="placeholder-key",
        provider="openai",
    )
    ai_functions.create_ai_model_endpoint(
        endpoint_name=created_ai_models["complete_endpoint"],
        ai_model_name=created_ai_models["complete_model"],
        url="https://api.example.com/v1",
        access_key="placeholder-key",
        provider="openai",
    )

    yield created_ai_models


@pytest.fixture
def live_provider_config() -> dict[str, str]:
    config = _provider_config_from_env()
    if config is None:
        pytest.skip(
            "Set OB_AI_TEST_URL, OB_AI_TEST_ACCESS_KEY, "
            "OB_AI_TEST_EMBED_MODEL, and OB_AI_TEST_COMPLETE_MODEL "
            "to run provider-backed AI function tests."
        )
    return config


@pytest.fixture
def live_ai_models(
    ai_functions: OceanBaseAIFunctions,
    live_provider_config: dict[str, str],
) -> Iterator[dict[str, str]]:
    names = _model_names("ci_live")
    _drop_endpoint_if_exists(ai_functions, names["embed_endpoint"])
    _drop_endpoint_if_exists(ai_functions, names["complete_endpoint"])
    _drop_model_if_exists(ai_functions, names["embed_model"])
    _drop_model_if_exists(ai_functions, names["complete_model"])

    ai_functions.create_ai_model(
        model_name=names["embed_model"],
        model_type="dense_embedding",
        provider_model_name=live_provider_config["embed_model"],
    )
    ai_functions.create_ai_model(
        model_name=names["complete_model"],
        model_type="completion",
        provider_model_name=live_provider_config["complete_model"],
    )

    ai_functions.create_ai_model_endpoint(
        endpoint_name=names["embed_endpoint"],
        ai_model_name=names["embed_model"],
        url=live_provider_config["url"],
        access_key=live_provider_config["access_key"],
        provider=live_provider_config["provider"],
    )
    ai_functions.create_ai_model_endpoint(
        endpoint_name=names["complete_endpoint"],
        ai_model_name=names["complete_model"],
        url=live_provider_config["url"],
        access_key=live_provider_config["access_key"],
        provider=live_provider_config["provider"],
    )

    try:
        yield names
    finally:
        _drop_endpoint_if_exists(ai_functions, names["embed_endpoint"])
        _drop_endpoint_if_exists(ai_functions, names["complete_endpoint"])
        _drop_model_if_exists(ai_functions, names["embed_model"])
        _drop_model_if_exists(ai_functions, names["complete_model"])


def test_create_and_list_ai_models(
    ai_functions: OceanBaseAIFunctions,
    created_ai_models: dict[str, str],
) -> None:
    models = ai_functions.list_ai_models()
    model_names = {_row_value(model, "model_name") for model in models}

    assert created_ai_models["embed_model"] in model_names
    assert created_ai_models["complete_model"] in model_names


def test_create_list_alter_and_drop_ai_model_endpoints(
    ai_functions: OceanBaseAIFunctions,
    lifecycle_ai_endpoints: dict[str, str],
) -> None:
    endpoints = ai_functions.list_ai_model_endpoints()
    endpoint_names = {_row_value(endpoint, "endpoint_name") for endpoint in endpoints}

    assert lifecycle_ai_endpoints["embed_endpoint"] in endpoint_names
    assert lifecycle_ai_endpoints["complete_endpoint"] in endpoint_names

    ai_functions.alter_ai_model_endpoint(
        endpoint_name=lifecycle_ai_endpoints["complete_endpoint"],
        ai_model_name=lifecycle_ai_endpoints["complete_model"],
        url="https://new-api.example.com/v1",
        access_key="updated-placeholder-key",
        provider="openai",
    )

    refreshed = ai_functions.list_ai_model_endpoints()
    altered = next(
        endpoint
        for endpoint in refreshed
        if _row_value(endpoint, "endpoint_name")
        == lifecycle_ai_endpoints["complete_endpoint"]
    )

    assert _row_value(altered, "url") == "https://new-api.example.com/v1"
    assert _row_value(altered, "provider") == "openai"

    ai_functions.drop_ai_model_endpoint(lifecycle_ai_endpoints["complete_endpoint"])
    remaining = ai_functions.list_ai_model_endpoints()
    remaining_names = {_row_value(endpoint, "endpoint_name") for endpoint in remaining}
    assert lifecycle_ai_endpoints["complete_endpoint"] not in remaining_names


def test_ai_complete_with_live_provider(
    ai_functions: OceanBaseAIFunctions,
    live_ai_models: dict[str, str],
) -> None:
    completion = ai_functions.ai_complete(
        prompt="Explain what machine learning is in one sentence.",
        model_name=live_ai_models["complete_model"],
    )

    assert isinstance(completion, str)
    assert completion.strip()


def test_ai_embed_with_live_provider(
    ai_functions: OceanBaseAIFunctions,
    live_ai_models: dict[str, str],
) -> None:
    embedding = ai_functions.ai_embed(
        text="Machine learning is a subset of artificial intelligence.",
        model_name=live_ai_models["embed_model"],
    )

    assert isinstance(embedding, list)
    assert embedding
    assert all(isinstance(value, float) for value in embedding)


def test_ai_rerank_with_live_provider(
    ai_functions: OceanBaseAIFunctions,
    live_ai_models: dict[str, str],
) -> None:
    reranked = ai_functions.ai_rerank(
        query="machine learning algorithms",
        documents=[
            "Deep learning uses multi-layer neural networks.",
            "Python is widely used in data science workflows.",
            "Supervised learning trains on labeled datasets.",
        ],
        model_name=live_ai_models["embed_model"],
        top_k=2,
    )

    assert isinstance(reranked, list)
    assert reranked
    assert len(reranked) <= 2
    assert all("rank" in result and "score" in result for result in reranked)
