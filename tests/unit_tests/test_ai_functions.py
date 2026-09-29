"""Exercise rerank fallback without a database or model endpoint."""

from unittest.mock import Mock

import pytest

from langchain_oceanbase.ai_functions import OceanBaseAIFunctions
from langchain_oceanbase.exceptions import OceanBaseConfigurationError


@pytest.fixture
def ai_functions() -> OceanBaseAIFunctions:
    return OceanBaseAIFunctions.__new__(OceanBaseAIFunctions)


@pytest.mark.parametrize("batch_result", [None, RuntimeError("batch unavailable")])
@pytest.mark.parametrize("top_k", [None, 1])
def test_rerank_falls_back_and_preserves_ordering(
    ai_functions: OceanBaseAIFunctions,
    batch_result: object,
    top_k: int | None,
) -> None:
    execute = Mock(side_effect=[batch_result, 0.25, 0.9])
    ai_functions._execute_sql = execute

    result = ai_functions.ai_rerank(
        "query", ["first", "second"], model_name="reranker", top_k=top_k
    )

    expected = [
        {"document": "second", "score": 0.9, "rank": 1},
        {"document": "first", "score": 0.25, "rank": 2},
    ]
    assert result == (expected if top_k is None else expected[:top_k])
    assert execute.call_count == 3


def test_rerank_returns_empty_list_when_all_results_are_null(
    ai_functions: OceanBaseAIFunctions,
) -> None:
    execute = Mock(side_effect=[None, None, None])
    ai_functions._execute_sql = execute

    assert ai_functions.ai_rerank(
        "query", ["first", "second"], model_name="reranker"
    ) == []
    assert execute.call_count == 3


def test_rerank_uses_successful_batch_without_fallback(
    ai_functions: OceanBaseAIFunctions,
) -> None:
    execute = Mock(return_value="[0.25, 0.9]")
    ai_functions._execute_sql = execute

    assert ai_functions.ai_rerank(
        "query", ["first", "second"], model_name="reranker", top_k=1
    ) == [{"document": "second", "score": 0.9, "rank": 1}]
    execute.assert_called_once()


def test_rerank_empty_input_and_missing_model_do_not_execute_sql(
    ai_functions: OceanBaseAIFunctions,
) -> None:
    execute = Mock()
    ai_functions._execute_sql = execute

    assert ai_functions.ai_rerank("query", []) == []
    with pytest.raises(OceanBaseConfigurationError, match="model_name is required"):
        ai_functions.ai_rerank("query", ["first"])
    execute.assert_not_called()
