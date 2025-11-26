"""OceanBase AI functions support.

This module provides support for AI functions in OceanBase 4.4.1+ and SeekDB,
including AI_EMBED, AI_COMPLETE, and AI_RERANK functions.

References:
    - https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000004018305
    - https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000004018306
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pyobvector import ObVecClient
from sqlalchemy import text

from langchain_oceanbase.vectorstores import DEFAULT_OCEANBASE_CONNECTION

logger = logging.getLogger(__name__)


def _parse_version(version_str: str) -> Optional[tuple[int, int, int]]:
    """Parse version string to (major, minor, patch) tuple.

    Args:
        version_str: Version string to parse.

    Returns:
        Tuple of (major, minor, patch) or None if parsing fails.
    """
    # Try multiple regex patterns to match version numbers
    patterns = [
        r"[Oo]cean[Bb]ase[-\s]*[vV]?(\d+)\.(\d+)\.(\d+)",
        r"[vV](\d+)\.(\d+)\.(\d+)",
        r"(\d+)\.(\d+)\.(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, version_str)
        if match:
            try:
                return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except (ValueError, IndexError):
                continue

    # Fallback: try simple split
    parts = version_str.split(".")
    if len(parts) >= 3:
        try:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            pass

    return None


def _is_version_supported(major: int, minor: int, patch: int) -> bool:
    """Check if version supports AI functions.

    AI functions are supported in OceanBase 4.4.1+.

    Args:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.

    Returns:
        True if version supports AI functions.
    """
    return (
        major > 4
        or (major == 4 and minor > 4)
        or (major == 4 and minor == 4 and patch >= 1)
    )


def _check_ai_function_support(obvector: ObVecClient) -> bool:
    """Check if the OceanBase instance supports AI functions.

    AI functions are supported in:
    - OceanBase 4.4.1 and later versions
    - SeekDB

    This function references pyobvector's version checking logic.
    See: https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000004018305

    Args:
        obvector: ObVecClient instance connected to OceanBase/SeekDB

    Returns:
        bool: True if AI functions are supported, False otherwise
    """
    try:
        if hasattr(obvector, "_is_seekdb") and obvector._is_seekdb():
            return True

        with obvector.engine.connect() as conn:
            result = conn.execute(text("SELECT VERSION()"))
            version_str = result.scalar()
            if not version_str:
                return False

            version_tuple = _parse_version(str(version_str))
            if version_tuple:
                major, minor, patch = version_tuple
                return _is_version_supported(major, minor, patch)

            logger.warning(f"Failed to parse version string: {version_str}")
        return False
    except Exception as e:
        logger.warning(f"Failed to check AI function support: {e}")
        return False


class OceanBaseAIFunctions:
    """OceanBase AI functions integration.

    This class provides access to AI functions available in OceanBase 4.4.1+
    and SeekDB, including:
    - AI_EMBED: Convert text to vector embeddings
    - AI_COMPLETE: Generate text using LLM
    - AI_RERANK: Rerank search results for better accuracy

    Setup:
        Install ``langchain-oceanbase`` and deploy OceanBase 4.4.1+ or SeekDB.

        .. code-block:: bash

            pip install -U langchain-oceanbase

        For OceanBase 4.4.1+:
            docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d oceanbase/oceanbase-ce:4.4.1.0-100000032025101610

        For SeekDB:
            Follow SeekDB installation guide

    Key init args:
        connection_args: Optional[dict[str, any]]
            The connection args used for this class comes in the form of a dict. Refer to
            `DEFAULT_OCEANBASE_CONNECTION` for example.

    Instantiate:
        .. code-block:: python

            from langchain_oceanbase.ai_functions import OceanBaseAIFunctions

            connection_args = {
                "host": "127.0.0.1",
                "port": "2881",
                "user": "root@test",
                "password": "",
                "db_name": "test",
            }

            ai_functions = OceanBaseAIFunctions(connection_args=connection_args)

    Use AI_EMBED:
        .. code-block:: python

            # Embed text to vector
            vector = ai_functions.ai_embed(
                text="Hello, world!",
                model_name="text-embedding-model"
            )

    Use AI_COMPLETE:
        .. code-block:: python

            # Generate text completion
            completion = ai_functions.ai_complete(
                prompt="What is the capital of France?",
                model_name="text-generation-model"
            )

    Use AI_RERANK:
        .. code-block:: python

            # Rerank search results
            reranked = ai_functions.ai_rerank(
                query="machine learning",
                documents=["doc1", "doc2", "doc3"],
                model_name="rerank-model",
                top_k=2
            )

    Features:
        - Support for AI_EMBED function for text-to-vector conversion
        - Support for AI_COMPLETE function for text generation
        - Support for AI_RERANK function for result reranking
        - Automatic version checking for OceanBase 4.4.1+ and SeekDB
        - Full compatibility with LangChain ecosystem
    """  # noqa: E501

    def __init__(
        self,
        connection_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the OceanBase AI functions client.

        Args:
            connection_args: Connection parameters for OceanBase/SeekDB.
                If None, uses DEFAULT_OCEANBASE_CONNECTION. Should include:
                - host: OceanBase server host (default: "localhost")
                - port: OceanBase server port (default: "2881")
                - user: Database username (default: "root@test")
                - password: Database password (default: "")
                - db_name: Database name (default: "test")
            **kwargs: Additional arguments passed to ObVecClient.

        Raises:
            ValueError: If AI functions are not supported by the database version.
        """
        self.connection_args = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION
        )
        self._create_client(**kwargs)
        assert self.obvector is not None

        if not _check_ai_function_support(self.obvector):
            raise ValueError(
                "AI functions are only supported in OceanBase 4.4.1+ or SeekDB. "
                "Please upgrade your database or use SeekDB."
            )

    def _create_client(self, **kwargs: Any) -> None:
        """Create and initialize the OceanBase vector client.

        Args:
            **kwargs: Additional arguments passed to ObVecClient constructor.
        """
        host = self.connection_args.get("host", "localhost")
        port = self.connection_args.get("port", "2881")
        user = self.connection_args.get("user", "root@test")
        password = self.connection_args.get("password", "")
        db_name = self.connection_args.get("db_name", "test")

        self.obvector: ObVecClient = ObVecClient(
            uri=f"{host}:{port}",
            user=user,
            password=password,
            db_name=db_name,
            **kwargs,
        )

    @staticmethod
    def _escape_sql_string(s: str) -> str:
        """Escape SQL string to prevent SQL injection.

        Args:
            s: String to escape.

        Returns:
            Escaped string safe for SQL queries.
        """
        return s.replace("'", "''").replace("\\", "\\\\")

    def _execute_sql(self, sql_str: str, use_driver_sql: bool = False) -> Any:
        """Execute SQL query and return scalar result.

        Args:
            sql_str: SQL query string.
            use_driver_sql: If True, use exec_driver_sql instead of execute.

        Returns:
            Scalar result from the query.

        Raises:
            Exception: If the query execution fails.
        """
        with self.obvector.engine.connect() as conn:
            if use_driver_sql:
                result = conn.exec_driver_sql(sql_str)
            else:
                result = conn.execute(text(sql_str))
            return result.scalar()

    def ai_embed(
        self,
        text: str,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Convert text to vector embedding using AI_EMBED function.

        According to OceanBase documentation, AI_EMBED function signature:
        AI_EMBED(model_name, text, dimension)
        where model_name is the first parameter.

        Args:
            text: The text to embed.
            model_name: Name of the embedding model to use.
                If not specified, will try without model_name first,
                and if that fails, will raise an error suggesting to provide model_name.
            dimension: Optional dimension for the embedding vector.
            **kwargs: Additional parameters for AI_EMBED function.

        Returns:
            List[float]: The vector embedding of the text.

        Raises:
            ValueError: If model_name is required but not provided.
            Exception: If the embedding operation fails.

        Example:
            .. code-block:: python

                vector = ai_functions.ai_embed(
                    text="Hello, world!",
                    model_name="text-embedding-v1"
                )
        """
        try:
            escaped_text = self._escape_sql_string(text)

            if model_name:
                escaped_model = self._escape_sql_string(model_name)
                if dimension is not None:
                    sql_str = (
                        f"SELECT AI_EMBED('{escaped_model}', '{escaped_text}', "
                        f"{dimension}) AS embedding"
                    )
                else:
                    sql_str = (
                        f"SELECT AI_EMBED('{escaped_model}', '{escaped_text}') "
                        f"AS embedding"
                    )
                embedding = self._execute_sql(sql_str)

                if embedding is None:
                    raise ValueError("Failed to generate embedding")

                # Parse JSON string if needed
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, try to parse as Python literal
                        import ast

                        try:
                            embedding = ast.literal_eval(embedding)
                        except (ValueError, SyntaxError):
                            raise ValueError(
                                f"Failed to parse embedding result: {embedding[:100]}..."
                            ) from None

                # Ensure it's a list
                if not isinstance(embedding, list):
                    raise ValueError(
                        f"Expected list but got {type(embedding)}: {embedding[:100] if hasattr(embedding, '__getitem__') else embedding}"
                    )

                return embedding
            else:
                try:
                    sql_str = f"SELECT AI_EMBED('{escaped_text}') AS embedding"
                    embedding = self._execute_sql(sql_str)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "required" in error_msg or "parameter" in error_msg:
                        raise ValueError(
                            "model_name is required for AI_EMBED. "
                            "Please provide a model name."
                        ) from e
                    raise

                if embedding is None:
                    raise ValueError("Failed to generate embedding")

                # Parse JSON string if needed
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except json.JSONDecodeError:
                        import ast

                        try:
                            embedding = ast.literal_eval(embedding)
                        except (ValueError, SyntaxError):
                            raise ValueError(
                                f"Failed to parse embedding result: {embedding[:100]}..."
                            ) from None

                return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def ai_complete(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        content: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text completion using AI_COMPLETE function.

        According to OceanBase documentation, AI_COMPLETE function signature:
        AI_COMPLETE(model_name, prompt, content, options)
        where model_name is the first parameter.

        Args:
            prompt: The prompt text for text generation.
            model_name: Optional name of the LLM model to use.
                If not specified, uses the default model.
            content: Optional content text to be processed (replaces {{TEXT}} in prompt).
            options: Optional dictionary of additional parameters.
                Common parameters include:
                - temperature: Control randomness (0.0-1.0)
                - top_p: Nucleus sampling parameter
                - presence_penalty: Presence penalty parameter
                - extra_body: Additional body parameters
            **kwargs: Additional parameters for AI_COMPLETE function.

        Returns:
            str: The generated text completion.

        Raises:
            Exception: If the completion operation fails.

        Example:
            .. code-block:: python

                completion = ai_functions.ai_complete(
                    prompt="Translate to English: {{TEXT}}",
                    model_name="text-generation-model",
                    content="你好世界"
                )
        """
        try:
            final_prompt = prompt
            if content is not None:
                final_prompt = prompt.replace("{{TEXT}}", content)

            escaped_prompt = self._escape_sql_string(final_prompt)

            if model_name:
                escaped_model = self._escape_sql_string(model_name)

                if options is not None:
                    options_json = json.dumps(
                        options, ensure_ascii=False, separators=(",", ":")
                    )
                    escaped_options = self._escape_sql_string(options_json)
                    sql_str = (
                        f"SELECT AI_COMPLETE('{escaped_model}', '{escaped_prompt}', "
                        f"'{escaped_options}') AS completion"
                    )
                    completion = self._execute_sql(sql_str, use_driver_sql=True)
                else:
                    sql_str = (
                        f"SELECT AI_COMPLETE('{escaped_model}', '{escaped_prompt}') "
                        f"AS completion"
                    )
                    completion = self._execute_sql(sql_str)
            else:
                try:
                    sql_str = f"SELECT AI_COMPLETE('{escaped_prompt}') AS completion"
                    completion = self._execute_sql(sql_str)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "required" in error_msg or "parameter" in error_msg:
                        raise ValueError(
                            "model_name is required for AI_COMPLETE. "
                            "Please provide a model name."
                        ) from e
                    raise

            if completion is None:
                raise ValueError("Failed to generate completion")
            return completion
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            raise

    def ai_rerank(
        self,
        query: str,
        documents: List[str],
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using AI_RERANK function.

        According to OceanBase documentation:
        - https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000004018305
        - https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000004018306

        AI_RERANK function signature: AI_RERANK(model_name, query, documents_json)
        where documents_json is a JSON array string containing all documents.

        Args:
            query: The search query text.
            documents: List of document texts to rerank.
            model_name: Name of the reranking model to use.
                This parameter is typically required for AI_RERANK function.
            top_k: Optional number of top results to return.
                If not specified, returns all reranked documents.
            **kwargs: Additional parameters for AI_RERANK function.

        Returns:
            List[Dict[str, Any]]: List of reranked documents with scores.
                Each dict contains:
                - document: The document text
                - score: The relevance score
                - rank: The ranking position

        Raises:
            ValueError: If model_name is required but not provided.
            Exception: If the reranking operation fails.

        Example:
            .. code-block:: python

                reranked = ai_functions.ai_rerank(
                    query="machine learning",
                    documents=["doc1", "doc2", "doc3"],
                    model_name="rerank-model",
                    top_k=2
                )
        """
        try:
            if not documents:
                return []

                if not model_name:
                    raise ValueError(
                        "model_name is required for AI_RERANK. "
                        "Please provide a model name."
                    )

            escaped_query = self._escape_sql_string(query)
            escaped_model = self._escape_sql_string(model_name)

            # Try batch rerank first
            try:
                escaped_docs = [
                    f"'{self._escape_sql_string(doc)}'" for doc in documents
                ]
                json_array_str = ", ".join(escaped_docs)
                sql_str = (
                    f"SELECT AI_RERANK('{escaped_model}', '{escaped_query}', "
                    f"JSON_ARRAY({json_array_str})) AS scores"
                )

                scores_json = self._execute_sql(sql_str)
                if scores_json is not None:
                    parsed_result = self._parse_rerank_result(scores_json, documents)
                    return self._format_rerank_results(parsed_result, top_k)
            except Exception as batch_error:
                logger.warning(
                    f"Batch rerank failed, trying individual rerank: {batch_error}"
                )
                return self._rerank_individual(query, documents, model_name, top_k)

        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            raise

    def _parse_rerank_result(
        self, scores_json: Any, documents: List[str]
    ) -> List[Dict[str, Any]]:
        """Parse rerank result from database.

        Args:
            scores_json: Raw result from database.
            documents: Original documents list.

        Returns:
            List of parsed results.
        """
        if isinstance(scores_json, str):
            try:
                parsed_result = json.loads(scores_json)
            except json.JSONDecodeError:
                parsed_result = scores_json
        else:
            parsed_result = scores_json

        results = []

        if isinstance(parsed_result, list) and len(parsed_result) > 0:
            first_item = parsed_result[0]
            if (
                isinstance(first_item, dict)
                and "document" in first_item
                and "score" in first_item
            ):
                # Result already contains document and score
                for item in parsed_result:
                    results.append(
                        {
                            "document": item.get("document", ""),
                            "score": float(item.get("score", 0)),
                        }
                    )
            else:
                # Result is just scores array
                for doc, score in zip(documents, parsed_result):
                    results.append(
                        {
                            "document": doc,
                            "score": float(score),
                        }
                    )
        else:
            logger.warning(f"Unexpected AI_RERANK result format: {type(parsed_result)}")
            raise ValueError("Unexpected AI_RERANK result format")

        return results

    def _format_rerank_results(
        self, results: List[Dict[str, Any]], top_k: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Format and sort rerank results.

        Args:
            results: List of results with document and score.
            top_k: Optional number of top results to return.

        Returns:
            Formatted results with rank.
        """
        results.sort(key=lambda x: x["score"], reverse=True)

        for idx, result in enumerate(results):
            result["rank"] = idx + 1

        if top_k is not None and top_k > 0:
            results = results[:top_k]

        return results

    def _rerank_individual(
        self,
        query: str,
        documents: List[str],
        model_name: str,
        top_k: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Rerank documents individually (fallback method).

        Args:
            query: Search query.
            documents: Documents to rerank.
            model_name: Model name.
            top_k: Optional number of top results.

        Returns:
            List of reranked results.
        """
        escaped_query = self._escape_sql_string(query)
        escaped_model = self._escape_sql_string(model_name)

        results = []
        for idx, doc in enumerate(documents):
            try:
                escaped_doc = self._escape_sql_string(doc)
                sql_str = (
                    f"SELECT AI_RERANK('{escaped_model}', '{escaped_query}', "
                    f"JSON_ARRAY('{escaped_doc}')) AS score"
                )

                score_result = self._execute_sql(sql_str)
                if score_result is not None:
                    score = self._extract_score(score_result)
                    results.append(
                        {
                            "document": doc,
                            "score": score,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to rerank document {idx}: {e}")
                continue

        return self._format_rerank_results(results, top_k)

    @staticmethod
    def _extract_score(score_result: Any) -> float:
        """Extract score from various result formats.

        Args:
            score_result: Score result from database.

        Returns:
            Extracted score as float.
        """
        if isinstance(score_result, (list, tuple)) and len(score_result) > 0:
            return float(score_result[0])
        elif isinstance(score_result, str):
            try:
                parsed = json.loads(score_result)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return float(parsed[0])
                else:
                    return float(parsed)
            except (json.JSONDecodeError, ValueError, TypeError):
                return float(score_result)
        else:
            return float(score_result)

    def batch_ai_embed(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Batch convert texts to vector embeddings using AI_EMBED function.

        Args:
            texts: List of texts to embed.
            model_name: Optional name of the embedding model to use.
            **kwargs: Additional parameters for AI_EMBED function.

        Returns:
            List[List[float]]: List of vector embeddings, one for each text.

        Example:
            .. code-block:: python

                vectors = ai_functions.batch_ai_embed(
                    texts=["Hello", "World"],
                    model_name="text-embedding-v1"
                )
        """
        embeddings = []
        for text_item in texts:
            embedding = self.ai_embed(text=text_item, model_name=model_name, **kwargs)
            embeddings.append(embedding)
        return embeddings

    def create_ai_model(
        self,
        model_name: str,
        model_type: str,
        provider_model_name: Optional[str] = None,
    ) -> bool:
        """Create an AI model in OceanBase.

        This method calls DBMS_AI_SERVICE.CREATE_AI_MODEL to register a model.

        Args:
            model_name: Name of the model to create in OceanBase.
            model_type: Type of the model (e.g., "dense_embedding", "completion", etc.).
                The type should match the model's actual type in OceanBase.
            provider_model_name: Optional name of the model in the provider's system.
                If not provided, uses model_name.

        Returns:
            bool: True if model was created successfully.

        Raises:
            Exception: If the model creation fails.

        Example:
            .. code-block:: python

                # Create an embedding model
                ai_functions.create_ai_model(
                    model_name="text-embedding-3-small",
                    model_type="dense_embedding"
                )

                # Create a completion model
                ai_functions.create_ai_model(
                    model_name="qwen3-coder-plus",
                    model_type="completion",
                    provider_model_name="deepseek-v3"
                )
        """
        try:
            with self.obvector.engine.begin() as conn:
                provider_name = (
                    provider_model_name if provider_model_name else model_name
                )
                # Use model_type directly as provided by user
                model_config = {"type": model_type, "model_name": provider_name}
                config_json = json.dumps(
                    model_config, ensure_ascii=False, separators=(",", ":")
                )

                escaped_model_name = self._escape_sql_string(model_name)
                escaped_config = self._escape_sql_string(config_json)

                sql_str = (
                    f"CALL DBMS_AI_SERVICE.CREATE_AI_MODEL("
                    f"'{escaped_model_name}', '{escaped_config}')"
                )
                conn.execute(text(sql_str))
                # begin() context manager automatically commits
                return True
        except Exception as e:
            logger.error(f"Failed to create AI model: {e}")
            raise

    def drop_ai_model(self, model_name: str) -> bool:
        """Drop an AI model from OceanBase.

        This method calls DBMS_AI_SERVICE.DROP_AI_MODEL to remove a model.

        Args:
            model_name: Name of the model to drop.

        Returns:
            bool: True if model was dropped successfully.

        Raises:
            Exception: If the model deletion fails.

        Example:
            .. code-block:: python

                ai_functions.drop_ai_model("text-embedding-3-small")
        """
        try:
            with self.obvector.engine.begin() as conn:
                escaped_model_name = self._escape_sql_string(model_name)
                sql_str = f"CALL DBMS_AI_SERVICE.DROP_AI_MODEL('{escaped_model_name}')"
                conn.execute(text(sql_str))
                # begin() context manager automatically commits
                return True
        except Exception as e:
            logger.error(f"Failed to drop AI model: {e}")
            raise

    def create_ai_model_endpoint(
        self,
        endpoint_name: str,
        ai_model_name: str,
        url: str,
        access_key: str,
        provider: str = "openai",
        scope: str = "all",
    ) -> bool:
        """Create an AI model endpoint in OceanBase.

        This method calls DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT to configure
        the endpoint for an AI model.

        Args:
            endpoint_name: Name of the endpoint to create.
            ai_model_name: Name of the AI model (must exist).
            url: API endpoint URL for the model service.
            access_key: API access key for authentication.
            provider: Provider type, default is "openai".
            scope: Scope of the endpoint, default is "all".

        Returns:
            bool: True if endpoint was created successfully.

        Raises:
            Exception: If the endpoint creation fails.

        Example:
            .. code-block:: python

                ai_functions.create_ai_model_endpoint(
                    endpoint_name="ob_complete_endpoint",
                    ai_model_name="ob_complete",
                    url="https://api.example.com/v1/chat/completions",
                    access_key="your-api-key",
                    provider="openai"
                )
        """
        try:
            with self.obvector.engine.begin() as conn:
                endpoint_config = {
                    "ai_model_name": ai_model_name,
                    "scope": scope,
                    "url": url,
                    "access_key": access_key,
                    "provider": provider,
                }
                config_json = json.dumps(
                    endpoint_config, ensure_ascii=False, separators=(",", ":")
                )

                escaped_endpoint_name = self._escape_sql_string(endpoint_name)
                escaped_config = self._escape_sql_string(config_json)

                sql_str = (
                    f"CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT("
                    f"'{escaped_endpoint_name}', '{escaped_config}')"
                )
                conn.execute(text(sql_str))
                # begin() context manager automatically commits
                return True
        except Exception as e:
            logger.error(f"Failed to create AI model endpoint: {e}")
            raise

    def alter_ai_model_endpoint(
        self,
        endpoint_name: str,
        ai_model_name: str,
        url: str,
        access_key: str,
        provider: str = "openai",
        scope: str = "all",
    ) -> bool:
        """Alter an existing AI model endpoint in OceanBase.

        This method calls DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT to update
        the endpoint configuration.

        Args:
            endpoint_name: Name of the endpoint to alter.
            ai_model_name: Name of the AI model.
            url: API endpoint URL for the model service.
            access_key: API access key for authentication.
            provider: Provider type, default is "openai".
            scope: Scope of the endpoint, default is "all".

        Returns:
            bool: True if endpoint was altered successfully.

        Raises:
            Exception: If the endpoint update fails.

        Example:
            .. code-block:: python

                ai_functions.alter_ai_model_endpoint(
                    endpoint_name="ob_complete_endpoint",
                    ai_model_name="ob_complete",
                    url="https://new-api.example.com/v1/chat/completions",
                    access_key="new-api-key",
                    provider="openai"
                )
        """
        try:
            with self.obvector.engine.begin() as conn:
                endpoint_config = {
                    "ai_model_name": ai_model_name,
                    "scope": scope,
                    "url": url,
                    "access_key": access_key,
                    "provider": provider,
                }
                config_json = json.dumps(
                    endpoint_config, ensure_ascii=False, separators=(",", ":")
                )

                escaped_endpoint_name = self._escape_sql_string(endpoint_name)
                escaped_config = self._escape_sql_string(config_json)

                sql_str = (
                    f"CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT("
                    f"'{escaped_endpoint_name}', '{escaped_config}')"
                )
                conn.execute(text(sql_str))
                # begin() context manager automatically commits
                return True
        except Exception as e:
            logger.error(f"Failed to alter AI model endpoint: {e}")
            raise

    def drop_ai_model_endpoint(self, endpoint_name: str) -> bool:
        """Drop an AI model endpoint from OceanBase.

        This method calls DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT to remove an endpoint.

        Args:
            endpoint_name: Name of the endpoint to drop.

        Returns:
            bool: True if endpoint was dropped successfully.

        Raises:
            Exception: If the endpoint deletion fails.

        Example:
            .. code-block:: python

                ai_functions.drop_ai_model_endpoint("ob_complete_endpoint")
        """
        try:
            with self.obvector.engine.begin() as conn:
                escaped_endpoint_name = self._escape_sql_string(endpoint_name)
                sql_str = (
                    f"CALL DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT("
                    f"'{escaped_endpoint_name}')"
                )
                conn.execute(text(sql_str))
                # begin() context manager automatically commits
                return True
        except Exception as e:
            logger.error(f"Failed to drop AI model endpoint: {e}")
            raise

    def list_ai_models(self) -> List[Dict[str, Any]]:
        """List all AI models configured in OceanBase.

        This method queries the oceanbase.DBA_OB_AI_MODELS view
        to retrieve all configured AI models.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing model information.
                Each dictionary contains model details such as model_name, model_type, etc.

        Raises:
            Exception: If the query fails.

        Example:
            .. code-block:: python

                models = ai_functions.list_ai_models()
                for model in models:
                    print(f"Model: {model.get('model_name')}")
                    print(f"Type: {model.get('model_type')}")
        """
        try:
            with self.obvector.engine.connect() as conn:
                sql_str = "SELECT * FROM oceanbase.DBA_OB_AI_MODELS"
                result = conn.execute(text(sql_str))

                # Get column names
                columns = result.keys()

                # Convert rows to dictionaries
                models = []
                for row in result:
                    model_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert non-serializable types to strings
                        if value is not None and not isinstance(
                            value, (str, int, float, bool, type(None))
                        ):
                            try:
                                value = json.dumps(value, ensure_ascii=False)
                            except (TypeError, ValueError):
                                value = str(value)
                        model_dict[col] = value
                    models.append(model_dict)

                return models
        except Exception as e:
            logger.error(f"Failed to list AI models: {e}")
            raise

    def list_ai_model_endpoints(self) -> List[Dict[str, Any]]:
        """List all AI model endpoints configured in OceanBase.

        This method queries the oceanbase.DBA_OB_AI_MODEL_ENDPOINTS view
        to retrieve all configured AI model endpoints.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing endpoint information.
                Each dictionary contains endpoint details such as endpoint_name,
                ai_model_name, url, provider, etc.

        Raises:
            Exception: If the query fails.

        Example:
            .. code-block:: python

                endpoints = ai_functions.list_ai_model_endpoints()
                for endpoint in endpoints:
                    print(f"Endpoint: {endpoint.get('endpoint_name')}")
                    print(f"Model: {endpoint.get('ai_model_name')}")
                    print(f"URL: {endpoint.get('url')}")
        """
        try:
            with self.obvector.engine.connect() as conn:
                sql_str = "SELECT * FROM oceanbase.DBA_OB_AI_MODEL_ENDPOINTS"
                result = conn.execute(text(sql_str))

                # Get column names
                columns = result.keys()

                # Convert rows to dictionaries
                endpoints = []
                for row in result:
                    endpoint_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert non-serializable types to strings
                        if value is not None and not isinstance(
                            value, (str, int, float, bool, type(None))
                        ):
                            try:
                                value = json.dumps(value, ensure_ascii=False)
                            except (TypeError, ValueError):
                                value = str(value)
                        endpoint_dict[col] = value
                    endpoints.append(endpoint_dict)

                return endpoints
        except Exception as e:
            logger.error(f"Failed to list AI model endpoints: {e}")
            raise
