"""Oceanbase vector stores."""

from __future__ import annotations

import json
import logging
import math
import traceback
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pyobvector import (
    SPARSE_VECTOR,
    VECTOR,
    ObVecClient,
    cosine_distance,
    inner_product,
    l2_distance,
)
from pyobvector.client.fts_index_param import FtsIndexParam, FtsParser
from pyobvector.client.index_param import VecIndexType
from sqlalchemy import JSON, Column, String, Table, func, text
from sqlalchemy.dialects.mysql import LONGTEXT

logger = logging.getLogger(__name__)

DEFAULT_OCEANBASE_CONNECTION = {
    "host": "localhost",
    "port": "2881",
    "user": "root@test",
    "password": "",
    "db_name": "test",
}
DEFAULT_OCEANBASE_VECTOR_TABLE_NAME = "langchain_vector"

# Default parameters for different index types
# from: https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000002012936
DEFAULT_OCEANBASE_HNSW_BUILD_PARAM = {"M": 16, "efConstruction": 200}
DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM = {"efSearch": 64}

# from: https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000002012936
DEFAULT_OCEANBASE_IVF_BUILD_PARAM = {"nlist": 128}

DEFAULT_OCEANBASE_IVF_SEARCH_PARAM = {}
DEFAULT_OCEANBASE_FLAT_BUILD_PARAM = {}
DEFAULT_OCEANBASE_FLAT_SEARCH_PARAM = {}

# Supported index types mapping
OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES = {
    "HNSW": VecIndexType.HNSW,
    "HNSW_SQ": VecIndexType.HNSW_SQ,
    "IVF": VecIndexType.IVFFLAT,  # Use IVFFLAT as default IVF implementation
    "IVF_FLAT": VecIndexType.IVFFLAT,
    "IVF_SQ": VecIndexType.IVFSQ,
    "IVF_PQ": VecIndexType.IVFPQ,
    "FLAT": VecIndexType.IVFFLAT,  # FLAT can be implemented as IVFFLAT with nlist=1
}

DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE = "l2"

DEFAULT_METADATA_FIELD = "metadata"


def _euclidean_similarity(distance: float) -> float:
    return 1.0 - distance / math.sqrt(2)


def _neg_inner_product_similarity(distance: float) -> float:
    return -distance


class OceanbaseVectorStore(VectorStore):
    """Oceanbase vector store integration.

    Setup:
        Install ``langchain-oceanbase`` and deploy a standalone OceanBase server with docker.

        .. code-block:: bash

            pip install -U langchain-oceanbase
            docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d oceanbase/oceanbase-ce:latest

        More methods to deploy OceanBase cluster:
        https://github.com/oceanbase/oceanbase-doc/blob/V4.3.1/en-US/400.deploy/500.deploy-oceanbase-database-community-edition/100.deployment-overview.md

    Key init args — indexing params:
        vidx_metric_type: str
            Metric method of distance between vectors.
            This parameter takes values in `l2`, `inner_product`, and `cosine`. Defaults to `l2`.
        vidx_algo_params: Optional[dict]
            Which index params to use. OceanBase supports multiple index types:
            - HNSW: Hierarchical Navigable Small World graph index, suitable for high-dimensional vectors
            - HNSW_SQ: Scalar quantized version of HNSW
            - IVF: Inverted File index, suitable for large-scale data
            - IVF_FLAT: Exact search version of IVF
            - IVF_SQ: Scalar quantized version of IVF
            - IVF_PQ: Product quantized version of IVF
            - FLAT: Brute force search index, suitable for small datasets
            Refer to `DEFAULT_OCEANBASE_HNSW_BUILD_PARAM` and other default parameters for examples.
        index_type: str
            Type of vector index to use. Supports "HNSW", "HNSW_SQ", "IVF", "IVF_FLAT", "IVF_SQ", "IVF_PQ", "FLAT".
            Defaults to "HNSW".
        drop_old: bool
            Whether to drop the current table. Defaults to False.
        primary_field: str
            Name of the primary key column. Defaults to "id".
        vector_field: str
            Name of the vector column. Defaults to "embedding".
        text_field: str
            Name of the text column. Defaults to "document".
        metadata_field: Optional[str]
            Name of the metadata column. Defaults to "metadata".
            When `metadata_field` is specified, the document's metadata will store as json.
        vidx_name: str
            Name of the vector index table.
        partitions: ObPartition
            Partition strategy of table.
            Refer to `pyobvector`'s documentation for more examples.
        extra_columns: Optional[List[Column]]
            Extra sqlalchemy columns to add to the table.
        normalize: bool
            Whether to perform L2 normalization on vectors. Defaults to False.
        embedding_dim: Optional[int]
            Dimension of vectors. If not specified, will be inferred from the first embedding vector.

    Key init args — client params:
        embedding_function: Embeddings
            Function used to embed the text.
        table_name: str
            Which table name to use. Defaults to "langchain_vector".
        connection_args: Optional[dict[str, any]]
            The connection args used for this class comes in the form of a dict. Refer to
            `DEFAULT_OCEANBASE_CONNECTION` for example.

    Instantiate:
        .. code-block:: python

            from langchain_oceanbase.vectorstores import OceanbaseVectorStore
            from langchain_community.embeddings import DashScopeEmbeddings

            DASHSCOPE_API = os.environ.get("DASHSCOPE_API_KEY", "")
            connection_args = {
                "host": "127.0.0.1",
                "port": "2881",
                "user": "root@test",
                "password": "",
                "db_name": "test",
            }
            embeddings = DashScopeEmbeddings(
                model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API
            )

            vector_store = OceanbaseVectorStore(
                embedding_function=embeddings,
                table_name="langchain_vector",
                connection_args=connection_args,
                vidx_metric_type="l2",
                index_type="HNSW",
                drop_old=True,
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

    Features:
        - Support for multiple vector index types: HNSW, IVF, FLAT, etc.
        - Support for multiple distance metrics: L2, inner product, cosine distance
        - Support for metadata filtering and querying
        - Support for batch operations and partitioning
        - Automatic table creation and index management
        - Full compatibility with LangChain ecosystem

    """  # noqa: E501

    def __init__(  # type: ignore[no-untyped-def]
        self,
        embedding_function: Embeddings,
        table_name: str = DEFAULT_OCEANBASE_VECTOR_TABLE_NAME,
        connection_args: Optional[dict[str, Any]] = None,
        vidx_metric_type: str = DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE,
        vidx_algo_params: Optional[dict] = None,
        drop_old: bool = False,
        *,
        primary_field: str = "id",
        vector_field: str = "embedding",
        text_field: str = "document",
        metadata_field: Optional[str] = DEFAULT_METADATA_FIELD,
        vidx_name: str = "vidx",
        partitions: Optional[Any] = None,
        extra_columns: Optional[List[Column]] = None,
        normalize: bool = False,
        embedding_dim: Optional[int] = None,
        index_type: str = "HNSW",
        include_sparse: bool = False,
        include_fulltext: bool = False,
        **kwargs,
    ):
        """Initialize the OceanBase vector store."""

        self.embedding_function = embedding_function
        self.table_name = table_name
        self.connection_args = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION
        )
        self.extra_columns = extra_columns
        self.normalize = normalize
        self._create_client(**kwargs)
        assert self.obvector is not None

        self.vidx_metric_type = vidx_metric_type.lower()
        if self.vidx_metric_type not in ("l2", "inner_product", "cosine"):
            raise ValueError(
                "`vidx_metric_type` should be set in `l2`/`inner_product`/`cosine`."
            )

        # Set index type and default parameters
        self.index_type = index_type.upper()
        if self.index_type not in OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES:
            raise ValueError(
                f"`index_type` should be one of "
                f"{list(OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES.keys())}. "
                f"Got {self.index_type}"
            )

        # Set default parameters based on index type
        if vidx_algo_params is None:
            # Map index types to their default build parameters
            index_param_map = {
                "HNSW": DEFAULT_OCEANBASE_HNSW_BUILD_PARAM,
                "HNSW_SQ": DEFAULT_OCEANBASE_HNSW_BUILD_PARAM,
                "IVF": DEFAULT_OCEANBASE_IVF_BUILD_PARAM,
                "IVF_FLAT": DEFAULT_OCEANBASE_IVF_BUILD_PARAM,
                "IVF_SQ": DEFAULT_OCEANBASE_IVF_BUILD_PARAM,
                "IVF_PQ": DEFAULT_OCEANBASE_IVF_BUILD_PARAM,
                "FLAT": DEFAULT_OCEANBASE_FLAT_BUILD_PARAM,
            }

            self.vidx_algo_params = index_param_map[self.index_type].copy()

            # Special handling for IVF_PQ: add 'm' parameter if not present
            if self.index_type == "IVF_PQ" and "m" not in self.vidx_algo_params:
                # IVF_PQ requires 'm' parameter that must divide the embedding dimension
                # Default to 3 as a reasonable divisor for most embedding dimensions
                self.vidx_algo_params["m"] = 3
        else:
            self.vidx_algo_params = vidx_algo_params.copy()
            # Add index_type to params for internal use
            self.vidx_algo_params["index_type"] = self.index_type

        self.drop_old = drop_old
        self.primary_field = primary_field
        self.vector_field = vector_field
        self.text_field = text_field
        self.metadata_field = metadata_field or DEFAULT_METADATA_FIELD
        self.vidx_name = vidx_name
        self.partition = partitions
        self.hnsw_ef_search = -1

        # Hybrid search parameters
        self.include_sparse = include_sparse
        self.include_fulltext = include_fulltext
        self.sparse_vector_field = "sparse_embedding"
        self.fulltext_field = "fulltext_content"

        # Define table names for sparse and fulltext operations
        # These tables are independent and don't affect each other
        self.sparse_table_name = f"{table_name}_sparse" if self.include_sparse else None
        # Fulltext requires sparse support, so use sparse_fulltext table
        self.fulltext_table_name = (
            f"{table_name}_sparse_fulltext" if self.include_fulltext else None
        )

        if self.include_fulltext and not self.include_sparse:
            raise ValueError(
                "Full-text search requires sparse vector support. "
                "Set include_sparse=True when include_fulltext=True."
            )

        if self.drop_old:
            self.obvector.drop_table_if_exist(self.table_name)
            if self.sparse_table_name:
                self.obvector.drop_table_if_exist(self.sparse_table_name)
            if self.fulltext_table_name:
                self.obvector.drop_table_if_exist(self.fulltext_table_name)

        if not self.obvector.check_table_exists(self.table_name):
            if embedding_dim is not None:
                self._create_table_with_index_by_embedding_dim(embedding_dim)
                self._load_table()
        else:
            self._load_table()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _create_client(self, **kwargs):  # type: ignore[no-untyped-def]
        host = self.connection_args.get("host", "localhost")
        port = self.connection_args.get("port", "2881")
        user = self.connection_args.get("user", "root@test")
        password = self.connection_args.get("password", "")
        db_name = self.connection_args.get("db_name", "test")

        self.obvector = ObVecClient(
            uri=f"{host}:{port}",
            user=user,
            password=password,
            db_name=db_name,
            **kwargs,
        )

    def _load_table(self) -> None:
        table = Table(
            self.table_name,
            self.obvector.metadata_obj,
            autoload_with=self.obvector.engine,
        )
        column_names = [column.name for column in table.columns]

        # Calculate expected column count including hybrid search fields
        base_columns = 4  # id, vector, text, metadata
        hybrid_columns = 0
        if self.include_sparse:
            hybrid_columns += 1
        if self.include_fulltext:
            hybrid_columns += 1
        extra_columns = len(self.extra_columns or [])

        expected_columns = base_columns + hybrid_columns + extra_columns
        assert len(column_names) == expected_columns, (
            f"Expected {expected_columns} columns, got {len(column_names)}: "
            f"{column_names}"
        )

        logging.info(f"load exist table with {column_names} columns")
        self.primary_field = column_names[0]
        self.vector_field = column_names[1]
        self.text_field = column_names[2]
        self.metadata_field = column_names[3]

    def _create_table_with_index_by_embedding_dim(self, dim: int) -> None:
        cols = [
            Column(
                self.primary_field, String(4096), primary_key=True, autoincrement=False
            ),
            Column(self.vector_field, VECTOR(dim)),
            Column(self.text_field, LONGTEXT),
            Column(self.metadata_field, JSON),
        ]

        # Add hybrid search columns if enabled
        if self.include_sparse:
            cols.append(Column(self.sparse_vector_field, SPARSE_VECTOR()))

        if self.include_fulltext:
            cols.append(Column(self.fulltext_field, LONGTEXT))

        if self.extra_columns is not None:
            cols.extend(self.extra_columns)

        vidx_params = self.obvector.prepare_index_params()
        vidx_params.add_index(
            field_name=self.vector_field,
            index_type=OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES[self.index_type],
            index_name=self.vidx_name,
            metric_type=self.vidx_metric_type,
            params=self.vidx_algo_params,
        )

        # Add sparse vector index if enabled
        if self.include_sparse:
            sparse_index_kwargs = {}
            # Check if SeekDB and add sindi type
            if self.obvector._is_seekdb():
                sparse_index_kwargs["sparse_index_type"] = "sindi"
            vidx_params.add_index(
                field_name=self.sparse_vector_field,
                index_type=VecIndexType.DAAT,
                index_name=f"{self.vidx_name}_sparse",
                metric_type="inner_product",
                **sparse_index_kwargs,
            )

        fts_idxs = None
        if self.include_fulltext:
            fts_idxs = [
                FtsIndexParam(
                    index_name=f"{self.vidx_name}_fts",
                    field_names=[self.fulltext_field],
                    parser_type=FtsParser.NGRAM,
                )
            ]

        self.obvector.create_table_with_index_params(
            table_name=self.table_name,
            columns=cols,
            indexes=None,
            vidxs=vidx_params,
            fts_idxs=fts_idxs,
            partitions=self.partition,
        )

    def _create_sparse_table(self, embedding_dim: int) -> None:
        """Create a table for sparse vector search only."""
        if not self.sparse_table_name:
            return
        if self.obvector.check_table_exists(self.sparse_table_name):
            return

        cols = [
            Column(
                self.primary_field, String(4096), primary_key=True, autoincrement=False
            ),
            Column(self.vector_field, VECTOR(embedding_dim)),
            Column(self.text_field, LONGTEXT),
            Column(self.metadata_field, JSON),
            Column(self.sparse_vector_field, SPARSE_VECTOR()),
        ]

        if self.extra_columns is not None:
            cols.extend(self.extra_columns)

        vidx_params = self.obvector.prepare_index_params()
        vidx_params.add_index(
            field_name=self.vector_field,
            index_type=OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES[self.index_type],
            index_name=f"{self.vidx_name}_sparse",
            metric_type=self.vidx_metric_type,
            params=self.vidx_algo_params,
        )

        # Add sparse vector index
        sparse_index_kwargs = {}
        # Check if SeekDB and add sindi type
        if self.obvector._is_seekdb():
            sparse_index_kwargs["sparse_index_type"] = "sindi"
        vidx_params.add_index(
            field_name=self.sparse_vector_field,
            index_type=VecIndexType.DAAT,
            index_name=f"{self.vidx_name}_sparse",
            metric_type="inner_product",
            **sparse_index_kwargs,
        )

        self.obvector.create_table_with_index_params(
            table_name=self.sparse_table_name,
            columns=cols,
            indexes=None,
            vidxs=vidx_params,
            fts_idxs=None,
            partitions=self.partition,
        )

    def _create_sparse_fulltext_table(self, embedding_dim: int) -> None:
        """Create a table for both sparse vector and fulltext search."""
        if not self.fulltext_table_name:
            return
        if self.obvector.check_table_exists(self.fulltext_table_name):
            return

        cols = [
            Column(
                self.primary_field, String(4096), primary_key=True, autoincrement=False
            ),
            Column(self.vector_field, VECTOR(embedding_dim)),
            Column(self.text_field, LONGTEXT),
            Column(self.metadata_field, JSON),
            Column(self.sparse_vector_field, SPARSE_VECTOR()),
            Column(self.fulltext_field, LONGTEXT),
        ]

        if self.extra_columns is not None:
            cols.extend(self.extra_columns)

        vidx_params = self.obvector.prepare_index_params()
        vidx_params.add_index(
            field_name=self.vector_field,
            index_type=OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES[self.index_type],
            index_name=f"{self.vidx_name}_sparse_fulltext",
            metric_type=self.vidx_metric_type,
            params=self.vidx_algo_params,
        )

        # Add sparse vector index
        sparse_index_kwargs = {}
        # Check if SeekDB and add sindi type
        if self.obvector._is_seekdb():
            sparse_index_kwargs["sparse_index_type"] = "sindi"
        vidx_params.add_index(
            field_name=self.sparse_vector_field,
            index_type=VecIndexType.DAAT,
            index_name=f"{self.vidx_name}_sparse",
            metric_type="inner_product",
            **sparse_index_kwargs,
        )

        fts_idxs = [
            FtsIndexParam(
                index_name=f"{self.vidx_name}_fts_sparse_fulltext",
                field_names=[self.fulltext_field],
                parser_type=FtsParser.NGRAM,
            )
        ]

        self.obvector.create_table_with_index_params(
            table_name=self.fulltext_table_name,
            columns=cols,
            indexes=None,
            vidxs=vidx_params,
            fts_idxs=fts_idxs,
            partitions=self.partition,
        )

    def _create_table_with_index(self, embeddings: list) -> None:
        if self.obvector.check_table_exists(self.table_name):
            self._load_table()
            return

        dim = len(embeddings[0])
        self._create_table_with_index_by_embedding_dim(dim)

    def _determine_search_table(self) -> str:
        """
        Determine which table to use for search based on hybrid search configuration.

        Returns:
            str: Table name to use for search
        """
        search_table = self.table_name
        if self.include_fulltext and self.fulltext_table_name:
            # Fulltext table contains dense vectors, use it for vector search
            try:
                with self.obvector.engine.connect() as conn:
                    fulltext_count = conn.execute(
                        text(f"SELECT COUNT(*) FROM `{self.fulltext_table_name}`")
                    ).scalar()
                    if fulltext_count > 0:
                        search_table = self.fulltext_table_name
            except Exception:
                # If query fails, fall back to main table
                pass
        elif self.include_sparse and self.sparse_table_name:
            # Sparse table also contains dense vectors, use it if fulltext table is empty
            try:
                with self.obvector.engine.connect() as conn:
                    sparse_count = conn.execute(
                        text(f"SELECT COUNT(*) FROM `{self.sparse_table_name}`")
                    ).scalar()
                    if sparse_count > 0:
                        search_table = self.sparse_table_name
            except Exception:
                # If query fails, fall back to main table
                pass
        return search_table

    def _handle_hnsw_ef_search(self, search_param: dict) -> None:
        """
        Handle HNSW-specific efSearch parameter.

        Args:
            search_param: Search parameters dictionary
        """
        if self.index_type in ["HNSW", "HNSW_SQ"]:
            ef_search = search_param.get(
                "efSearch", self._get_default_search_params()["efSearch"]
            )
            if ef_search != self.hnsw_ef_search:
                self.obvector.set_ob_hnsw_ef_search(ef_search)
                self.hnsw_ef_search = ef_search

    def _convert_results_to_documents(
        self, results, include_score: bool = False
    ) -> List[Document] | List[Tuple[Document, float]]:
        """
        Convert search results to Document objects.

        Args:
            results: CursorResult from ann_search
            include_score: Whether to include score in results

        Returns:
            List[Document] or List[Tuple[Document, float]]: Converted documents
        """
        if include_score:
            return [
                (
                    Document(
                        id=r[2],
                        page_content=r[0],
                        metadata=json.loads(r[1]) if isinstance(r[1], str) else r[1],
                    ),
                    r[3],
                )
                for r in results.fetchall()
            ]
        else:
            return [
                Document(
                    id=r[2],
                    page_content=r[0],
                    metadata=json.loads(r[1]) if isinstance(r[1], str) else r[1],
                )
                for r in results.fetchall()
            ]

    def _parse_metric_type_str_to_dist_func(self) -> Any:
        if self.vidx_metric_type == "l2":
            return func.l2_distance
        if self.vidx_metric_type == "cosine":
            return func.cosine_distance
        if self.vidx_metric_type == "inner_product":
            return func.negative_inner_product
        raise ValueError(f"Invalid vector index metric type: {self.vidx_metric_type}")

    def _normalize(self, vector: List[float]) -> List[float]:
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        arr = arr / norm
        return arr.tolist()

    def _get_default_search_params(self) -> dict:
        """Get default search parameters based on index type."""
        search_param_map = {
            "HNSW": DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM,
            "HNSW_SQ": DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM,
            "IVF": DEFAULT_OCEANBASE_IVF_SEARCH_PARAM,
            "IVF_FLAT": DEFAULT_OCEANBASE_IVF_SEARCH_PARAM,
            "IVF_SQ": DEFAULT_OCEANBASE_IVF_SEARCH_PARAM,
            "IVF_PQ": DEFAULT_OCEANBASE_IVF_SEARCH_PARAM,
            "FLAT": DEFAULT_OCEANBASE_FLAT_SEARCH_PARAM,
        }
        return search_param_map.get(
            self.index_type, DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        extras: Optional[List[dict]] = None,
        partition_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into OceanBase.

        Inserting data when the table has not be created yet will result
        in creating a new table. The data of the first record decides
        the schema of the new table, the dim is extracted from the first
        embedding.

        Args:
            texts (Iterable[str]): The texts to embed. OceanBase use a `LONGTEXT`
                type column to hold the data.
            metadatas (Optional[List[dict]]): Metadata dicts attached to each of
                the texts. Defaults to None.
            batch_size (int, optional): Batch size to use for insertion.
                Defaults to 1000.
            ids (Optional[List[str]]): List of text ids.
            extras (Optional[List[dict]]): Extra data to store in the table.
            partition_name (Optional[str]): The partition name to insert data into.

        Raises:
            Exception: Failure to add texts

        Returns:
            List[str]: The resulting ids for each inserted element.
        """
        texts = list(texts)

        try:
            embeddings = self.embedding_function.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_function.embed_query(x) for x in texts]

        total_count = len(embeddings)
        if total_count == 0:
            return []

        self._create_table_with_index(embeddings)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        extra_data = extras or [{} for _ in texts]

        pks: list[str] = []
        for i in range(0, total_count, batch_size):
            data = [
                {
                    self.primary_field: id,
                    self.vector_field: (
                        embedding if not self.normalize else self._normalize(embedding)
                    ),
                    self.text_field: text,
                    self.metadata_field: metadata,
                    **extra,
                }
                for id, embedding, text, metadata, extra in zip(
                    ids[i : i + batch_size],
                    embeddings[i : i + batch_size],
                    texts[i : i + batch_size],
                    metadatas[i : i + batch_size],
                    extra_data[i : i + batch_size],
                )
            ]
            try:
                self.obvector.upsert(
                    table_name=self.table_name,
                    data=data,
                    partition_name=(partition_name or ""),
                )
                pks.extend(ids[i : i + batch_size])
            except Exception:
                traceback.print_exc()
                logger.error(
                    f"Failed to insert batch starting at entity:[{i}, {i + batch_size})"
                )
        return pks

    def delete(  # type: ignore[no-untyped-def]
        self, ids: Optional[List[str]] = None, fltr: Optional[str] = None, **kwargs
    ) -> Optional[bool]:
        """Delete by vector ID or boolean expression.

        Args:
            ids (Optional[List[str]]): List of ids to delete.
            fltr (Optional[str]): Boolean filter that specifies the entities to delete.
        """
        self.obvector.delete(
            table_name=self.table_name,
            ids=ids,
            where_clause=([text(fltr)] if fltr is not None else None),
        )
        return None

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get entities by vector ID.

        Args:
            ids (Optional[List[str]]): List of ids to get.

        Returns:
            List[Document]: Document results for search.
        """
        res = self.obvector.get(
            table_name=self.table_name,
            ids=ids,
            output_column_name=[
                self.text_field,
                self.metadata_field,
                self.primary_field,
            ],
        )
        return [
            Document(
                id=r[2],
                page_content=r[0],
                metadata=(
                    json.loads(r[1])
                    if isinstance(r[1], str) or isinstance(r[1], bytes)
                    else r[1]
                ),
            )
            for r in res.fetchall()
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to default search parameters for each
                index type.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Document]: Document results for search.
        """
        if k < 0:
            return []

        query_vector = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(
            embedding=query_vector, k=k, param=param, fltr=fltr, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        Args:
            query (str): The text being searched.
            k (int, optional): How many results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to default search parameters for each
                index type.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: Document results with score for search.
        """
        if k < 0:
            return []

        query_vector = self.embedding_function.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding=query_vector, k=k, param=param, fltr=fltr, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            embedding (List[float]): The embedding vector to search.
            k (int, optional): How many results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to default search parameters for each
                index type.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Document]: Document results for search.
        """
        if k < 0:
            return []

        search_param = param if param is not None else self._get_default_search_params()
        self._handle_hnsw_ef_search(search_param)
        search_table = self._determine_search_table()

        res = self.obvector.ann_search(
            table_name=search_table,
            vec_data=(embedding if not self.normalize else self._normalize(embedding)),
            vec_column_name=self.vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            topk=k,
            output_column_names=[
                self.text_field,
                self.metadata_field,
                self.primary_field,
            ],
            where_clause=([text(fltr)] if fltr is not None else None),
            **kwargs,
        )
        return self._convert_results_to_documents(res, include_score=False)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 10,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to default search parameters for each
                index type.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: Document results with score for search.
        """
        if k < 0:
            return []

        search_param = param if param is not None else self._get_default_search_params()
        self._handle_hnsw_ef_search(search_param)
        search_table = self._determine_search_table()

        res = self.obvector.ann_search(
            table_name=search_table,
            vec_data=(embedding if not self.normalize else self._normalize(embedding)),
            vec_column_name=self.vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            with_dist=True,
            topk=k,
            output_column_names=[
                self.text_field,
                self.metadata_field,
                self.primary_field,
            ],
            where_clause=([text(fltr)] if fltr is not None else None),
            **kwargs,
        )
        return self._convert_results_to_documents(res, include_score=True)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        table_name: str = DEFAULT_OCEANBASE_VECTOR_TABLE_NAME,
        connection_args: Optional[dict[str, Any]] = None,
        vidx_metric_type: str = DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE,
        vidx_algo_params: Optional[dict] = None,
        drop_old: bool = False,
        *,
        ids: Optional[List[str]] = None,
        extra_columns: Optional[List[Column]] = None,
        normalize: bool = False,
        extras: Optional[List[dict]] = None,
        index_type: str = "HNSW",
        **kwargs: Any,
    ) -> "OceanbaseVectorStore":
        """Create a OceanBase table, indexes it with specified index type, and
        insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            table_name (str): Table name to use. Defaults to "langchain_vector".
            connection_args (Optional[dict[str, Any]]): Refer to
                `DEFAULT_OCEANBASE_CONNECTION` for example.
            vidx_metric_type (str): Metric method of distance between vectors.
                This parameter takes values in `l2`, `inner_product`, and `cosine`.
                Defaults to `l2`.
            vidx_algo_params (Optional[dict]): Which index params to use. Supports
                HNSW, IVF, and FLAT index types. Refer to default parameters for
                examples.
            drop_old (bool): Whether to drop the current table. Defaults
                to False.
            ids (Optional[List[str]]): List of text ids. Defaults to None.
            extra_columns (Optional[List[Column]]): Extra columns to add to the table.
            extras (Optional[List[dict]]): Extra data to insert. Defaults to None.
            index_type (str): Type of vector index to use. Supports "HNSW",
                "HNSW_SQ", "IVF", "IVF_FLAT", "IVF_SQ", "IVF_PQ", "FLAT".
                Defaults to "HNSW".

        Returns:
            OceanBase: OceanBase Vector Store
        """
        oceanbase = cls(
            embedding_function=embedding,
            table_name=table_name,
            connection_args=connection_args,
            vidx_metric_type=vidx_metric_type,
            vidx_algo_params=vidx_algo_params,
            drop_old=drop_old,
            extra_columns=extra_columns,
            normalize=normalize,
            index_type=index_type,
            **kwargs,
        )
        oceanbase.add_texts(texts, metadatas, ids=ids, extras=extras)
        return oceanbase

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Select the relevance score function based on the distance strategy.
        """
        if self.vidx_metric_type == "inner_product":
            return _neg_inner_product_similarity
        elif self.vidx_metric_type == "l2":
            return _euclidean_similarity
        elif self.vidx_metric_type == "cosine":
            # Cosine distance uses same similarity function as L2
            return _euclidean_similarity
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self.vidx_metric_type}."
            )

    def _get_distance_function(self, metric_type: str):
        """
        Get the appropriate distance function for the given metric type.
        """
        if metric_type == "inner_product":
            return inner_product
        elif metric_type == "l2":
            return l2_distance
        elif metric_type == "cosine":
            return cosine_distance
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")

    def _convert_list_results_to_documents(self, results):
        """
        Convert search results (list format) to Document objects.
        Used for converting combined results from hybrid search.
        """
        documents = []
        for result in results:
            doc_id = None
            page_content = ""
            metadata = {}

            if isinstance(result, dict):
                # Dictionary format: extract fields by name
                doc_id = result.get(self.primary_field)
                page_content = result.get(self.text_field, "")
                metadata = result.get(self.metadata_field, {})

                # Ensure metadata is a dict, not a JSON string
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
            else:
                # Tuple/row format: (text_field, metadata_field, primary_field)
                # Based on output_column_names order in ann_search
                metadata_raw = None
                if len(result) >= 3:
                    page_content = str(result[0]) if result[0] is not None else ""
                    metadata_raw = result[1] if result[1] is not None else {}
                    doc_id = result[2] if result[2] is not None else None
                elif len(result) == 2:
                    page_content = str(result[0]) if result[0] is not None else ""
                    metadata_raw = result[1] if result[1] is not None else {}
                else:
                    page_content = (
                        str(result[0])
                        if len(result) > 0 and result[0] is not None
                        else ""
                    )

                if metadata_raw is not None:
                    if isinstance(metadata_raw, str):
                        try:
                            metadata = json.loads(metadata_raw)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    else:
                        metadata = metadata_raw or {}

            documents.append(
                Document(id=doc_id, page_content=page_content, metadata=metadata)
            )

        return documents

    # Hybrid Search Methods
    def _add_documents_with_hybrid_field(
        self,
        documents: List[Document],
        hybrid_data: List[Any],
        table_name: str,
        create_table_func: Callable,
        field_name: str,
        field_value_getter: Callable[[Any], Any],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Internal helper method to add documents with hybrid search fields.

        Args:
            documents: List of documents to add
            hybrid_data: List of hybrid field data (sparse embeddings or fulltext content)
            table_name: Name of the table to upsert to
            create_table_func: Function to create the table if it doesn't exist
            field_name: Name of the hybrid field in the record
            field_value_getter: Function to extract the field value from hybrid_data item
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        # Generate dense embeddings to determine dimension
        try:
            sample_embedding = self.embedding_function.embed_query(
                documents[0].page_content
            )
            embedding_dim = len(sample_embedding)
        except Exception:
            raise ValueError(
                "Failed to generate embedding. Cannot determine embedding dimension."
            )

        # Create table if it doesn't exist
        if not self.obvector.check_table_exists(table_name):
            create_table_func(embedding_dim=embedding_dim)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Prepare data for insertion
        data = []
        for i, (doc, hybrid_item) in enumerate(zip(documents, hybrid_data)):
            # Generate dense embedding for the document
            dense_embedding = self.embedding_function.embed_query(doc.page_content)

            record = {
                self.primary_field: ids[i],
                self.text_field: doc.page_content,
                self.vector_field: dense_embedding,
                field_name: field_value_getter(hybrid_item),
                self.metadata_field: doc.metadata,
            }
            data.append(record)

        # Upsert data to table
        self.obvector.upsert(
            table_name=table_name,
            data=data,
            partition_name="",
        )
        return ids

    def add_sparse_documents(
        self,
        documents: List[Document],
        sparse_embeddings: List[Dict[int, float]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents with sparse vector embeddings.

        Note: This method uses upsert behavior. If documents with the same IDs already
        exist, they will be updated rather than creating duplicates. To add both sparse
        and fulltext fields to the same documents, use the same IDs for both calls, or
        use `add_documents_with_hybrid_fields` to add both in a single call.

        Args:
            documents: List of documents to add
            sparse_embeddings: List of sparse vector embeddings (dict of index: value)
            ids: Optional list of document IDs. If not provided, will be generated.
                 If provided and documents with these IDs already exist, they will be
                 updated (upsert behavior).

        Returns:
            List of document IDs

        Raises:
            ValueError: If sparse vector support is not enabled

        Example:
            .. code-block:: python

                # First add documents with fulltext
                ids = store.add_documents_with_fulltext(documents, fulltext_content)

                # Then add sparse embeddings to the same documents (using same IDs)
                store.add_sparse_documents(documents, sparse_embeddings, ids=ids)
                # This will update existing records instead of creating duplicates
        """
        if not self.include_sparse:
            raise ValueError(
                "Sparse vector support not enabled. Set include_sparse=True when "
                "initializing."
            )

        if len(documents) != len(sparse_embeddings):
            raise ValueError(
                "Number of documents must match number of sparse embeddings"
            )

        return self._add_documents_with_hybrid_field(
            documents=documents,
            hybrid_data=sparse_embeddings,
            table_name=self.sparse_table_name,
            create_table_func=self._create_sparse_table,
            field_name=self.sparse_vector_field,
            field_value_getter=lambda x: x,
            ids=ids,
        )

    def add_documents_with_fulltext(
        self,
        documents: List[Document],
        fulltext_content: List[str],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents with full-text content for full-text search.

        Note: This method uses upsert behavior. If documents with the same IDs already
        exist, they will be updated rather than creating duplicates. To add both sparse
        and fulltext fields to the same documents, use the same IDs for both calls, or
        use `add_documents_with_hybrid_fields` to add both in a single call.

        Args:
            documents: List of documents to add
            fulltext_content: List of full-text content strings
            ids: Optional list of document IDs. If not provided, will be generated.
                 If provided and documents with these IDs already exist, they will be
                 updated (upsert behavior).

        Returns:
            List of document IDs

        Raises:
            ValueError: If full-text search support is not enabled

        Example:
            .. code-block:: python

                # First add documents with sparse embeddings
                ids = store.add_sparse_documents(documents, sparse_embeddings)

                # Then add fulltext to the same documents (using same IDs)
                store.add_documents_with_fulltext(documents, fulltext_content, ids=ids)
                # This will update existing records instead of creating duplicates
        """
        if not self.include_fulltext:
            raise ValueError(
                "Full-text search support not enabled. Set include_fulltext=True "
                "when initializing."
            )

        if len(documents) != len(fulltext_content):
            raise ValueError(
                "Number of documents must match number of fulltext content items"
            )

        # Fulltext requires sparse support
        if not self.include_sparse:
            raise ValueError(
                "Full-text search requires sparse vector support. "
                "Set include_sparse=True when include_fulltext=True."
            )

        return self._add_documents_with_hybrid_field(
            documents=documents,
            hybrid_data=fulltext_content,
            table_name=self.fulltext_table_name,
            create_table_func=self._create_sparse_fulltext_table,
            field_name=self.fulltext_field,
            field_value_getter=lambda x: x,
            ids=ids,
        )

    def add_documents_with_hybrid_fields(
        self,
        documents: List[Document],
        sparse_embeddings: Optional[List[Dict[int, float]]] = None,
        fulltext_content: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents with optional sparse vectors and/or full-text content.
        This method allows adding all hybrid search fields in a single call,
        avoiding duplicate records when adding sparse and fulltext separately.

        Args:
            documents: List of documents to add
            sparse_embeddings: Optional list of sparse vector embeddings (dict of index: value)
            fulltext_content: Optional list of full-text content strings
            ids: Optional list of document IDs. If provided, must match documents length.
                 If not provided, will be generated. If documents already exist with these IDs,
                 they will be updated (upsert behavior).

        Returns:
            List of document IDs

        Raises:
            ValueError: If required features are not enabled or input validation fails

        Example:
            .. code-block:: python

                # Add documents with both sparse and fulltext in one call
                ids = hybrid_store.add_documents_with_hybrid_fields(
                    documents=docs,
                    sparse_embeddings=sparse_embeds,
                    fulltext_content=fulltext_contents,
                    ids=existing_ids  # Use same IDs to update existing records
                )
        """
        if sparse_embeddings is None and fulltext_content is None:
            # If neither sparse nor fulltext provided, use standard add_documents
            return self.add_documents(documents, ids=ids)

        if sparse_embeddings is not None and not self.include_sparse:
            raise ValueError(
                "Sparse vector support not enabled. Set include_sparse=True when "
                "initializing."
            )

        if fulltext_content is not None and not self.include_fulltext:
            raise ValueError(
                "Full-text search support not enabled. Set include_fulltext=True "
                "when initializing."
            )

        if sparse_embeddings is not None and len(documents) != len(sparse_embeddings):
            raise ValueError(
                "Number of documents must match number of sparse embeddings"
            )

        if fulltext_content is not None and len(documents) != len(fulltext_content):
            raise ValueError(
                "Number of documents must match number of fulltext content items"
            )

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        # Prepare data for upsert
        data = []
        for i, doc in enumerate(documents):
            # Generate dense embedding for the document
            dense_embedding = self.embedding_function.embed_query(doc.page_content)

            record = {
                self.primary_field: ids[i],
                self.text_field: doc.page_content,
                self.vector_field: dense_embedding,
                self.metadata_field: doc.metadata,
            }

            # Add sparse vector if provided
            if sparse_embeddings is not None:
                record[self.sparse_vector_field] = sparse_embeddings[i]

            # Add fulltext if provided
            if fulltext_content is not None:
                record[self.fulltext_field] = fulltext_content[i]

            data.append(record)

        # Upsert data using ObVecClient (upsert updates existing records with same ID)
        self.obvector.upsert(
            table_name=self.table_name,
            data=data,
            partition_name="",
        )
        return ids

    def similarity_search_with_sparse_vector(
        self,
        sparse_query: Dict[int, float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search using sparse vectors.

        Args:
            sparse_query: Sparse vector query (dict of index: value)
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar documents

        Raises:
            ValueError: If sparse vector support is not enabled
        """
        if not self.include_sparse:
            raise ValueError(
                "Sparse vector support not enabled. Set include_sparse=True when "
                "initializing."
            )

        # Determine which table to search
        # If hybrid fields were added to main table, search from main table
        # Otherwise, search from sparse table
        search_table = self.table_name
        if self.sparse_table_name and self.obvector.check_table_exists(
            self.sparse_table_name
        ):
            # Check if sparse table has data, prefer sparse table for sparse-only search
            try:
                with self.obvector.engine.connect() as conn:
                    from sqlalchemy import text

                    sparse_count = conn.execute(
                        text(f"SELECT COUNT(*) FROM `{self.sparse_table_name}`")
                    ).scalar()
                    if sparse_count > 0:
                        search_table = self.sparse_table_name
            except Exception:
                # If query fails, fall back to main table
                pass

        # Perform sparse vector search using ann_search
        res = self.obvector.ann_search(
            table_name=search_table,
            vec_data=sparse_query,
            vec_column_name=self.sparse_vector_field,
            distance_func=inner_product,
            topk=k,
            output_column_names=[
                self.text_field,
                self.metadata_field,
                self.primary_field,
            ],
            where_clause=filter,
        )

        return self._convert_results_to_documents(res, include_score=False)

    def _combine_hybrid_results(self, vector_results, fulltext_results, k: int) -> List:
        """
        Combine and rank results from vector and full-text search.

        Args:
            vector_results: Results from vector similarity search (CursorResult)
            fulltext_results: Results from full-text search (CursorResult)
            k: Number of final results to return

        Returns:
            Combined and ranked results
        """
        # Convert CursorResult to list of dictionaries
        vector_list = []
        if vector_results:
            for row in vector_results:
                if hasattr(row, "_asdict"):
                    vector_list.append(row._asdict())
                elif hasattr(row, "_mapping"):
                    vector_list.append(dict(row._mapping))
                else:
                    vector_list.append(row)

        fulltext_list = []
        if fulltext_results:
            for row in fulltext_results:
                if hasattr(row, "_asdict"):
                    fulltext_list.append(row._asdict())
                elif hasattr(row, "_mapping"):
                    fulltext_list.append(dict(row._mapping))
                else:
                    fulltext_list.append(row)

        # Create a dictionary to store combined scores
        combined_scores = {}

        # Process vector results (higher weight for semantic similarity)
        for i, result in enumerate(vector_list):
            if isinstance(result, dict):
                doc_id = result.get(self.primary_field)
            else:
                # For tuple format: (text_field, metadata_field, primary_field)
                doc_id = result[2] if len(result) > 2 else result[0]
            # Normalize to 0-1
            vector_score = 1.0 - (i / len(vector_list)) if vector_list else 0
            combined_scores[doc_id] = (
                combined_scores.get(doc_id, 0) + vector_score * 0.7
            )

        # Process full-text results (lower weight for keyword matching)
        for i, result in enumerate(fulltext_list):
            if isinstance(result, dict):
                doc_id = result.get(self.primary_field)
            else:
                # For tuple format: (text_field, metadata_field, primary_field)
                doc_id = result[2] if len(result) > 2 else result[0]
            # Normalize to 0-1
            fulltext_score = 1.0 - (i / len(fulltext_list)) if fulltext_list else 0
            combined_scores[doc_id] = (
                combined_scores.get(doc_id, 0) + fulltext_score * 0.3
            )

        # Sort by combined score and get top k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        # Create a mapping from doc_id to result for quick lookup
        # Prefer vector results over fulltext results when both exist
        id_to_result = {}
        for result in vector_list:
            if isinstance(result, dict):
                result_id = result.get(self.primary_field)
            else:
                # For tuple format: (text_field, metadata_field, primary_field)
                result_id = result[2] if len(result) > 2 else result[0]
            if result_id not in id_to_result:
                id_to_result[result_id] = result
        # Add fulltext results only if not already in the mapping
        for result in fulltext_list:
            if isinstance(result, dict):
                result_id = result.get(self.primary_field)
            else:
                # For tuple format: (text_field, metadata_field, primary_field)
                result_id = result[2] if len(result) > 2 else result[0]
            if result_id not in id_to_result:
                id_to_result[result_id] = result

        # Return results in the same format as input, ensuring no duplicates
        final_results = []
        seen_ids = set()
        for doc_id, score in sorted_docs:
            if doc_id not in seen_ids and doc_id in id_to_result:
                final_results.append(id_to_result[doc_id])
                seen_ids.add(doc_id)

        return final_results

    def _combine_multi_modal_results(
        self,
        all_results: List[tuple],
        k: int,
        modality_weights: Optional[Dict[str, float]] = None,
    ) -> List:
        """
        Combine and rank results from multiple search modalities.

        Args:
            all_results: List of (modality_type, results) tuples
            k: Number of final results to return
            modality_weights: Optional dictionary specifying weights for each modality

        Returns:
            Combined and ranked results
        """
        # Use provided weights or default weights
        if modality_weights is None:
            modality_weights = {
                "vector": 0.5,  # Semantic similarity
                "sparse": 0.3,  # Keyword matching
                "fulltext": 0.2,  # Text search
            }

        combined_scores = {}
        all_converted_results = {}  # Store converted results by modality

        # Process results from each modality
        for modality_type, results in all_results:
            weight = modality_weights.get(modality_type, 0.1)

            # Convert CursorResult to list if needed
            results_list = []
            if results:
                for row in results:
                    if hasattr(row, "_asdict"):
                        results_list.append(row._asdict())
                    elif hasattr(row, "_mapping"):
                        results_list.append(dict(row._mapping))
                    else:
                        results_list.append(row)

            # Store converted results for later use
            all_converted_results[modality_type] = results_list

            for i, result in enumerate(results_list):
                if isinstance(result, dict):
                    doc_id = result.get(self.primary_field)
                else:
                    # For tuple format: (text_field, metadata_field, primary_field)
                    doc_id = result[2] if len(result) > 2 else result[0]
                # Normalize score based on position (higher position = lower score)
                normalized_score = 1.0 - (i / len(results_list)) if results_list else 0
                combined_scores[doc_id] = (
                    combined_scores.get(doc_id, 0) + normalized_score * weight
                )

        # Sort by combined score and get top k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        # Create a mapping from doc_id to result for quick lookup
        # Prefer results from modalities in order: vector > sparse > fulltext
        id_to_result = {}
        modality_priority = ["vector", "sparse", "fulltext"]
        for modality_type in modality_priority:
            if modality_type in all_converted_results:
                for result in all_converted_results[modality_type]:
                    if isinstance(result, dict):
                        result_id = result.get(self.primary_field)
                    else:
                        # For tuple format: (text_field, metadata_field, primary_field)
                        result_id = result[2] if len(result) > 2 else result[0]
                    if result_id not in id_to_result:
                        id_to_result[result_id] = result

        # Return results in the same format as input, ensuring no duplicates
        final_results = []
        seen_ids = set()
        for doc_id, score in sorted_docs:
            if doc_id not in seen_ids and doc_id in id_to_result:
                final_results.append(id_to_result[doc_id])
                seen_ids.add(doc_id)

        return final_results

    def similarity_search_with_fulltext(
        self,
        query: str,
        fulltext_query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform hybrid search combining vector similarity and full-text search.

        This method performs a true hybrid search by:
        1. Using vector similarity search for semantic matching
        2. Using full-text search for keyword matching
        3. Combining and ranking results from both modalities

        Args:
            query: Vector similarity query
            fulltext_query: Full-text search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar documents ranked by hybrid score

        Raises:
            ValueError: If full-text search support is not enabled
        """
        if not self.include_fulltext:
            raise ValueError(
                "Full-text search support not enabled. Set include_fulltext=True "
                "when initializing."
            )

        # Fulltext requires sparse support
        if not self.include_sparse:
            raise ValueError(
                "Full-text search requires sparse vector support. "
                "Set include_sparse=True when include_fulltext=True."
            )

        # Determine which table to search
        # If hybrid fields were added to main table, search from main table
        # Otherwise, search from fulltext table
        search_table = self.table_name
        if self.fulltext_table_name and self.obvector.check_table_exists(
            self.fulltext_table_name
        ):
            # Check if fulltext table has data, prefer fulltext table for fulltext search
            try:
                with self.obvector.engine.connect() as conn:
                    from sqlalchemy import text

                    fulltext_count = conn.execute(
                        text(f"SELECT COUNT(*) FROM `{self.fulltext_table_name}`")
                    ).scalar()
                    if fulltext_count > 0:
                        search_table = self.fulltext_table_name
            except Exception:
                # If query fails, fall back to main table
                pass

        # Step 1: Vector similarity search
        query_embedding = self.embedding_function.embed_query(query)
        vector_results = self.obvector.ann_search(
            table_name=search_table,
            vec_data=query_embedding,
            vec_column_name=self.vector_field,
            distance_func=self._get_distance_function(self.vidx_metric_type),
            topk=k * 2,  # Get more results for better ranking
            where_clause=filter,
        )

        # Step 2: Full-text search using post_ann_search with str_list
        fulltext_results = self.obvector.post_ann_search(
            table_name=search_table,
            vec_data=query_embedding,
            vec_column_name=self.vector_field,
            distance_func=self._get_distance_function(self.vidx_metric_type),
            topk=k * 2,
            str_list=[fulltext_query],  # Use str_list for full-text search
            where_clause=filter,
        )

        # Step 3: Combine and rank results
        combined_results = self._combine_hybrid_results(
            vector_results, fulltext_results, k
        )

        return self._convert_list_results_to_documents(combined_results)

    def advanced_hybrid_search(
        self,
        vector_query: Optional[str] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        fulltext_query: Optional[str] = None,
        k: int = 4,
        modality_weights: Optional[Dict[str, float]] = None,
    ) -> List[Document]:
        """
        Perform advanced hybrid search combining multiple modalities.

        Args:
            vector_query: Optional vector similarity query
            sparse_query: Optional sparse vector query
            fulltext_query: Optional full-text search query
            k: Number of results to return
            modality_weights: Optional dictionary specifying weights for each modality.
                             Keys: 'vector', 'sparse', 'fulltext'
                             Values: float weights (should sum to 1.0)
                             Default: {'vector': 0.5, 'sparse': 0.3, 'fulltext': 0.2}

        Returns:
            List of similar documents

        Raises:
            ValueError: If no search modality is provided, required features are
                not enabled, or weights don't sum to 1.0
        """
        if not any([vector_query, sparse_query, fulltext_query]):
            raise ValueError("At least one search modality must be provided")

        if sparse_query and not self.include_sparse:
            raise ValueError(
                "Sparse vector support not enabled. Set include_sparse=True when "
                "initializing."
            )

        if fulltext_query and not self.include_fulltext:
            raise ValueError(
                "Full-text search support not enabled. Set include_fulltext=True "
                "when initializing."
            )

        # Validate and set up modality weights
        default_weights = {"vector": 0.5, "sparse": 0.3, "fulltext": 0.2}

        if modality_weights is None:
            modality_weights = default_weights
        else:
            # Validate weights
            total_weight = sum(modality_weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError(
                    f"Modality weights must sum to 1.0, got {total_weight}"
                )

            # Ensure all required keys are present
            for key in default_weights.keys():
                if key not in modality_weights:
                    modality_weights[key] = 0.0

        # Collect results from all available modalities
        all_results = []

        # Vector similarity search
        if vector_query:
            query_embedding = self.embedding_function.embed_query(vector_query)
            vector_results = self.obvector.ann_search(
                table_name=self.table_name,
                vec_data=query_embedding,
                vec_column_name=self.vector_field,
                distance_func=self._get_distance_function(self.vidx_metric_type),
                topk=k * 2,
            )
            all_results.append(("vector", vector_results))

        # Sparse vector search (on sparse table)
        if sparse_query:
            # Check if sparse table exists before searching
            if self.sparse_table_name and self.obvector.check_table_exists(
                self.sparse_table_name
            ):
                sparse_results = self.obvector.ann_search(
                    table_name=self.sparse_table_name,
                    vec_data=sparse_query,
                    vec_column_name=self.sparse_vector_field,
                    distance_func=inner_product,
                    topk=k * 2,
                )
                all_results.append(("sparse", sparse_results))
            # If table doesn't exist, skip sparse search (table may not have been created yet)

        # Full-text search (on sparse_fulltext table)
        if fulltext_query:
            # Check if fulltext table exists before searching
            if self.fulltext_table_name and self.obvector.check_table_exists(
                self.fulltext_table_name
            ):
                if vector_query:
                    # Use the same embedding for consistency
                    query_embedding = self.embedding_function.embed_query(vector_query)
                else:
                    # Create a dummy embedding for full-text only search
                    query_embedding = [0.0] * 6

                fulltext_results = self.obvector.post_ann_search(
                    table_name=self.fulltext_table_name,
                    vec_data=query_embedding,
                    vec_column_name=self.vector_field,
                    distance_func=self._get_distance_function(self.vidx_metric_type),
                    topk=k * 2,
                    str_list=[fulltext_query],
                )
                all_results.append(("fulltext", fulltext_results))
            # If table doesn't exist, skip fulltext search (table may not have been created yet)

        # Combine results from all modalities
        if len(all_results) == 1:
            # Single modality, return as is
            single_results = all_results[0][1]
            # Convert CursorResult to list and take first k results
            results_list = []
            if single_results:
                for i, row in enumerate(single_results):
                    if i >= k:
                        break
                    if hasattr(row, "_asdict"):
                        results_list.append(row._asdict())
                    elif hasattr(row, "_mapping"):
                        results_list.append(dict(row._mapping))
                    else:
                        results_list.append(row)
            return self._convert_list_results_to_documents(results_list)
        else:
            # Multiple modalities, combine and rank
            combined_results = self._combine_multi_modal_results(
                all_results, k, modality_weights
            )
            return self._convert_list_results_to_documents(combined_results)

    def similarity_search_with_advanced_filters(
        self,
        query: str,
        filters: Dict[str, Any],
        k: int = 4,
    ) -> List[Document]:
        """
        Perform similarity search with advanced filtering capabilities.

        Args:
            query: Vector similarity query
            filters: Dictionary of filters including scalar, fulltext, and
                metadata filters
            k: Number of results to return

        Returns:
            List of similar documents
        """
        # Perform advanced filtering with multiple search modalities
        query_embedding = self.embedding_function.embed_query(query)

        # Extract different types of filters
        where_clause = None
        fulltext_query = None
        # scalar_filters = []  # Currently not used in implementation

        if "metadata" in filters:
            where_clause = filters["metadata"]
        if "fulltext" in filters:
            fulltext_query = filters["fulltext"]
        # Extract scalar filters if present (currently not used in implementation)
        # if 'scalar' in filters:
        #     scalar_filters = filters['scalar']

        # Collect results from different search modalities
        all_results = []

        # Vector similarity search with metadata filtering
        vector_results = self.obvector.ann_search(
            table_name=self.table_name,
            vec_data=query_embedding,
            vec_column_name=self.vector_field,
            distance_func=self._get_distance_function(self.vidx_metric_type),
            topk=k * 2,
            where_clause=where_clause,
        )
        all_results.append(("vector", vector_results))

        # Full-text search if specified (on sparse_fulltext table)
        if fulltext_query and self.include_fulltext:
            fulltext_results = self.obvector.post_ann_search(
                table_name=self.fulltext_table_name,
                vec_data=query_embedding,
                vec_column_name=self.vector_field,
                distance_func=self._get_distance_function(self.vidx_metric_type),
                topk=k * 2,
                str_list=[fulltext_query],
                where_clause=where_clause,
            )
            all_results.append(("fulltext", fulltext_results))

        # Combine results if multiple modalities
        if len(all_results) == 1:
            # Single modality, convert CursorResult to list
            single_results = all_results[0][1]
            results_list = []
            if single_results:
                for i, row in enumerate(single_results):
                    if i >= k:
                        break
                    if hasattr(row, "_asdict"):
                        results_list.append(row._asdict())
                    elif hasattr(row, "_mapping"):
                        results_list.append(dict(row._mapping))
                    else:
                        results_list.append(row)
            return self._convert_list_results_to_documents(results_list)
        else:
            combined_results = self._combine_multi_modal_results(all_results, k)
            return self._convert_list_results_to_documents(combined_results)
