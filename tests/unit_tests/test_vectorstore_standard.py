import json
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from langchain_oceanbase import OceanbaseVectorStore

class TestOceanbaseVectorStoreUnit:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        # Mock check_table_exists to avoid table creation logic during init if needed
        client.check_table_exists.return_value = True
        return client

    @pytest.fixture
    def vectorstore(self, mock_client):
        # Patch ObVecClient where it is imported in the module
        with patch("langchain_oceanbase.vectorstores.ObVecClient", return_value=mock_client):
            # Patch sqlalchemy.Table to return expected columns during _load_table
            with patch("langchain_oceanbase.vectorstores.Table") as MockTable:
                mock_table_instance = MagicMock()
                # Define columns expected by _load_table
                c1, c2, c3, c4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
                c1.name = "id"
                c2.name = "embedding"
                c3.name = "document"
                c4.name = "metadata"
                mock_table_instance.columns = [c1, c2, c3, c4]
                MockTable.return_value = mock_table_instance

                store = OceanbaseVectorStore(
                    embedding_function=MagicMock(),
                    connection_args={
                        "host": "localhost",
                        "port": "2881",
                        "user": "root",
                        "password": "",
                        "db_name": "test"
                    },
                    table_name="test_table",
                    drop_old=False
                )
                # Force set the client just in case
                store.obvector = mock_client
                return store

    def test_add_texts(self, vectorstore, mock_client):
        """Test adding texts calls upsert correctly."""
        texts = ["foo", "bar"]
        metadatas = [{"source": "1"}, {"source": "2"}]
        ids = ["1", "2"]

        # Mock embedding function
        vectorstore.embedding_function.embed_documents.return_value = [[1.0]*384, [2.0]*384]

        # Mock _create_table_with_index to avoid table loading logic
        vectorstore._create_table_with_index = MagicMock()

        # Run
        result_ids = vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        # Verify
        assert result_ids == ids
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args.kwargs
        assert call_kwargs['table_name'] == "test_table"
        assert len(call_kwargs['data']) == 2
        assert call_kwargs['data'][0]['document'] == "foo"
        assert call_kwargs['data'][0]['id'] == "1"

    def test_similarity_search(self, vectorstore, mock_client):
        """Test similarity search calls ann_search correctly."""
        # Mock embedding
        vectorstore.embedding_function.embed_query.return_value = [1.0]*384

        # Mock search result
        mock_result = MagicMock()
        # fetchall return format: [(text, metadata, id)]
        mock_result.fetchall.return_value = [
            ("foo", json.dumps({"source": "1"}), "1")
        ]
        mock_client.ann_search.return_value = mock_result

        # Run
        docs = vectorstore.similarity_search("query", k=1)

        # Verify
        assert len(docs) == 1
        assert docs[0].page_content == "foo"
        assert docs[0].metadata["source"] == "1"
        mock_client.ann_search.assert_called_once()

    def test_delete(self, vectorstore, mock_client):
        """Test delete calls client delete."""
        vectorstore.delete(["1"])
        mock_client.delete.assert_called_with(
            table_name="test_table",
            ids=["1"],
            where_clause=None
        )
