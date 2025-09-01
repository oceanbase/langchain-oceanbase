#!/usr/bin/env python3
"""
Integration test for from_texts method
"""


def test_from_texts():
    """Test from_texts class method in CI"""
    print("Testing from_texts method...")
    
    try:
        from langchain_oceanbase.vectorstores import OceanbaseVectorStore
        from langchain_community.embeddings import FakeEmbeddings
        
        embeddings = FakeEmbeddings(size=6)
        
        connection_args = {
            "host": "127.0.0.1",
            "port": "2881",
            "user": "root",
            "password": "",
            "db_name": "test",
        }
        
        texts = ["First CI text", "Second CI text", "Third CI text"]
        metadatas = [{"source": "ci1"}, {"source": "ci2"}, {"source": "ci3"}]
        
        vectorstore = OceanbaseVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            table_name="ci_from_texts",
            connection_args=connection_args,
            vidx_metric_type="l2",
            drop_old=True,
            embedding_dim=6,
        )
        
        print("✓ from_texts method successful")
        
        # Test search
        results = vectorstore.similarity_search("First", k=1)
        print("✓ Search after from_texts successful")
        
        return True
        
    except Exception as e:
        print(f"✗ from_texts error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if test_from_texts():
        print("🎉 Integration test passed!")
    else:
        print("❌ Integration test failed!")
        exit(1)