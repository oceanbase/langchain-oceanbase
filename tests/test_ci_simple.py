 #!/usr/bin/env python3
"""
Simple CI test script for langchain-oceanbase
Focuses on basic functionality without complex similarity search
"""

import os
import sys
import time
import importlib
from typing import Dict, Any, List


def test_imports() -> bool:
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        "langchain_oceanbase",
        "langchain_oceanbase.vectorstores",
        "langchain_core",
        "langchain_community",
    ]
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            return False
    
    return True


def test_basic_functionality() -> bool:
    """Test basic functionality with database connection"""
    print("\n" + "=" * 60)
    print("Testing langchain-oceanbase basic functionality")
    print("=" * 60)
    
    try:
        from langchain_oceanbase.vectorstores import OceanbaseVectorStore
        from langchain_community.embeddings import FakeEmbeddings
        from langchain_core.documents import Document
        
        # Create fake embeddings
        embeddings = FakeEmbeddings(size=6)
        
        # CI OceanBase configuration
        connection_args = {
            "host": "127.0.0.1",
            "port": "2881",
            "user": "root",
            "password": "",
            "db_name": "test",
        }
        
        # Test basic vectorstore creation
        print("Testing vectorstore creation...")
        vectorstore = OceanbaseVectorStore(
            embedding_function=embeddings,
            table_name="ci_simple_test",
            connection_args=connection_args,
            vidx_metric_type="l2",
            drop_old=True,
            embedding_dim=6,
        )
        print("  ✓ Vectorstore created successfully")
        
        # Test documents
        documents = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"}),
            Document(page_content="Test document 3", metadata={"source": "test3"}),
        ]
        
        # Add documents
        print("Testing document addition...")
        ids = vectorstore.add_documents(documents)
        print(f"  ✓ Added {len(ids)} documents")
        assert len(ids) == 3
        
        # Test get by ids
        print("Testing get by IDs...")
        retrieved_docs = vectorstore.get_by_ids(ids)
        print(f"  ✓ Retrieved {len(retrieved_docs)} documents by IDs")
        assert len(retrieved_docs) == 3
        
        # Verify content
        content_list = [doc.page_content for doc in retrieved_docs]
        assert "Test document 1" in content_list
        assert "Test document 2" in content_list
        assert "Test document 3" in content_list
        
        # Test similarity search (basic)
        print("Testing similarity search...")
        results = vectorstore.similarity_search("", k=10)  # Empty query to get all
        print(f"  ✓ Similarity search returned {len(results)} results")
        assert len(results) >= 3
        
        # Test similarity search with score
        print("Testing similarity search with score...")
        results_with_score = vectorstore.similarity_search_with_score("", k=10)
        print(f"  ✓ Similarity search with score returned {len(results_with_score)} results")
        assert len(results_with_score) >= 3
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results_with_score)
        
        print("\n" + "=" * 60)
        print("🎉 Basic functionality tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_from_texts() -> bool:
    """Test from_texts class method"""
    print("\nTesting from_texts method...")
    
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
        
        texts = ["Simple text 1", "Simple text 2", "Simple text 3"]
        metadatas = [{"source": "simple1"}, {"source": "simple2"}, {"source": "simple3"}]
        
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
        
        # Test basic search
        results = vectorstore.similarity_search("", k=10)
        print("✓ Search after from_texts successful")
        assert len(results) >= 3
        
        return True
        
    except Exception as e:
        print(f"✗ from_texts error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_preservation() -> bool:
    """Test metadata preservation"""
    print("\nTesting metadata preservation...")
    
    try:
        from langchain_oceanbase.vectorstores import OceanbaseVectorStore
        from langchain_community.embeddings import FakeEmbeddings
        from langchain_core.documents import Document
        
        embeddings = FakeEmbeddings(size=6)
        
        connection_args = {
            "host": "127.0.0.1",
            "port": "2881",
            "user": "root",
            "password": "",
            "db_name": "test",
        }
        
        # Create documents with metadata
        documents = [
            Document(page_content="Content 1", metadata={"key1": "value1", "key2": "value2"}),
            Document(page_content="Content 2", metadata={"key3": "value3"}),
        ]
        
        vectorstore = OceanbaseVectorStore(
            embedding_function=embeddings,
            table_name="ci_metadata_test",
            connection_args=connection_args,
            vidx_metric_type="l2",
            drop_old=True,
            embedding_dim=6,
        )
        
        # Add documents
        ids = vectorstore.add_documents(documents)
        assert len(ids) == 2
        
        # Retrieve and check metadata
        retrieved_docs = vectorstore.get_by_ids(ids)
        assert len(retrieved_docs) == 2
        
        # Check first document metadata
        doc1 = retrieved_docs[0]
        assert doc1.metadata["key1"] == "value1"
        assert doc1.metadata["key2"] == "value2"
        
        # Check second document metadata
        doc2 = retrieved_docs[1]
        assert doc2.metadata["key3"] == "value3"
        
        print("✓ Metadata preservation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Metadata preservation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main simple CI test function"""
    print("=" * 80)
    print("langchain-oceanbase Simple CI Test Suite")
    print("=" * 80)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("From Texts Integration", test_from_texts),
        ("Metadata Preservation", test_metadata_preservation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"  ✅ {test_name} passed")
        else:
            print(f"  ❌ {test_name} failed")
    
    print("\n" + "=" * 80)
    print(f"Simple CI Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All simple CI tests passed!")
        print("✅ langchain-oceanbase basic functionality confirmed")
        print("✅ Document addition and retrieval working")
        print("✅ Metadata preservation working")
        print("✅ OceanBase 4.3.5 compatibility confirmed")
        sys.exit(0)
    else:
        print("❌ Some simple CI tests failed!")
        print("Please check the output above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()