#!/usr/bin/env python3
"""
CI-specific test script for langchain-oceanbase
"""

import os
import sys
import time


def test_basic_functionality():
    """Test basic functionality in CI environment"""
    print("=" * 60)
    print("Testing langchain-oceanbase in CI")
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
        
        # Test different index types
        index_types = ["HNSW", "IVF", "FLAT"]
        
        for index_type in index_types:
            print(f"\nTesting {index_type} index type...")
            
            try:
                # Create vectorstore
                vectorstore = OceanbaseVectorStore(
                    embedding_function=embeddings,
                    table_name=f"ci_test_{index_type.lower()}",
                    connection_args=connection_args,
                    vidx_metric_type="l2",
                    drop_old=True,
                    embedding_dim=6,
                    index_type=index_type,
                )
                
                # Test documents
                documents = [
                    Document(page_content="Hello world from CI", metadata={"source": "ci_test1"}),
                    Document(page_content="Python programming in CI", metadata={"source": "ci_test2"}),
                    Document(page_content="Machine learning test", metadata={"source": "ci_test3"}),
                ]
                
                # Add documents
                ids = vectorstore.add_documents(documents)
                print(f"  ✓ Added {len(ids)} documents")
                
                # Test similarity search
                results = vectorstore.similarity_search("Hello", k=2)
                print(f"  ✓ Similarity search returned {len(results)} results")
                
                # Test similarity search with score
                results_with_score = vectorstore.similarity_search_with_score("Python", k=2)
                print(f"  ✓ Similarity search with score returned {len(results_with_score)} results")
                
                # Test get by ids
                if ids:
                    retrieved_docs = vectorstore.get_by_ids(ids[:2])
                    print(f"  ✓ Retrieved {len(retrieved_docs)} documents by IDs")
                
                print(f"  ✓ {index_type} test passed")
                
            except Exception as e:
                print(f"  ✗ Error testing {index_type}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n" + "=" * 60)
        print("🎉 All CI tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ CI test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metric_types():
    """Test different metric types"""
    print("\nTesting metric types...")
    
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
        
        metric_types = ["l2", "inner_product", "cosine"]
        
        for metric_type in metric_types:
            print(f"  Testing {metric_type} metric...")
            
            try:
                vectorstore = OceanbaseVectorStore(
                    embedding_function=embeddings,
                    table_name=f"ci_metric_{metric_type}",
                    connection_args=connection_args,
                    vidx_metric_type=metric_type,
                    drop_old=True,
                    embedding_dim=6,
                )
                
                documents = [
                    Document(page_content=f"Test document for {metric_type}", metadata={"metric": metric_type}),
                ]
                
                ids = vectorstore.add_documents(documents)
                results = vectorstore.similarity_search("Test", k=1)
                print(f"    ✓ {metric_type} metric test passed")
                
            except Exception as e:
                print(f"    ✗ Error with {metric_type}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Metric types test error: {e}")
        return False


if __name__ == "__main__":
    success = True
    
    if not test_basic_functionality():
        success = False
        
    if not test_metric_types():
        success = False
        
    if success:
        print("\n🎉 All CI tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some CI tests failed!")
        sys.exit(1)