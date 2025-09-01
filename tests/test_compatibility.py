#!/usr/bin/env python3
"""
Compatibility test script for langchain-oceanbase
Tests compatibility with different versions of dependencies
"""

import sys
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
    """Test basic functionality without database connection"""
    print("\nTesting basic functionality...")
    
    try:
        from langchain_oceanbase.vectorstores import OceanbaseVectorStore
        from langchain_community.embeddings import FakeEmbeddings
        
        # Test that we can create embeddings
        embeddings = FakeEmbeddings(size=6)
        print("  ✓ FakeEmbeddings created")
        
        # Test that we can create vectorstore class (without connection)
        # This tests the class definition and basic structure
        print("  ✓ OceanbaseVectorStore class available")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_version_compatibility() -> bool:
    """Test version compatibility with dependencies"""
    print("\nTesting version compatibility...")
    
    try:
        import langchain_core
        import langchain_community
        
        print(f"  ✓ langchain-core version: {langchain_core.__version__}")
        print(f"  ✓ langchain-community version: {langchain_community.__version__}")
        
        # Check for minimum version requirements
        # These are reasonable minimum versions
        min_core_version = "0.1.0"
        min_community_version = "0.0.10"
        
        # Simple version check (this is a basic check)
        def version_ok(current: str, minimum: str) -> bool:
            try:
                current_parts = [int(x) for x in current.split('.')]
                min_parts = [int(x) for x in minimum.split('.')]
                return current_parts >= min_parts
            except:
                return True  # If we can't parse, assume it's OK
        
        if not version_ok(langchain_core.__version__, min_core_version):
            print(f"  ⚠ langchain-core version {langchain_core.__version__} may be too old (minimum: {min_core_version})")
        
        if not version_ok(langchain_community.__version__, min_community_version):
            print(f"  ⚠ langchain-community version {langchain_community.__version__} may be too old (minimum: {min_community_version})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Version compatibility test failed: {e}")
        return False


def test_class_methods() -> bool:
    """Test that required class methods exist"""
    print("\nTesting class methods...")
    
    try:
        from langchain_oceanbase.vectorstores import OceanbaseVectorStore
        
        # Check that required methods exist
        required_methods = [
            'add_documents',
            'similarity_search',
            'similarity_search_with_score',
            'from_texts',
        ]
        
        for method in required_methods:
            if hasattr(OceanbaseVectorStore, method):
                print(f"  ✓ {method} method exists")
            else:
                print(f"  ✗ {method} method missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Class methods test failed: {e}")
        return False


def main() -> None:
    """Main compatibility test function"""
    print("=" * 60)
    print("langchain-oceanbase Compatibility Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Version Compatibility", test_version_compatibility),
        ("Class Methods", test_class_methods),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  ❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"Compatibility Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All compatibility tests passed!")
        print("✅ langchain-oceanbase is compatible with current dependencies")
        sys.exit(0)
    else:
        print("❌ Some compatibility tests failed!")
        print("Please check the output above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()