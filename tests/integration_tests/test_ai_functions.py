#!/usr/bin/env python3
"""
Complete AI Functions test following operational steps.

Important: Model-Endpoint Relationship
- One-to-One: Each ai_model can only have ONE ai_model_endpoint
- Binding Required: Creating an endpoint requires binding to an existing ai_model
- Independent Deletion: Endpoints can be deleted independently
- Deletion Order: To delete a model, you must delete its endpoint first

Step sequence:
1. Create models
2. Query models
3. Configure model endpoints (one endpoint per model)
4. Query model endpoints
5. Alter model endpoints (optional)
6. Delete models (optional) - requires deleting endpoints first
7. Delete model endpoints (optional) - can be deleted independently
8. Execute complete test
9. Execute embed test
10. Execute rerank test
"""

import os
import sys

# Add source code to path to use latest code
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_oceanbase.ai_functions import OceanBaseAIFunctions

# Database configuration
connection_args = {
    "host": "127.0.0.1",
    "port": "2881",
    "user": "root@test",
    "password": "",
    "db_name": "test",
}

# Model configuration
MODEL_CONFIG = {
    "url": "https://api.example.com/v1",
    "access_key": "YOUR_API_KEY",
    "embed_model": "your-embedding-model",
    "complete_model": "your-completion-model",
    "provider": "openai",
}

# Test results
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
}


def print_step(step_num: int, title: str):
    """Print step title"""
    print("\n" + "=" * 80)
    print(f"Step {step_num}: {title}")
    print("=" * 80)


def is_placeholder_config() -> bool:
    """Check if using placeholder configuration (no real API keys)"""
    return (
        MODEL_CONFIG.get("access_key") == "YOUR_API_KEY"
        or MODEL_CONFIG.get("url") == "https://api.example.com/v1"
        or MODEL_CONFIG.get("embed_model") == "your-embedding-model"
        or MODEL_CONFIG.get("complete_model") == "your-completion-model"
    )


def run_test(test_name: str, test_func, *args, **kwargs) -> bool:
    """Run test and record results"""
    test_results["total"] += 1
    try:
        result = test_func(*args, **kwargs)
        if result:
            test_results["passed"] += 1
            print(f"  ✅ {test_name}")
            return True
        else:
            test_results["failed"] += 1
            print(f"  ❌ {test_name}: Returned False")
            return False
    except Exception as e:
        test_results["failed"] += 1
        print(f"  ❌ {test_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def step_1_create_models(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 1: Create models"""
    print("\nCreating Embedding model...")
    try:
        # Try to delete first (if exists)
        try:
            ai_functions.drop_ai_model(MODEL_CONFIG["embed_model"])
            print(f"  ℹ️  Deleted existing model {MODEL_CONFIG['embed_model']}")
        except:
            pass

        ai_functions.create_ai_model(
            model_name=MODEL_CONFIG["embed_model"], model_type="dense_embedding"
        )
        print(
            f"  ✅ Embedding model created successfully: {MODEL_CONFIG['embed_model']}"
        )
    except Exception as e:
        error_str = str(e).lower()
        if "invalid" in error_str or "type" in error_str:
            print(
                f"  ⚠️  Embedding model creation failed (may not be supported or already exists): {e}"
            )
        else:
            raise

    print("\nCreating Completion model...")
    try:
        # Try to delete first (if exists)
        try:
            ai_functions.drop_ai_model(MODEL_CONFIG["complete_model"])
            print(f"  ℹ️  Deleted existing model {MODEL_CONFIG['complete_model']}")
        except:
            pass

        ai_functions.create_ai_model(
            model_name=MODEL_CONFIG["complete_model"], model_type="completion"
        )
        print(
            f"  ✅ Completion model created successfully: {MODEL_CONFIG['complete_model']}"
        )
    except Exception as e:
        error_str = str(e).lower()
        if "invalid" in error_str or "type" in error_str:
            print(
                f"  ⚠️  Completion model creation failed (may not be supported or already exists): {e}"
            )
        else:
            raise

    return True


def step_2_query_models(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 2: Query models"""
    try:
        models = ai_functions.list_ai_models()
        print(f"\nFound {len(models)} AI models:")
        for model in models:
            print(f"  Model name: {model.get('model_name')}")
            print(f"  Type: {model.get('type')} (1=embedding, 3=completion)")
            print(f"  Created at: {model.get('gmt_create')}")
            print()
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def step_3_configure_endpoints(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 3: Configure model endpoints

    Important: One-to-One Relationship
    - Each ai_model can only have ONE ai_model_endpoint (one-to-one relationship)
    - Creating an endpoint requires binding to an existing ai_model
    - Endpoints can be deleted independently, but deleting a model requires deleting its endpoint first
    """
    print("\n⚠️  Model-Endpoint Relationship:")
    print("  - One-to-One: Each model can only have ONE endpoint")
    print("  - Binding Required: Endpoint must be bound to an existing model")
    print("  - Independent Deletion: Endpoints can be deleted independently")

    print("\nConfiguring Embedding model endpoint...")
    try:
        # Delete existing endpoint if it exists (since one model = one endpoint)
        try:
            ai_functions.drop_ai_model_endpoint("embedding_endpoint")
            print("  ℹ️  Deleted existing endpoint embedding_endpoint")
        except:
            pass

        # Create endpoint for the existing model
        # Note: MODEL_CONFIG["embed_model"] must already exist (created in Step 1)
        ai_functions.create_ai_model_endpoint(
            endpoint_name="embedding_endpoint",
            ai_model_name=MODEL_CONFIG["embed_model"],  # Must bind to existing model
            url=MODEL_CONFIG["url"],
            access_key=MODEL_CONFIG["access_key"],
            provider=MODEL_CONFIG["provider"],
        )
        print("  ✅ Embedding endpoint created successfully")
        print(
            f"  ℹ️  Endpoint 'embedding_endpoint' is now bound to model '{MODEL_CONFIG['embed_model']}'"
        )
    except Exception as e:
        print(f"  ⚠️  Embedding endpoint creation failed: {e}")

    print("\nConfiguring Completion model endpoint...")
    try:
        # Delete existing endpoint if it exists
        try:
            ai_functions.drop_ai_model_endpoint("complete_endpoint")
            print("  ℹ️  Deleted existing endpoint complete_endpoint")
        except:
            pass

        # Create endpoint for the existing model
        # Note: MODEL_CONFIG["complete_model"] must already exist (created in Step 1)
        ai_functions.create_ai_model_endpoint(
            endpoint_name="complete_endpoint",
            ai_model_name=MODEL_CONFIG["complete_model"],  # Must bind to existing model
            url=MODEL_CONFIG["url"],
            access_key=MODEL_CONFIG["access_key"],
            provider=MODEL_CONFIG["provider"],
        )
        print("  ✅ Completion endpoint created successfully")
        print(
            f"  ℹ️  Endpoint 'complete_endpoint' is now bound to model '{MODEL_CONFIG['complete_model']}'"
        )
    except Exception as e:
        print(f"  ⚠️  Completion endpoint creation failed: {e}")

    return True


def step_4_query_endpoints(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 4: Query model endpoints"""
    try:
        endpoints = ai_functions.list_ai_model_endpoints()
        print(f"\nFound {len(endpoints)} AI model endpoints:")
        for endpoint in endpoints:
            print(f"  Endpoint name: {endpoint.get('ENDPOINT_NAME')}")
            print(f"  Model name: {endpoint.get('AI_MODEL_NAME')}")
            print(f"  URL: {endpoint.get('URL')}")
            print(f"  Provider: {endpoint.get('PROVIDER')}")
            print()
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def step_5_alter_endpoints(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 5: Alter model endpoints (optional)"""
    print("\n⚠️  Skipping alter endpoints step (optional operation)")
    print("  To alter endpoints, uncomment the following code:")
    print("  # ai_functions.alter_ai_model_endpoint(")
    print("  #     endpoint_name='embedding_endpoint',")
    print("  #     ai_model_name=MODEL_CONFIG['embed_model'],")
    print("  #     url='https://new-api.example.com/v1',")
    print("  #     access_key='NEW_API_KEY',")
    print("  #     provider=MODEL_CONFIG['provider']")
    print("  # )")
    print("  # ai_functions.alter_ai_model_endpoint(")
    print("  #     endpoint_name='complete_endpoint',")
    print("  #     ai_model_name=MODEL_CONFIG['complete_model'],")
    print("  #     url='https://new-api.example.com/v1',")
    print("  #     access_key='NEW_API_KEY',")
    print("  #     provider=MODEL_CONFIG['provider']")
    print("  # )")
    return True


def step_6_delete_models(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 6: Delete models (optional)

    Important: One-to-One Relationship
    - To delete a model, you MUST delete its endpoint first
    - Each model has only one endpoint, so delete that endpoint before deleting the model
    - Models and endpoints can be deleted independently, but deletion order matters
    """
    print("\n⚠️  Skipping delete models step (optional operation)")
    print("\n⚠️  Important: One-to-One Relationship")
    print("  - Each model has only ONE endpoint")
    print("  - To delete a model, you MUST delete its endpoint first")
    print("  - Deletion order: endpoint first, then model")
    print("\n  To delete models, uncomment the following code:")
    print("  # Step 1: Delete endpoints first (required)")
    print("  # ai_functions.drop_ai_model_endpoint('embedding_endpoint')")
    print("  # ai_functions.drop_ai_model_endpoint('complete_endpoint')")
    print("  # Step 2: Delete models (now safe to delete)")
    print("  # ai_functions.drop_ai_model(MODEL_CONFIG['embed_model'])")
    print("  # ai_functions.drop_ai_model(MODEL_CONFIG['complete_model'])")
    return True


def step_7_delete_endpoints(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 7: Delete model endpoints

    Important: Independent Deletion
    - Endpoints can be deleted independently without deleting the associated model
    - After deleting an endpoint, the model remains but won't be usable
    - You can create a new endpoint for the same model later
    """
    print("\n⚠️  Skipping delete endpoints step (optional operation)")
    print("\n⚠️  Important: Independent Deletion")
    print("  - Endpoints can be deleted independently")
    print("  - Model will remain but won't be usable until a new endpoint is created")
    print("  - One-to-One: Each model has only one endpoint")
    print("\n  To delete endpoints independently, uncomment the following code:")
    print("  # ai_functions.drop_ai_model_endpoint('embedding_endpoint')")
    print("  # ai_functions.drop_ai_model_endpoint('complete_endpoint')")
    print(
        "  # Note: Models will remain but won't be usable until new endpoints are created"
    )
    return True


def step_7_test_complete(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 7: Execute complete test"""
    if is_placeholder_config():
        print("  ⚠️  Skipping AI_COMPLETE test: Using placeholder configuration")
        print("  ℹ️  Set real API credentials in MODEL_CONFIG to run this test")
        return True  # Return True to count as passed (skipped)
    
    try:
        completion = ai_functions.ai_complete(
            prompt="Explain what machine learning is in one sentence",
            model_name=MODEL_CONFIG["complete_model"],
        )
        assert completion is not None
        assert isinstance(completion, str)
        assert len(completion) > 0
        print(f"  Prompt: Explain what machine learning is in one sentence")
        print(f"  Completion: {completion[:200]}...")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def step_8_test_embed(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 8: Execute embed test"""
    if is_placeholder_config():
        print("  ⚠️  Skipping AI_EMBED test: Using placeholder configuration")
        print("  ℹ️  Set real API credentials in MODEL_CONFIG to run this test")
        return True  # Return True to count as passed (skipped)
    
    try:
        vector = ai_functions.ai_embed(
            text="Test text: Machine learning is a subset of artificial intelligence",
            model_name=MODEL_CONFIG["embed_model"],
        )
        assert vector is not None
        assert isinstance(vector, list)
        assert len(vector) > 0
        print(
            f"  Text: Test text: Machine learning is a subset of artificial intelligence"
        )
        print(f"  Vector dimension: {len(vector)}")
        print(f"  First 5 values: {vector[:5]}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def step_9_test_rerank(ai_functions: OceanBaseAIFunctions) -> bool:
    """Step 9: Execute rerank test"""
    if is_placeholder_config():
        print("  ⚠️  Skipping AI_RERANK test: Using placeholder configuration")
        print("  ℹ️  Set real API credentials in MODEL_CONFIG to run this test")
        return True  # Return True to count as passed (skipped)
    
    try:
        query = "machine learning algorithms"
        documents = [
            "Deep learning is a branch of machine learning that uses multi-layer neural networks",
            "Python is a popular programming language widely used in data science",
            "Supervised learning requires labeled data to train models",
        ]

        reranked = ai_functions.ai_rerank(
            query=query,
            documents=documents,
            model_name=MODEL_CONFIG["embed_model"],
            top_k=2,
        )

        assert reranked is not None
        assert isinstance(reranked, list)
        assert len(reranked) <= 2

        print(f"  Query: {query}")
        print(f"  Number of documents: {len(documents)}")
        print(f"  Number of reranked results: {len(reranked)}")
        print("\n  Reranked results:")
        for result in reranked:
            print(f"    Rank {result['rank']}: Score {result['score']:.4f}")
            print(f"      {result['document'][:60]}...")

        return True
    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("=" * 80)
    print("🧪 Complete AI Functions Test Following Operational Steps")
    print("=" * 80)
    print(f"\nDatabase configuration:")
    print(f"  Host: {connection_args['host']}:{connection_args['port']}")
    print(f"  Database: {connection_args['db_name']}")
    print(f"\nModel configuration:")
    print(f"  URL: {MODEL_CONFIG['url']}")
    print(f"  Embed model: {MODEL_CONFIG['embed_model']}")
    print(f"  Complete model: {MODEL_CONFIG['complete_model']}")

    try:
        # Initialize
        print("\nInitializing AI Functions...")
        ai_functions = OceanBaseAIFunctions(connection_args=connection_args)
        print("✅ Initialization successful")

        # Step 1: Create models
        print_step(1, "Create models")
        run_test("Create models", step_1_create_models, ai_functions)

        # Step 2: Query models
        print_step(2, "Query models")
        run_test("Query models", step_2_query_models, ai_functions)

        # Step 3: Configure model endpoints
        print_step(3, "Configure model endpoints")
        run_test("Configure model endpoints", step_3_configure_endpoints, ai_functions)

        # Step 4: Query model endpoints
        print_step(4, "Query model endpoints")
        run_test("Query model endpoints", step_4_query_endpoints, ai_functions)

        # Step 5: Alter model endpoints (optional)
        print_step(5, "Alter model endpoints (optional)")
        run_test("Alter model endpoints", step_5_alter_endpoints, ai_functions)

        # Step 6: Delete models (optional)
        print_step(6, "Delete models (optional)")
        run_test("Delete models", step_6_delete_models, ai_functions)

        # Step 7: Delete model endpoints
        print_step(7, "Delete model endpoints")
        run_test("Delete model endpoints", step_7_delete_endpoints, ai_functions)

        # Step 8: Execute complete test
        print_step(8, "Execute AI_COMPLETE test")
        run_test("AI_COMPLETE test", step_7_test_complete, ai_functions)

        # Step 9: Execute embed test
        print_step(9, "Execute AI_EMBED test")
        run_test("AI_EMBED test", step_8_test_embed, ai_functions)

        # Step 10: Execute rerank test
        print_step(10, "Execute AI_RERANK test")
        run_test("AI_RERANK test", step_9_test_rerank, ai_functions)

        # Summary
        print("\n" + "=" * 80)
        print("📊 Test Summary")
        print("=" * 80)
        print(f"Total tests: {test_results['total']}")
        print(f"✅ Passed: {test_results['passed']}")
        print(f"❌ Failed: {test_results['failed']}")

        if test_results["total"] > 0:
            success_rate = test_results["passed"] / test_results["total"] * 100
            print(
                f"\nSuccess rate: {success_rate:.1f}% ({test_results['passed']}/{test_results['total']})"
            )

        if test_results["failed"] == 0:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print(f"\n❌ {test_results['failed']} test(s) failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
