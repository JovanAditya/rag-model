#!/usr/bin/env python3
"""Test script for unified ChromaDB + BM25 indexing system."""

import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_model.indexing.unified_index_manager import UnifiedIndexManager
from rag_model.core.config import RAGConfig


def load_test_documents(file_path: str = "data/processed/chunks.json"):
    """Load test documents for indexing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"✅ Loaded {len(documents)} documents from {file_path}")
        return documents
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return []


def test_unified_indexing():
    """Test the unified indexing system."""
    print("🧪 Testing Unified Indexing System")
    print("=" * 50)

    # Load test documents
    documents = load_test_documents()
    if not documents:
        print("❌ No test documents available")
        return

    # Use subset for testing
    test_docs = documents[:50]  # Use first 50 documents
    print(f"📄 Using {len(test_docs)} documents for testing")

    # Create configuration
    config = RAGConfig()

    # Prepare configuration for unified manager
    vector_config = {
        "collection_name": config.index.chroma_collection,
        "persist_directory": config.index.chroma_path,
        "embedding_model": config.embedding.model_name
    }

    bm25_config = {
        "k1": config.bm25.k1,
        "b": config.bm25.b,
        "ngram_range": (config.bm25.ngram_range_min, config.bm25.ngram_range_max)
    }

    try:
        # Initialize unified index manager
        print("\n🔧 Initializing Unified Index Manager...")
        manager = UnifiedIndexManager(
            vector_config=vector_config,
            bm25_config=bm25_config,
            cache_dir="./test_cache"
        )

        # Index documents
        print("\n📊 Indexing documents...")
        start_time = time.time()
        manager.index_documents(test_docs)
        indexing_time = time.time() - start_time

        print(f"✅ Indexing completed in {indexing_time:.2f} seconds")
        print(f"📈 Average time per document: {(indexing_time / len(test_docs) * 1000):.1f}ms")

        # Get statistics
        stats = manager.get_unified_stats()
        print("\n📊 Unified Statistics:")
        print(f"  Total documents: {stats['unified_manager']['indexed_documents']}")
        print(f"  Status: {stats['unified_manager']['status']}")

        if 'vector_store' in stats:
            print(f"  Vector store documents: {stats['vector_store'].get('document_count', 'N/A')}")
        if 'bm25_index' in stats:
            print(f"  BM25 vocabulary: {stats['bm25_index'].get('vocabulary_size', 'N/A')}")
            print(f"  BM25 avg doc length: {stats['bm25_index'].get('avg_doc_length', 'N/A'):.1f}")

        # Health check
        print("\n🏥 Health Check:")
        health = manager.health_check()
        print(f"  Overall status: {health['status']}")
        for component, status in health['components'].items():
            print(f"  {component}: {status}")

        if health.get('issues'):
            print("  Issues:")
            for issue in health['issues']:
                print(f"    - {issue}")

        # Test search functionality
        print("\n🔍 Testing Search Functionality:")
        test_queries = [
            "Universitas Mercu Buana",
            "visi misi",
            "fakultas teknik",
            "program studi",
            "lokasi kampus"
        ]

        for query in test_queries:
            print(f"\n  Query: '{query}'")
            try:
                search_start = time.time()
                results = manager.search_unified(
                    query=query,
                    k=5,
                    vector_weight=0.6,
                    bm25_weight=0.4,
                    strategy="rrf"
                )
                search_time = time.time() - search_start

                print(f"    Results: {len(results['results'])} documents")
                print(f"    Time: {search_time:.3f}s")
                print(f"    Stats: {results['stats']}")

                if results['results']:
                    top_result = results['results'][0]
                    score_key = 'rrf_score' if 'rrf_score' in top_result else 'weighted_score'
                    print(f"    Top score: {top_result.get(score_key, 0):.4f}")
                    print(f"    Top source: {top_result.get('text', '')[:100]}...")

            except Exception as e:
                print(f"    ❌ Error: {e}")

        print("\n✅ Unified indexing test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_strategies():
    """Test different fusion strategies."""
    print("\n🔄 Testing Fusion Strategies")
    print("=" * 50)

    # Load documents and setup manager
    documents = load_test_documents()
    if len(documents) < 20:
        print("❌ Not enough documents for fusion testing")
        return

    test_docs = documents[:20]  # Use smaller set for fusion testing

    config = RAGConfig()
    vector_config = {
        "collection_name": "test_fusion_collection",
        "persist_directory": "./test_chroma_db_fusion",
        "embedding_model": config.embedding.model_name
    }
    bm25_config = {
        "k1": config.bm25.k1,
        "b": config.bm25.b,
        "ngram_range": (1, 2)
    }

    try:
        manager = UnifiedIndexManager(
            vector_config=vector_config,
            bm25_config=bm25_config,
            cache_dir="./test_cache_fusion"
        )

        print("📊 Indexing test documents...")
        manager.index_documents(test_docs)

        test_query = "Universitas Mercu Buana visi misi"
        strategies = ["rrf", "weighted", "max", "simple"]

        print(f"\n🔍 Testing query: '{test_query}'")
        for strategy in strategies:
            try:
                results = manager.search_unified(
                    query=test_query,
                    k=5,
                    vector_weight=0.6,
                    bm25_weight=0.4,
                    strategy=strategy
                )

                print(f"\n  {strategy.upper()} Strategy:")
                print(f"    Results: {len(results['results'])}")
                print(f"    Time: {results['stats']['total_time']:.3f}s")

                if results['results']:
                    top_result = results['results'][0]
                    print(f"    Top source: {top_result.get('fusion_source', 'N/A')}")
                    if 'rrf_score' in top_result:
                        print(f"    RRF score: {top_result['rrf_score']:.4f}")
                    elif 'weighted_score' in top_result:
                        print(f"    Weighted score: {top_result['weighted_score']:.4f}")
                    elif 'max_score' in top_result:
                        print(f"    Max score: {top_result['max_score']:.4f}")

            except Exception as e:
                print(f"    ❌ Error: {e}")

        print("\n✅ Fusion strategy testing completed!")

    except Exception as e:
        print(f"❌ Fusion testing failed: {e}")
        import traceback
        traceback.print_exc()


def benchmark_performance():
    """Benchmark performance of unified indexing."""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)

    documents = load_test_documents()
    if len(documents) < 100:
        print("❌ Not enough documents for benchmarking")
        return

    # Test with different document counts
    test_sizes = [10, 25, 50, 100]
    test_query = "Universitas Mercu Buana"

    for size in test_sizes:
        if size > len(documents):
            continue

        print(f"\n📊 Testing with {size} documents:")
        test_docs = documents[:size]

        config = RAGConfig()
        vector_config = {
            "collection_name": f"benchmark_{size}_docs",
            "persist_directory": f"./test_chroma_db_benchmark_{size}",
            "embedding_model": config.embedding.model_name
        }
        bm25_config = {
            "k1": config.bm25.k1,
            "b": config.bm25.b,
            "ngram_range": (1, 2)
        }

        try:
            manager = UnifiedIndexManager(
                vector_config=vector_config,
                bm25_config=bm25_config,
                cache_dir=f"./test_cache_benchmark_{size}"
            )

            # Index documents
            start_time = time.time()
            manager.index_documents(test_docs)
            indexing_time = time.time() - start_time

            # Test search
            search_start = time.time()
            results = manager.search_unified(
                query=test_query,
                k=5,
                strategy="rrf"
            )
            search_time = time.time() - search_start

            print(f"  Indexing: {indexing_time:.2f}s ({(indexing_time/size*1000):.1f}ms/doc)")
            print(f"  Search: {search_time:.3f}s")
            print(f"  Results: {len(results['results'])}")

        except Exception as e:
            print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Starting Unified Indexing System Tests")
    print("=" * 60)

    # Test basic functionality
    success = test_unified_indexing()

    if success:
        # Test fusion strategies
        test_fusion_strategies()

        # Benchmark performance
        benchmark_performance()

    print("\n🏁 Testing completed!")
    print("=" * 60)