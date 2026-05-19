#!/usr/bin/env python3
"""
Index Building Script for Academic RAG

This script builds unified ChromaDB + BM25 indexes from processed data.
Configuration via .env file or command line arguments.

Usage:
    python scripts/build_indexes.py  # Uses .env defaults
    python scripts/build_indexes.py --documents ../data/processed/chunks.json --verify
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system env vars

# Add rag_model to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rag_model.indexing import UnifiedIndexManager
    from rag_model.core.config import RAGConfig, BM25Config, IndexConfig
    import torch
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Please install: pip install torch sentence-transformers chromadb scikit-learn")
    sys.exit(1)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Build indexes for AcademicRAG system."""

    def __init__(
        self,
        embedding_model: str = "indobenchmark/indobert-base-p2",
        device: str = "auto",
        batch_size: int = 1000,
        verify: bool = False
    ):
        # Auto-detect device if requested
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.verify = verify

        if self.device == "cuda":
            try:
                import torch
                logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            except ImportError:
                logger.warning("⚠️  CUDA device requested but torch not available, falling back to CPU")
                self.device = "cpu"

    def _validate_documents(self, documents_file: str) -> bool:
        """Validate processed documents file."""
        logger.info(f"📖 Validating documents: {documents_file}")

        try:
            with open(documents_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error("❌ Documents file should contain a list of chunks")
                return False

            if len(data) == 0:
                logger.error("❌ No documents found in chunks file")
                return False

            # Sample validation of first few documents
            valid_chunks = 0
            for i, doc in enumerate(data[:10]):
                if isinstance(doc, dict) and 'text' in doc and 'metadata' in doc:
                    if len(doc.get('text', '').strip()) > 10:
                        valid_chunks += 1
                else:
                    logger.warning(f"⚠️  Invalid chunk structure at index {i}")

            valid_percentage = (valid_chunks / min(10, len(data))) * 100
            logger.info(f"✅ Documents validated: {len(data)} total chunks, {valid_percentage:.1f}% valid")

            if valid_percentage < 80:
                logger.warning(f"⚠️  Low quality chunks detected: {valid_percentage:.1f}% valid")

            return True

        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error validating documents: {e}")
            return False

    def _verify_only(self, documents_file: str, collection_name: str, chroma_dir: str, cache_dir: str = "./cache") -> bool:
        """Verify existing unified indexes without rebuilding."""
        logger.info("🔍 Verifying existing unified indexes...")

        try:
            # Setup unified index manager for verification
            vector_config = {
                "collection_name": collection_name,
                "persist_directory": chroma_dir,
                "embedding_model": self.embedding_model
            }

            bm25_config = {
                "k1": 1.5,
                "b": 0.75,
                "ngram_range": (1, 2)
            }

            index_manager = UnifiedIndexManager(
                vector_config=vector_config,
                bm25_config=bm25_config,
                cache_dir=cache_dir
            )

            # Initialize indexes before health check
            index_manager.initialize_indexes()

            # Check unified index health
            health = index_manager.health_check()

            if health["status"] == "healthy":
                print("✅ Unified index verification PASSED")
                stats = index_manager.get_unified_stats()

                # Display structured index statistics
                print(f"\n{'─'*60}")
                print(f"📊 STATISTIK INDEKS")
                print(f"{'─'*60}")

                # BM25 Index stats
                bm25_stats = stats.get('bm25_index', {})
                print(f"\n   ╔{'═'*56}╗")
                print(f"   ║ {'BM25 INDEX (scikit-learn TfidfVectorizer)':<54s} ║")
                print(f"   ╠{'═'*56}╣")
                print(f"   ║ {'Jumlah dokumen':<30s} : {str(bm25_stats.get('documents_count', 'N/A')):<22s} ║")
                print(f"   ║ {'Ukuran vocabulary':<30s} : {str(bm25_stats.get('vocabulary_size', 'N/A')):<22s} ║")
                print(f"   ║ {'Parameter k1':<30s} : {str(bm25_stats.get('k1', '1.5')):<22s} ║")
                print(f"   ║ {'Parameter b':<30s} : {str(bm25_stats.get('b', '0.75')):<22s} ║")
                print(f"   ║ {'N-gram range':<30s} : {str(bm25_stats.get('ngram_range', '(1, 2)')):<22s} ║")
                print(f"   ╚{'═'*56}╝")

                # Vector Store stats
                vector_stats = stats.get('vector_store', {})
                print(f"\n   ╔{'═'*56}╗")
                print(f"   ║ {'VECTOR INDEX (ChromaDB + IndoBERT)':<54s} ║")
                print(f"   ╠{'═'*56}╣")
                print(f"   ║ {'Collection name':<30s} : {str(vector_stats.get('collection_name', 'N/A')):<22s} ║")
                print(f"   ║ {'Jumlah embedding':<30s} : {str(vector_stats.get('document_count', 'N/A')):<22s} ║")
                print(f"   ║ {'Dimensi vektor':<30s} : {'768':<22s} ║")
                print(f"   ║ {'Model embedding':<30s} : {'indobert-base-p2':<22s} ║")
                print(f"   ╚{'═'*56}╝")

                return True
            else:
                print("❌ Unified index verification FAILED")
                print(f"   Status: {health['status']}")
                for issue in health.get('issues', []):
                    print(f"   Issue: {issue}")
                return False

        except Exception as e:
            logger.error(f"❌ Error verifying unified indexes: {e}")
            return False

    def build_unified_indexes(self, documents_file: str, collection_name: str,
                          chroma_dir: str, cache_dir: str = "./cache",
                          skip_existing: bool = False) -> bool:
        """Build unified ChromaDB + BM25 indexes."""

        # Check if indexes already exist (for --skip-existing)
        if skip_existing:
            chroma_path = Path(chroma_dir)
            if chroma_path.exists():
                logger.info(f"⚠️  ChromaDB directory already exists: {chroma_path}")

                # Quick verification by checking if collection exists
                try:
                    import chromadb
                    client = chromadb.PersistentClient(str(chroma_dir))
                    collections = client.list_collections()
                    if any(col.name == collection_name for col in collections):
                        logger.info("✅ Existing unified index verified, skipping rebuild")
                        return True
                    else:
                        logger.info("⚠️  Collection not found, rebuilding...")
                except Exception as e:
                    logger.warning(f"⚠️  Could not verify existing index: {e}")
                    logger.info("Rebuilding...")

        logger.info("🏗 Building unified ChromaDB + BM25 indexes...")
        start_time = time.time()

        try:
            # Load documents
            with open(documents_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            logger.info(f"📄 Loaded {len(documents)} documents for indexing")

            # Setup unified index manager
            vector_config = {
                "collection_name": collection_name,
                "persist_directory": chroma_dir,
                "embedding_model": self.embedding_model
            }

            bm25_config = {
                "k1": 1.5,
                "b": 0.75,
                "ngram_range": (1, 2)
            }

            index_manager = UnifiedIndexManager(
                vector_config=vector_config,
                bm25_config=bm25_config,
                cache_dir=cache_dir
            )

            # Clear existing indexes to prevent duplicates
            logger.info("🗑️  Clearing existing indexes to prevent duplicates...")
            try:
                index_manager.clear_indexes()
            except Exception as clear_err:
                logger.debug(f"Note: {clear_err} (this is normal for new indexes)")

            # Build unified indexes
            logger.info(f"🏗 Building unified index: {collection_name}")

            index_manager.index_documents(documents)

            build_time = time.time() - start_time

            if self.verify:
                logger.info("🔍 Verifying unified index...")
                health = index_manager.health_check()

                if health["status"] == "healthy":
                    logger.info("✅ Unified index built and verified successfully!")

                stats = index_manager.get_unified_stats()

                # Display structured index statistics
                print(f"\n{'─'*60}")
                print(f"📊 STATISTIK INDEKS")
                print(f"{'─'*60}")

                # BM25 Index stats
                bm25_stats = stats.get('bm25_index', {})
                print(f"\n   ╔{'═'*56}╗")
                print(f"   ║ {'BM25 INDEX (scikit-learn TfidfVectorizer)':<54s} ║")
                print(f"   ╠{'═'*56}╣")
                print(f"   ║ {'Jumlah dokumen':<30s} : {str(bm25_stats.get('documents_count', 'N/A')):<22s} ║")
                print(f"   ║ {'Ukuran vocabulary':<30s} : {str(bm25_stats.get('vocabulary_size', 'N/A')):<22s} ║")
                print(f"   ║ {'Parameter k1':<30s} : {str(bm25_stats.get('k1', '1.5')):<22s} ║")
                print(f"   ║ {'Parameter b':<30s} : {str(bm25_stats.get('b', '0.75')):<22s} ║")
                print(f"   ║ {'N-gram range':<30s} : {str(bm25_stats.get('ngram_range', '(1, 2)')):<22s} ║")
                print(f"   ╚{'═'*56}╝")

                # Vector Store stats
                vector_stats = stats.get('vector_store', {})
                print(f"\n   ╔{'═'*56}╗")
                print(f"   ║ {'VECTOR INDEX (ChromaDB + IndoBERT)':<54s} ║")
                print(f"   ╠{'═'*56}╣")
                print(f"   ║ {'Collection name':<30s} : {str(vector_stats.get('collection_name', 'N/A')):<22s} ║")
                print(f"   ║ {'Jumlah embedding':<30s} : {str(vector_stats.get('document_count', 'N/A')):<22s} ║")
                print(f"   ║ {'Dimensi vektor':<30s} : {'768':<22s} ║")
                print(f"   ║ {'Model embedding':<30s} : {'indobert-base-p2':<22s} ║")
                print(f"   ╚{'═'*56}╝")

                if health.get("status") != "healthy":
                    logger.warning("⚠️  Index built but verification found issues:")
                    logger.warning(f"   Status: {health['status']}")
                    for issue in health.get('issues', []):
                        logger.warning(f"   Issue: {issue}")

            logger.info(f"⏱️  Unified index building completed in {build_time:.2f}s")
            logger.info(f"📈 Average time per document: {(build_time / len(documents) * 1000):.1f}ms")

            return True

        except Exception as e:
            logger.error(f"❌ Error building unified index: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    # Get defaults from environment variables
    default_processed = os.getenv("PROCESSED_DIR", "../data/processed")
    default_documents = f"{default_processed}/chunks.json"
    default_collection = os.getenv("COLLECTION_NAME", "academic_docs")
    default_chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "../data/chroma_db")
    default_cache_dir = os.getenv("INDEX_CACHE_DIR", "../data/cache")
    default_embedding = os.getenv("EMBEDDING_MODEL", "indobenchmark/indobert-base-p2")
    default_device = os.getenv("EMBEDDING_DEVICE", "auto")
    default_batch_size = int(os.getenv("BATCH_SIZE", "1000"))

    parser = argparse.ArgumentParser(
        description="Build unified ChromaDB + BM25 indexes for Academic RAG"
    )

    parser.add_argument(
        "--documents", "-d",
        default=default_documents,
        help=f"Path to processed chunks JSON file (default: {default_documents})"
    )

    parser.add_argument(
        "--collection", "-c",
        default=default_collection,
        help=f"Name for ChromaDB collection (default: {default_collection})"
    )

    parser.add_argument(
        "--chroma-dir",
        default=default_chroma_dir,
        help=f"Directory for ChromaDB storage (default: {default_chroma_dir})"
    )

    parser.add_argument(
        "--cache-dir",
        default=default_cache_dir,
        help=f"Directory for unified index cache (default: {default_cache_dir})"
    )

    parser.add_argument(
        "--embedding-model",
        default=default_embedding,
        help=f"Embedding model name (default: {default_embedding})"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default=default_device,
        help=f"Device for embedding computation (default: {default_device})"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help=f"Batch size for embedding computation (default: {default_batch_size})"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify indexes after building"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip building if indexes already exist and are healthy"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing indexes, don't rebuild"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild indexes (overwrite existing)"
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old indexes after successful build (use with caution)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🏗 ACADEMIC RAG UNIFIED INDEX BUILDER")
    print("=" * 50)
    print(f"📁 Documents: {args.documents}")
    print(f"📋 Collection: {args.collection}")
    print(f"🗂️  ChromaDB: {args.chroma_dir}")
    print(f"💾 Cache: {args.cache_dir}")
    print(f"🤖 Model: {args.embedding_model}")
    print(f"💻 Device: {args.device}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"✅ Verify: {args.verify}")
    print(f"⚡ Skip existing: {args.skip_existing}")
    print(f"💪 Force rebuild: {args.force}")
    print()

    # Initialize index builder
    builder = IndexBuilder(
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        verify=args.verify
    )

    # Validate documents file
    if not builder._validate_documents(args.documents):
        sys.exit(1)

    # Handle different operation modes
    if args.verify_only:
        # Simple verification - check if collection exists and has documents
        success = builder._verify_only(
            documents_file=args.documents,
            collection_name=args.collection,
            chroma_dir=args.chroma_dir,
            cache_dir=args.cache_dir
        )
    elif args.force or not args.skip_existing:
        success = builder.build_unified_indexes(
            documents_file=args.documents,
            collection_name=args.collection,
            chroma_dir=args.chroma_dir,
            cache_dir=args.cache_dir,
            skip_existing=args.skip_existing
        )
    else:  # skip_existing without force
        success = builder.build_unified_indexes(
            documents_file=args.documents,
            collection_name=args.collection,
            chroma_dir=args.chroma_dir,
            cache_dir=args.cache_dir,
            skip_existing=True
        )

    print("\n" + "=" * 50)
    if success:
        if args.verify_only:
            print("✅ Index verification completed successfully!")
        else:
            print("✅ Index building completed successfully!")
            print("\n💡 Next steps:")
            print("   1. Verify system: python scripts/verify_system.py")
            print("   2. Test usage: python examples/basic_usage.py")
    else:
        if args.verify_only:
            print("❌ Index verification failed!")
        else:
            print("❌ Index building failed!")
        print("\n💡 Troubleshooting:")
        print("   1. Check if Ollama is running: ollama list")
        print("   2. Check processed data: python scripts/verify_system.py --component data")
        print("   3. Verify dependencies: python scripts/verify_system.py --component environment")
        print("   4. Ensure sufficient memory and disk space")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
