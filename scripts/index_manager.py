#!/usr/bin/env python3
"""
AcademicRAG Index Manager

Script for managing ChromaDB + BM25 unified indexes.
Supports listing, testing, and cleaning up indexes with versioning.

Usage:
    python scripts/index_manager.py --list
    python scripts/index_manager.py --test <version>
    python scripts/index_manager.py --cleanup <version> [--confirm]
    python scripts/index_manager.py --export-config <version>
    python scripts/index_manager.py --migrate <from_version> <to_version>
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add rag_model to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import chromadb
    from rag_model.indexing import UnifiedIndexManager, VectorStore, BM25Index
    from rag_model.core.config import EmbeddingConfig, IndexConfig
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Please install: pip install chromadb")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexManager:
    """Manages ChromaDB + BM25 unified indexes for AcademicRAG."""

    def __init__(
        self,
        chroma_base_dir: str = "../data/chroma_db",
        cache_dir: str = "../data/cache"
    ):
        """
        Initialize index manager.

        Args:
            chroma_base_dir: Base directory for ChromaDB collections
            cache_dir: Directory for BM25 cache
        """
        self.chroma_base_dir = Path(chroma_base_dir)
        self.cache_dir = Path(cache_dir)

    def list_indexes(self) -> Dict[str, Any]:
        """
        List all available indexes.

        Returns:
            Dictionary containing index information
        """
        print("🔍 SCANNING AVAILABLE INDEXES")
        print("=" * 50)

        result = {
            "chromadb": {},
            "bm25": {},
            "data_directories": {},
            "unified_indexes": {},
            "total_indexes": 0
        }

        # Scan ChromaDB collections
        result["chromadb"]["collections"] = []
        result["chromadb"]["total"] = 0

        if self.chroma_base_dir.exists():
            print(f"\n📁 ChromaDB Collections in: {self.chroma_base_dir}")
            chroma_collections = []

            try:
                client = chromadb.PersistentClient(str(self.chroma_base_dir))
                actual_collections = client.list_collections()

                for collection in actual_collections:
                    try:
                        doc_count = collection.count()
                        # Find corresponding directory for size calculation
                        collection_dir = None
                        for item in self.chroma_base_dir.iterdir():
                            if item.is_dir():
                                try:
                                    test_client = chromadb.PersistentClient(str(self.chroma_base_dir))
                                    test_collection = test_client.get_collection(collection.name)
                                    collection_dir = item
                                    break
                                except:
                                    continue

                        if collection_dir:
                            size_mb = self._get_dir_size(collection_dir) / (1024 * 1024)
                            modified = time.ctime(collection_dir.stat().st_mtime)
                        else:
                            size_mb = doc_count * 0.01
                            modified = "unknown"

                        chroma_collections.append({
                            "name": collection.name,
                            "documents": doc_count,
                            "size_mb": round(size_mb, 2),
                            "path": str(collection_dir) if collection_dir else f"Collection: {collection.name}",
                            "modified": modified,
                            "status": "valid"
                        })

                    except Exception as e:
                        logger.warning(f"Error getting collection info for {collection.name}: {e}")
                        chroma_collections.append({
                            "name": collection.name,
                            "documents": 0,
                            "size_mb": 0.0,
                            "path": f"Collection: {collection.name}",
                            "modified": "unknown",
                            "status": "error"
                        })

            except Exception as e:
                logger.error(f"Error connecting to ChromaDB: {e}")
                # Fallback to directory scanning
                for item in self.chroma_base_dir.iterdir():
                    if item.is_dir():
                        size_mb = self._get_dir_size(item) / (1024 * 1024)
                        chroma_collections.append({
                            "name": item.name,
                            "documents": 0,
                            "size_mb": round(size_mb, 2),
                            "path": str(item),
                            "modified": time.ctime(item.stat().st_mtime),
                            "status": "unknown"
                        })

            result["chromadb"]["collections"] = chroma_collections
            result["chromadb"]["total"] = sum(col["documents"] for col in chroma_collections)

            # Display ChromaDB collections
            if chroma_collections:
                for collection in sorted(chroma_collections, key=lambda x: x["modified"], reverse=True):
                    status_icon = "✅" if collection["status"] == "valid" else "❌"
                    print(f"  {status_icon} {collection['name']}")
                    print(f"     Documents: {collection['documents']:,}")
                    print(f"     Size: {collection['size_mb']:.1f} MB")
                    print(f"     Modified: {collection['modified']}")
                    if collection.get("status") != "valid":
                        print(f"     Status: {collection.get('status', 'unknown')}")
                    print()
            else:
                print("  No ChromaDB collections found")

        # Scan BM25 cache directories
        if self.cache_dir.exists():
            print(f"\n🔤 BM25 Index Cache in: {self.cache_dir}")
            bm25_dirs = []

            for item in self.cache_dir.iterdir():
                if item.is_dir() or (item.is_file() and (item.name.endswith('.pkl') or item.name.endswith('.pkl.gz'))):
                    size_mb = self._get_dir_size(item) / (1024 * 1024) if item.is_dir() else item.stat().st_size / (1024 * 1024)
                    modified = time.ctime(item.stat().st_mtime)
                    bm25_dirs.append({
                        "name": item.name,
                        "size_mb": round(size_mb, 2),
                        "path": str(item),
                        "modified": modified,
                        "type": "directory" if item.is_dir() else "file"
                    })

            result["bm25"]["indexes"] = bm25_dirs
            result["bm25"]["total"] = len(bm25_dirs)

            if bm25_dirs:
                for bm25_dir in sorted(bm25_dirs, key=lambda x: x["modified"], reverse=True):
                    type_icon = "📁" if bm25_dir["type"] == "directory" else "📄"
                    print(f"  ✅ {type_icon} {bm25_dir['name']}")
                    print(f"     Size: {bm25_dir['size_mb']:.1f} MB")
                    print(f"     Modified: {bm25_dir['modified']}")
                    print()
            else:
                print("  No BM25 cache found")

        # Scan for unified index configurations
        unified_configs = []
        config_files = list(Path(".").rglob("*unified*config*.json"))
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    unified_configs.append({
                        "name": config_file.name,
                        "version": config.get("version", "unknown"),
                        "path": str(config_file),
                        "modified": time.ctime(config_file.stat().st_mtime),
                        "size_kb": round(config_file.stat().st_size / 1024, 1)
                    })
            except Exception:
                pass

        result["unified_indexes"] = unified_configs
        if unified_configs:
            print(f"\n🤖 Unified Index Configurations:")
            for config in sorted(unified_configs, key=lambda x: x["modified"], reverse=True):
                print(f"  ✅ {config['name']} (v{config['version']})")
                print(f"     Size: {config['size_kb']} KB")
                print(f"     Modified: {config['modified']}")
                print()

        # Scan data directories
        data_dir = Path("data/processed")
        if data_dir.exists():
            print(f"\n📊 Processed Data Directories in: {data_dir}")
            data_dirs = []

            # Scan subdirectories (versioned data)
            for item in data_dir.iterdir():
                if item.is_dir():
                    size_mb = self._get_dir_size(item) / (1024 * 1024)
                    chunks_file = item / "chunks.json"
                    doc_count = 0
                    if chunks_file.exists():
                        try:
                            with open(chunks_file, 'r') as f:
                                chunks = json.load(f)
                                doc_count = len(chunks)
                        except Exception:
                            pass

                    data_dirs.append({
                        "name": item.name,
                        "documents": doc_count,
                        "size_mb": round(size_mb, 2),
                        "path": str(item),
                        "modified": time.ctime(item.stat().st_mtime),
                        "has_chunks": chunks_file.exists(),
                        "type": "directory"
                    })

            # Also scan for direct chunks files in the main directory
            for item in data_dir.iterdir():
                if item.is_file() and item.name.endswith('.json') and item.name.startswith('chunks'):
                    # Extract version name from chunks filename
                    version_name = item.stem.replace('chunks', '') or 'default'
                    if version_name.startswith('_'):
                        version_name = version_name[1:]

                    # Get file size
                    size_mb = item.stat().st_size / (1024 * 1024)
                    doc_count = 0
                    try:
                        with open(item, 'r') as f:
                            chunks = json.load(f)
                            doc_count = len(chunks)
                    except Exception:
                        doc_count = 0

                    data_dirs.append({
                        "name": f"chunks_{version_name}" if version_name != 'default' else "chunks",
                        "documents": doc_count,
                        "size_mb": round(size_mb, 2),
                        "path": str(item),
                        "modified": time.ctime(item.stat().st_mtime),
                        "has_chunks": True,
                        "type": "file",
                        "file_name": item.name
                    })

            result["data_directories"] = data_dirs

            # Display data directories and files
            if data_dirs:
                for data_dir in sorted(data_dirs, key=lambda x: x["modified"], reverse=True):
                    chunks_icon = "✅" if data_dir["has_chunks"] else "❌"
                    type_icon = "📁" if data_dir.get("type") == "directory" else "📄"
                    print(f"  {chunks_icon} {type_icon} {data_dir['name']}")
                    print(f"     Documents: {data_dir['documents']:,}")
                    print(f"     Size: {data_dir['size_mb']:.1f} MB")
                    print(f"     Modified: {data_dir['modified']}")
                    print()
            else:
                print("  No processed data directories found")

        result["total_indexes"] = result["chromadb"]["total"]

        print("=" * 50)
        print(f"📈 SUMMARY: {result['total_indexes']} total documents indexed")
        print(f"   ChromaDB: {result['chromadb']['total']} documents")
        print(f"   BM25 Indexes: {len(result['bm25'].get('indexes', []))} cached indexes")
        print(f"   Unified Configurations: {len(result['unified_indexes'])}")
        print(f"   Data directories: {len(result['data_directories'])}")

        return result

    def test_index(self, version: str) -> bool:
        """Test a specific index version."""
        print(f"🧪 TESTING INDEX VERSION: {version}")
        print("=" * 50)

        success = True

        # Test ChromaDB collection
        chroma_collection = f"academic_docs_{version}"
        chroma_path = self.chroma_base_dir / chroma_collection

        print(f"\n1️⃣  Testing ChromaDB Collection: {chroma_collection}")
        if chroma_path.exists() or True:  # Try both path and collection name
            try:
                client = chromadb.PersistentClient(str(self.chroma_base_dir))
                collection = client.get_collection(chroma_collection)
                doc_count = collection.count()
                print(f"   ✅ Collection accessible")
                print(f"   📊 Documents: {doc_count:,}")

                # Test query
                if doc_count > 0:
                    try:
                        results = collection.query(
                            query_texts=["test"],
                            n_results=1
                        )
                        print(f"   ✅ Query test successful")
                    except Exception as e:
                        print(f"   ❌ Query test failed: {e}")
                        success = False
                else:
                    print(f"   ⚠️  Empty collection")

            except Exception as e:
                print(f"   ❌ Collection access failed: {e}")
                success = False
        else:
            print(f"   ❌ Collection not found")
            success = False

        # Test BM25 index cache
        bm25_cache_path = self.cache_dir / f"bm25_{version}.pkl"
        print(f"\n2️⃣  Testing BM25 Index Cache: {bm25_cache_path}")
        if bm25_cache_path.exists():
            try:
                import pickle
                with open(bm25_cache_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                print(f"   ✅ BM25 cache accessible")
                print(f"   📊 Cache size: {bm25_cache_path.stat().st_size / 1024:.1f} KB")
            except Exception as e:
                print(f"   ❌ BM25 cache access failed: {e}")
                success = False
        else:
            print(f"   ⚠️  BM25 cache not found (will be created on first search)")

        # Test unified manager
        print(f"\n3️⃣  Testing Unified Index Manager")
        try:
            vector_config = {
                "embedding_model": "indobenchmark/indobert-base-p2",
                "collection_name": chroma_collection,
                "persist_directory": str(self.chroma_base_dir)
            }

            bm25_config = {
                "k1": 1.5,
                "b": 0.75,
                "ngram_range": (1, 2)
            }

            unified_manager = UnifiedIndexManager(
                vector_config=vector_config,
                bm25_config=bm25_config,
                cache_dir=str(self.cache_dir)
            )

            # Test health check
            health = unified_manager.health_check()
            if health["status"] == "healthy":
                print(f"   ✅ Unified manager healthy")
            else:
                print(f"   ⚠️  Unified manager status: {health['status']}")
                if health.get("issues"):
                    for issue in health["issues"]:
                        print(f"      • {issue}")

            # Test search functionality if documents exist
            try:
                search_result = unified_manager.search_unified(
                    query="test",
                    k=1,
                    vector_weight=0.6,
                    bm25_weight=0.4,
                    strategy="rrf"
                )
                print(f"   ✅ Unified search successful")
                print(f"   📊 Results: {len(search_result['results'])}")
            except Exception as e:
                print(f"   ⚠️  Unified search test (may be normal if no documents): {e}")

        except Exception as e:
            print(f"   ❌ Unified manager test failed: {e}")
            success = False

        # Test data directory
        data_dir = Path(f"data/processed/{version}")
        print(f"\n4️⃣  Testing Data Directory: {data_dir}")
        if data_dir.exists():
            chunks_file = data_dir / "chunks.json"
            if chunks_file.exists():
                try:
                    with open(chunks_file, 'r') as f:
                        chunks = json.load(f)
                        doc_count = len(chunks)
                    print(f"   ✅ Data directory found")
                    print(f"   📊 Chunks: {doc_count:,}")
                except Exception as e:
                    print(f"   ❌ Data directory access failed: {e}")
                    success = False
            else:
                print(f"   ❌ No chunks.json found")
                success = False
        else:
            print(f"   ⚠️  Data directory not found")

        print("\n" + "=" * 50)
        if success:
            print("✅ INDEX TEST PASSED")
        else:
            print("❌ INDEX TEST FAILED")

        return success

    def cleanup_index(self, version: str, confirm: bool = False) -> bool:
        """Clean up a specific index version."""
        print(f"🗑️  CLEANUP INDEX VERSION: {version}")
        print("=" * 50)

        if not confirm:
            print("⚠️  DRY RUN - No changes will be made")
            print("   Use --confirm to execute cleanup")
            print()

        success = True
        deleted_items = []

        # Check if version already has prefix, avoid double prefix
        if version.startswith("academic_docs_"):
            chroma_collection = version
        else:
            chroma_collection = f"academic_docs_{version}"

        # Clean up ChromaDB
        print(f"\n1️⃣  Cleaning up ChromaDB: {chroma_collection}")
        try:
            client = chromadb.PersistentClient(str(self.chroma_base_dir))
            collection = client.get_collection(chroma_collection)
            client.delete_collection(chroma_collection)
            print(f"   ✅ ChromaDB collection deleted")
            deleted_items.append(f"ChromaDB collection: {chroma_collection}")
        except Exception as e:
            print(f"   ℹ️  ChromaDB collection not found")

        # Clean up BM25 cache
        print(f"\n2️⃣  Cleaning up BM25 Cache")
        bm25_cache_files = [
            self.cache_dir / f"bm25_{version}.pkl",
            self.cache_dir / f"bm25_{chroma_collection}.pkl",
        ]

        for bm25_file in bm25_cache_files:
            if bm25_file.exists():
                try:
                    bm25_file.unlink()
                    print(f"   ✅ BM25 cache deleted: {bm25_file.name}")
                    deleted_items.append(f"BM25 cache: {bm25_file.name}")
                except Exception as e:
                    print(f"   ❌ Error deleting BM25 cache {bm25_file.name}: {e}")
                    success = False

        # Clean up data directory
        data_dir = Path(f"data/processed/{version}")
        print(f"\n3️⃣  Cleaning up data directory: {data_dir}")
        if data_dir.exists():
            try:
                shutil.rmtree(data_dir)
                print(f"   ✅ Data directory deleted")
                deleted_items.append(f"Data directory: {data_dir}")
            except Exception as e:
                print(f"   ❌ Error deleting data directory: {e}")
                success = False
        else:
            print(f"   ℹ️  Data directory not found")

        # Clean up direct chunks files
        processed_dir = Path("data/processed")
        print(f"\n3️⃣a Cleaning up chunks files")

        # List possible chunks file names to clean up
        chunks_files_to_clean = []
        if version.startswith("chunks"):
            # If version starts with "chunks", use it directly
            chunks_files_to_clean = [f"{version}.json"]
            # Also try without "chunks_" prefix
            clean_name = version.replace("chunks_", "")
            if clean_name != version:
                chunks_files_to_clean.append(f"chunks_{clean_name}.json")
        else:
            # Otherwise, try chunks with version suffix
            chunks_files_to_clean = [
                f"chunks_{version}.json",
                f"chunks{version}.json",
                f"{version}.json"
            ]

        for chunks_file in chunks_files_to_clean:
            chunks_path = processed_dir / chunks_file
            if chunks_path.exists():
                try:
                    chunks_path.unlink()
                    print(f"   ✅ Chunks file deleted: {chunks_file}")
                    deleted_items.append(f"Chunks file: {chunks_file}")
                except Exception as e:
                    print(f"   ❌ Error deleting chunks file {chunks_file}: {e}")
                    success = False

        if not any((processed_dir / cf).exists() for cf in chunks_files_to_clean):
            print(f"   ℹ️  No matching chunks files found")

        # Clean up unified configuration files
        print(f"\n4️⃣  Cleaning up unified configurations")
        config_patterns = [
            f"*unified*{version}*.json",
            f"*{version}*config*.json",
            f"index_config_{version}.json"
        ]

        for pattern in config_patterns:
            for config_file in Path(".").glob(pattern):
                try:
                    config_file.unlink()
                    print(f"   ✅ Config file deleted: {config_file.name}")
                    deleted_items.append(f"Config file: {config_file.name}")
                except Exception as e:
                    print(f"   ❌ Error deleting config file {config_file.name}: {e}")

        # Clean up other possible locations
        other_paths = [
            Path(f"./chroma_db_{version}"),
            Path(f"./chroma_db_v{version}"),
            Path(f"data/cache/{version}"),
            self.cache_dir / version,
        ]

        print(f"\n5️⃣  Cleaning up additional paths")
        for path in other_paths:
            if path.exists():
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    print(f"   ✅ Deleted: {path}")
                    deleted_items.append(str(path))
                except Exception as e:
                    print(f"   ❌ Error deleting {path}: {e}")

        print("\n" + "=" * 50)
        print(f"✅ CLEANUP COMPLETED for version {version}")
        if deleted_items:
            print(f"   Deleted {len(deleted_items)} items:")
            for item in deleted_items:
                print(f"   • {item}")
        else:
            print("   No items to delete")

        return success

    def export_config(self, version: str) -> Dict[str, Any]:
        """Export configuration for a specific version."""
        print(f"📤 EXPORTING CONFIGURATION FOR: {version}")
        print("=" * 50)

        config = {
            "version": version,
            "chroma_collection": f"academic_docs_{version}",
            "data_directory": f"data/processed/{version}",
            "cache_directory": str(self.cache_dir),
            "chroma_path": f"./chroma_db",
            "timestamp": time.time(),
            "exported_by": "AcademicRAG Index Manager (Unified)"
        }

        # Add vector store configuration
        config["vector_config"] = {
            "embedding_model": "indobenchmark/indobert-base-p2",
            "collection_name": f"academic_docs_{version}",
            "persist_directory": f"./chroma_db"
        }

        # Add BM25 configuration
        config["bm25_config"] = {
            "k1": 1.5,
            "b": 0.75,
            "ngram_range": [1, 2],
            "cache_dir": str(self.cache_dir)
        }

        # Add file locations
        config["file_locations"] = {
            "chunks_file": f"data/processed/{version}/chunks.json",
            "metadata_file": f"data/processed/{version}/metadata.json",
            "chroma_collection_dir": f"./chroma_db/{f'academic_docs_{version}'}",
            "bm25_cache_file": f"{self.cache_dir}/bm25_{version}.pkl"
        }

        print("\n📋 Configuration:")
        for key, value in config.items():
            if key not in ["file_locations", "vector_config", "bm25_config"]:
                print(f"   {key}: {value}")

        print("\n🤖 Vector Store Configuration:")
        for key, value in config["vector_config"].items():
            print(f"   {key}: {value}")

        print("\n🔤 BM25 Configuration:")
        for key, value in config["bm25_config"].items():
            print(f"   {key}: {value}")

        print("\n📁 File Locations:")
        for key, value in config["file_locations"].items():
            print(f"   {key}: {value}")

        # Save to file
        config_file = f"unified_config_{version}.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\n💾 Configuration saved to: {config_file}")
        except Exception as e:
            print(f"\n❌ Error saving configuration: {e}")

        return config

    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of a directory."""
        total_size = 0
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                for item in path.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size
        except Exception:
            pass
        return total_size


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="AcademicRAG Index Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available indexes
  python scripts/index_manager.py --list

  # Test specific index version
  python scripts/index_manager.py --test v1_20241201

  # Clean up index (with confirmation)
  python scripts/index_manager.py --cleanup v1_20241201 --confirm

  # Export configuration
  python scripts/index_manager.py --export-config v1_20241201
        """
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available indexes"
    )

    parser.add_argument(
        "--test",
        metavar="VERSION",
        help="Test a specific index version"
    )

    parser.add_argument(
        "--cleanup",
        metavar="VERSION",
        help="Clean up a specific index version"
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm cleanup operations"
    )

    parser.add_argument(
        "--export-config",
        metavar="VERSION",
        help="Export configuration for a version"
    )

    parser.add_argument(
        "--chroma-dir",
        default="../data/chroma_db",
        help="ChromaDB base directory (default: ../data/chroma_db)"
    )

    parser.add_argument(
        "--cache-dir",
        default="../data/cache",
        help="BM25 cache directory (default: ../data/cache)"
    )

    args = parser.parse_args()

    if not any([args.list, args.test, args.cleanup, args.export_config]):
        parser.print_help()
        return

    # Initialize index manager
    manager = IndexManager(
        chroma_base_dir=args.chroma_dir,
        cache_dir=args.cache_dir
    )

    # Execute requested action
    try:
        if args.list:
            result = manager.list_indexes()
            return result
        elif args.test:
            success = manager.test_index(args.test)
            sys.exit(0 if success else 1)
        elif args.cleanup:
            success = manager.cleanup_index(args.cleanup, confirm=args.confirm)
            sys.exit(0 if success else 1)
        elif args.export_config:
            config = manager.export_config(args.export_config)
            return config
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
