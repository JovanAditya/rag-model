"""Unified Index Manager for AcademicRAG.

Replaces the problematic DualIndexManager with a simplified, reliable
ChromaDB + Python BM25 unified architecture. Ensures perfect consistency
by building both indexes from the same source data in a single process.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from rag_model.indexing.vector_store import VectorStore
from rag_model.indexing.bm25_index import BM25Index
from rag_model.export.index_exporter import IndexExporter
from rag_model.core.exceptions import IndexingError, ConfigurationError

logger = logging.getLogger(__name__)


class UnifiedIndexManager:
    """
    Unified index manager that handles both ChromaDB vector store and Python BM25 index.

    This manager ensures perfect consistency between both indexes by building them
    simultaneously from the same source documents in a single process.

    Key benefits:
    - Single-pass indexing for both stores
    - Perfect consistency guaranteed
    - No sync issues or false positives
    - Simplified deployment and maintenance
    - Memory efficient processing
    """

    def __init__(
        self,
        vector_config: Dict[str, Any],
        bm25_config: Dict[str, Any],
        cache_dir: str = "./cache"
    ):
        """
        Initialize unified index manager.

        Args:
            vector_config: Configuration for ChromaDB vector store
            bm25_config: Configuration for BM25 index
            cache_dir: Directory for caching intermediate results
        """
        self.vector_config = vector_config
        self.bm25_config = bm25_config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize indexes (but don't load data yet)
        self.vector_store = None
        self.bm25_index = None
        self.indexed_documents = []

        logger.info("UnifiedIndexManager initialized")
        logger.info(f"Vector config: {vector_config}")
        logger.info(f"BM25 config: {bm25_config}")

    def initialize_indexes(self) -> None:
        """Initialize both vector store and BM25 index."""
        try:
            # Initialize ChromaDB vector store
            from rag_model.core.config import EmbeddingConfig, IndexConfig

            embedding_config = EmbeddingConfig(
                model_name=self.vector_config["embedding_model"]
            )

            index_config = IndexConfig(
                chroma_dir=self.vector_config["persist_directory"],
                chroma_collection=self.vector_config["collection_name"]
            )

            self.vector_store = VectorStore(
                collection_name=self.vector_config["collection_name"],
                embedding_config=embedding_config,
                index_config=index_config
            )

            # Initialize BM25 index
            self.bm25_index = BM25Index(
                k1=self.bm25_config.get("k1", 1.5),
                b=self.bm25_config.get("b", 0.75),
                ngram_range=self.bm25_config.get("ngram_range", (1, 2)),
                cache_dir=str(self.cache_dir)
            )

            # Try to load existing BM25 cache
            cache_name = f"bm25_{self.vector_config.get('collection_name', 'default')}"
            bm25_loaded = self.bm25_index.load_cache(cache_name)
            
            if bm25_loaded:
                bm25_stats = self.bm25_index.get_stats()
                doc_count = bm25_stats.get("documents_count", 0)
                if doc_count > 0:
                    logger.info(f"✅ BM25 cache loaded successfully ({doc_count} documents)")
                else:
                    logger.warning("⚠️ BM25 cache loaded but empty, will need reindexing")
                    bm25_loaded = False
            
            if not bm25_loaded:
                logger.info("📝 BM25 cache not found or empty, will be created during indexing")

            logger.info("Both indexes initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize indexes: {e}")
            raise IndexingError(f"Index initialization failed: {e}")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents into both ChromaDB and BM25 simultaneously.

        Args:
            documents: List of document chunks with text and metadata
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return

        start_time = time.time()
        logger.info(f"Starting unified indexing for {len(documents)} documents")

        # Initialize indexes if not already done
        if self.vector_store is None or self.bm25_index is None:
            self.initialize_indexes()

        # Validate and prepare documents
        processed_docs = self._prepare_documents(documents)
        if not processed_docs:
            logger.warning("No valid documents after preprocessing")
            return

        # Index into both stores
        try:
            # Index into ChromaDB vector store
            logger.info("Indexing into ChromaDB vector store...")
            vector_added = self.vector_store.add_documents(processed_docs)
            
            # Check if vector indexing actually succeeded
            if vector_added is None or vector_added == 0:
                logger.error(f"❌ ChromaDB indexing FAILED: 0 of {len(processed_docs)} documents added")
                raise IndexingError("Vector store indexing failed - 0 documents added")
            
            logger.info(f"✅ ChromaDB indexing completed: {vector_added} documents")

            # Index into BM25
            logger.info("Indexing into BM25...")
            self.bm25_index.index_documents(processed_docs)

            # Save BM25 index to cache
            logger.info("Saving BM25 index to cache...")
            cache_name = f"bm25_{self.vector_config.get('collection_name', 'default')}"
            self.bm25_index.save_cache(cache_name)
            logger.info("✅ BM25 indexing completed")

            # Store document references for consistency
            self.indexed_documents = processed_docs.copy()

            # Log statistics
            total_time = time.time() - start_time
            self._log_indexing_stats(total_time)

        except Exception as e:
            logger.error(f"Unified indexing failed: {e}")
            raise IndexingError(f"Failed to index documents: {e}")

    def _prepare_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare and validate documents for indexing.

        Args:
            documents: Raw document list

        Returns:
            List of validated and prepared documents
        """
        processed_docs = []

        for i, doc in enumerate(documents):
            try:
                # Extract and validate text content
                text = doc.get('text', '').strip()
                if not text or len(text) < 10:  # Skip very short texts
                    logger.debug(f"Skipping document {i}: text too short or empty")
                    continue

                # Ensure metadata exists
                metadata = doc.get('metadata', {})
                if not metadata:
                    metadata = {'source': f'document_{i}'}

                # Create prepared document
                prepared_doc = {
                    'text': text,
                    'metadata': {
                        **metadata,
                        'doc_id': metadata.get('doc_id', f'doc_{i}'),
                        'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                        'unified_index_timestamp': time.time()
                    }
                }

                processed_docs.append(prepared_doc)

            except Exception as e:
                logger.warning(f"Error preparing document {i}: {e}")
                continue

        logger.info(f"Prepared {len(processed_docs)} valid documents for indexing")
        return processed_docs

    def _log_indexing_stats(self, total_time: float) -> None:
        """Log comprehensive indexing statistics."""
        try:
            vector_stats = self.vector_store.get_stats()
            bm25_stats = self.bm25_index.get_stats()

            logger.info("=== UNIFIED INDEXING STATISTICS ===")
            logger.info(f"Total documents indexed: {len(self.indexed_documents)}")
            logger.info(f"Total indexing time: {total_time:.2f}s")
            logger.info(f"Average time per document: {(total_time / len(self.indexed_documents) * 1000):.1f}ms")

            logger.info("Vector Store (ChromaDB):")
            for key, value in vector_stats.items():
                logger.info(f"  {key}: {value}")

            logger.info("BM25 Index:")
            for key, value in bm25_stats.items():
                logger.info(f"  {key}: {value}")

        except Exception as e:
            logger.warning(f"Could not generate complete statistics: {e}")

    def search_unified(
        self,
        query: str,
        k: int = 10,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        strategy: str = "rrf"
    ) -> Dict[str, Any]:
        """
        Perform unified search across both vector and BM25 indexes.

        Args:
            query: Search query
            k: Number of results to return
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 search results
            strategy: Fusion strategy ("rrf", "weighted", "max", "simple")

        Returns:
            Dictionary with unified search results and metadata
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return {"results": [], "query": query, "stats": {}}

        # Auto-initialize if not done
        if self.vector_store is None or self.bm25_index is None:
            logger.info("Auto-initializing indexes for search...")
            try:
                self.initialize_indexes()
            except Exception as e:
                logger.error(f"Failed to auto-initialize indexes: {e}")
                raise IndexingError("Indexes not initialized. Call index_documents() first.")

        start_time = time.time()
        logger.info(f"Unified search: '{query}' (k={k}, strategy={strategy})")

        try:
            # Vector search
            vector_start = time.time()
            vector_results = self.vector_store.similarity_search(query, k=k)
            vector_time = time.time() - vector_start

            # BM25 search - with graceful fallback if not ready
            bm25_start = time.time()
            bm25_results = []
            bm25_available = True
            try:
                bm25_stats = self.bm25_index.get_stats()
                if bm25_stats.get("documents_count", 0) > 0:
                    bm25_results = self.bm25_index.search(query, k=k)
                else:
                    logger.warning("BM25 index empty, using vector-only search")
                    bm25_available = False
            except Exception as bm25_err:
                logger.warning(f"BM25 search failed: {bm25_err}, using vector-only")
                bm25_available = False
            bm25_time = time.time() - bm25_start

            # If BM25 not available, return vector results directly
            if not bm25_available:
                return {
                    "results": vector_results[:k],
                    "query": query,
                    "stats": {
                        "total_time": time.time() - start_time,
                        "vector_search_time": vector_time,
                        "bm25_search_time": 0,
                        "fusion_time": 0,
                        "vector_results_count": len(vector_results),
                        "bm25_results_count": 0,
                        "final_results_count": len(vector_results),
                        "fusion_strategy": "vector_only",
                        "note": "BM25 not available, vector-only results"
                    }
                }

            # Fuse results
            fusion_start = time.time()
            unified_results = self._fuse_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                strategy=strategy
            )
            fusion_time = time.time() - fusion_start

            # Prepare response
            total_time = time.time() - start_time

            return {
                "results": unified_results[:k],  # Limit to k results
                "query": query,
                "stats": {
                    "total_time": total_time,
                    "vector_search_time": vector_time,
                    "bm25_search_time": bm25_time,
                    "fusion_time": fusion_time,
                    "vector_results_count": len(vector_results),
                    "bm25_results_count": len(bm25_results),
                    "final_results_count": len(unified_results),
                    "fusion_strategy": strategy,
                    "vector_weight": vector_weight,
                    "bm25_weight": bm25_weight
                }
            }

        except Exception as e:
            logger.error(f"Unified search failed: {e}")
            raise IndexingError(f"Search failed: {e}")

    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float,
        bm25_weight: float,
        strategy: str
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from vector and BM25 search using specified strategy.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            vector_weight: Weight for vector results
            bm25_weight: Weight for BM25 results
            strategy: Fusion strategy

        Returns:
            Fused and ranked results
        """
        if strategy == "rrf":
            return self._reciprocal_rank_fusion(vector_results, bm25_results)
        elif strategy == "weighted":
            return self._weighted_fusion(vector_results, bm25_results, vector_weight, bm25_weight)
        elif strategy == "max":
            return self._max_fusion(vector_results, bm25_results)
        else:  # simple
            return self._simple_fusion(vector_results, bm25_results)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) implementation.

        Args:
            vector_results: Vector search results
            bm25_results: BM25 search results
            k: RRF constant (typically 60)

        Returns:
            RRF-fused results
        """
        # Score dictionaries by document identifier
        doc_scores = {}
        doc_data = {}

        # Process vector results
        for rank, doc in enumerate(vector_results):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = doc

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = doc

        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc['rrf_score'] = rrf_score
            doc['fusion_source'] = 'rrf'
            fused_results.append(doc)

        return fused_results

    def _weighted_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Dict[str, Any]]:
        """Weighted score fusion."""
        # Normalize weights
        total_weight = vector_weight + bm25_weight
        vector_weight = vector_weight / total_weight
        bm25_weight = bm25_weight / total_weight

        # Score dictionaries
        doc_scores = {}
        doc_data = {}

        # Process vector results
        for doc in vector_results:
            doc_id = self._get_doc_id(doc)
            vector_score = doc.get('similarity_score', 0.0)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + vector_weight * vector_score
            doc_data[doc_id] = doc

        # Process BM25 results
        for doc in bm25_results:
            doc_id = self._get_doc_id(doc)
            bm25_score = doc.get('bm25_score', 0.0)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + bm25_weight * bm25_score
            doc_data[doc_id] = doc

        # Sort by weighted score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        fused_results = []
        for doc_id, weighted_score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc['weighted_score'] = weighted_score
            doc['fusion_source'] = 'weighted'
            fused_results.append(doc)

        return fused_results

    def _max_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Max score fusion (keep highest score for each document)."""
        doc_scores = {}
        doc_data = {}

        # Process vector results
        for doc in vector_results:
            doc_id = self._get_doc_id(doc)
            vector_score = doc.get('similarity_score', 0.0)
            if doc_id not in doc_scores or vector_score > doc_scores[doc_id]:
                doc_scores[doc_id] = vector_score
                doc_data[doc_id] = doc

        # Process BM25 results
        for doc in bm25_results:
            doc_id = self._get_doc_id(doc)
            bm25_score = doc.get('bm25_score', 0.0)
            if doc_id not in doc_scores or bm25_score > doc_scores[doc_id]:
                doc_scores[doc_id] = bm25_score
                doc_data[doc_id] = doc.copy()
                doc_data[doc_id]['max_source'] = 'bm25'

        # Sort by max score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        fused_results = []
        for doc_id, max_score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc['max_score'] = max_score
            doc['fusion_source'] = 'max'
            fused_results.append(doc)

        return fused_results

    def _simple_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simple fusion with deduplication."""
        seen_docs = set()
        fused_results = []

        # Add vector results first
        for doc in vector_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                doc['fusion_source'] = 'vector'
                fused_results.append(doc)

        # Add BM25 results that weren't already added
        for doc in bm25_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                doc['fusion_source'] = 'bm25'
                fused_results.append(doc)

        return fused_results

    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """Get unique document identifier for fusion."""
        # Try to get unique ID from metadata
        metadata = doc.get('metadata', {})

        # Priority order for document ID
        for id_field in ['chunk_id', 'doc_id', 'id']:
            if id_field in metadata and metadata[id_field]:
                return str(metadata[id_field])

        # Fallback to text-based ID (hash of first 100 chars)
        text = doc.get('text', '')
        if text:
            return str(hash(text[:100]))

        # Last resort - use document object id
        return str(id(doc))

    def get_unified_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for both indexes."""
        try:
            stats = {
                "unified_manager": {
                    "indexed_documents": len(self.indexed_documents),
                    "status": "ready" if (self.vector_store and self.bm25_index) else "not_initialized"
                },
                "vector_store": {},
                "bm25_index": {}
            }

            if self.vector_store:
                stats["vector_store"] = self.vector_store.get_collection_stats()

            if self.bm25_index:
                stats["bm25_index"] = self.bm25_index.get_stats()

            return stats

        except Exception as e:
            logger.error(f"Error getting unified stats: {e}")
            return {"error": str(e)}



    def clear_indexes(self) -> None:
        """Clear all indexes (for rebuilding)."""
        try:
            if self.vector_store:
                self.vector_store.clear_collection()
                logger.info("Vector store cleared")

            if self.bm25_index:
                # BM25 index is recreated on next indexing
                self.bm25_index = None
                logger.info("BM25 index cleared")

            self.indexed_documents = []
            logger.info("All indexes cleared successfully")

        except Exception as e:
            logger.error(f"Error clearing indexes: {e}")
            raise IndexingError(f"Failed to clear indexes: {e}")

    # Export/Import Functionality

    def initialize_exporter(self, output_dir: str = "./exports") -> IndexExporter:
        """
        Initialize IndexExporter for system export operations.

        Args:
            output_dir: Directory for export files

        Returns:
            Configured IndexExporter instance
        """
        return IndexExporter(output_dir)

    def export_complete_system(
        self,
        components: Optional[List[str]] = None,
        version_suffix: Optional[str] = None,
        output_dir: str = "./exports"
    ) -> Dict[str, Any]:
        """
        Export complete system including indexes, documents, and configuration.

        Args:
            components: List of components to export ["chromadb", "bm25", "documents", "config"]
            version_suffix: Version suffix for export package
            output_dir: Output directory for export files

        Returns:
            Dictionary with export results and metadata
        """
        try:
            # Initialize exporter
            exporter = self.initialize_exporter(output_dir)

            # Set default components if not specified
            if components is None:
                components = ["chromadb", "bm25", "documents", "config"]

            # Export complete system
            export_package = exporter.export_complete_system(components, version_suffix)

            logger.info(f"✅ Complete system exported successfully")
            logger.info(f"Export package: {export_package.archive_path}")
            logger.info(f"Components: {', '.join(components)}")

            return {
                "status": "success",
                "export_package": export_package,
                "components_exported": components,
                "archive_path": str(export_package.archive_path),
                "metadata": export_package.metadata
            }

        except Exception as e:
            logger.error(f"System export failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "components_attempted": components
            }

    def export_chromadb_only(self, output_dir: str = "./exports") -> Dict[str, Any]:
        """
        Export only ChromaDB collection.

        Args:
            output_dir: Output directory for export files

        Returns:
            Dictionary with export results
        """
        try:
            exporter = self.initialize_exporter(output_dir)
            export_package = exporter.export_complete_system(["chromadb"])

            return {
                "status": "success",
                "export_package": export_package,
                "archive_path": str(export_package.archive_path),
                "component": "chromadb"
            }

        except Exception as e:
            logger.error(f"ChromaDB export failed: {e}")
            return {"status": "error", "error": str(e)}

    def export_bm25_only(self, output_dir: str = "./exports") -> Dict[str, Any]:
        """
        Export only BM25 index.

        Args:
            output_dir: Output directory for export files

        Returns:
            Dictionary with export results
        """
        try:
            exporter = self.initialize_exporter(output_dir)
            export_package = exporter.export_complete_system(["bm25"])

            return {
                "status": "success",
                "export_package": export_package,
                "archive_path": str(export_package.archive_path),
                "component": "bm25"
            }

        except Exception as e:
            logger.error(f"BM25 export failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_export_info(self) -> Dict[str, Any]:
        """
        Get information about current system that would be exported.

        Returns:
            Dictionary with system information for export
        """
        try:
            stats = self.get_unified_stats()

            export_info = {
                "system_ready": self.vector_store is not None and self.bm25_index is not None,
                "components": {
                    "chromadb": {
                        "available": self.vector_store is not None,
                        "stats": stats.get("vector_store", {})
                    },
                    "bm25": {
                        "available": self.bm25_index is not None,
                        "stats": stats.get("bm25_index", {})
                    },
                    "documents": {
                        "available": len(self.indexed_documents) > 0,
                        "count": len(self.indexed_documents)
                    }
                },
                "configuration": {
                    "vector_config": self.vector_config,
                    "bm25_config": self.bm25_config,
                    "cache_dir": str(self.cache_dir)
                }
            }

            return export_info

        except Exception as e:
            logger.error(f"Error getting export info: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on unified index manager components.

        Returns:
            Dictionary with health status and component details
        """
        try:
            health_status = {
                "status": "healthy",
                "components": {
                    "vector_store": {
                        "initialized": self.vector_store is not None,
                        "available": False,
                        "document_count": 0
                    },
                    "bm25_index": {
                        "initialized": self.bm25_index is not None,
                        "available": False,
                        "document_count": 0
                    }
                },
                "issues": []
            }

            # Check vector store
            if self.vector_store is not None:
                try:
                    # Try to get collection info
                    if hasattr(self.vector_store, 'collection') and self.vector_store.collection:
                        health_status["components"]["vector_store"]["available"] = True
                        if hasattr(self.vector_store.collection, 'count'):
                            health_status["components"]["vector_store"]["document_count"] = self.vector_store.collection.count()
                    else:
                        health_status["components"]["vector_store"]["available"] = False
                        health_status["issues"].append("Vector store collection not available")
                except Exception as e:
                    health_status["components"]["vector_store"]["available"] = False
                    health_status["issues"].append(f"Vector store error: {str(e)}")
            else:
                health_status["issues"].append("Vector store not initialized")

            # Check BM25 index
            if self.bm25_index is not None:
                try:
                    # Check if BM25 index has documents
                    bm25_stats = self.bm25_index.get_stats()
                    if bm25_stats.get("documents_count", 0) > 0:
                        health_status["components"]["bm25_index"]["available"] = True
                        health_status["components"]["bm25_index"]["document_count"] = bm25_stats["documents_count"]
                    else:
                        health_status["components"]["bm25_index"]["available"] = False
                        health_status["issues"].append("BM25 index document count not available")
                except Exception as e:
                    health_status["components"]["bm25_index"]["available"] = False
                    health_status["issues"].append(f"BM25 index error: {str(e)}")
            else:
                health_status["issues"].append("BM25 index not initialized")

            # Determine overall status
            if health_status["issues"]:
                health_status["status"] = "unhealthy"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "components": {},
                "issues": [f"Health check error: {str(e)}"]
            }
