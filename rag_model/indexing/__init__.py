"""
Indexing components for the Academic RAG Model.

This module provides unified indexing with ChromaDB vector store
and Python BM25 for efficient hybrid search.
"""

try:
    from .vector_store import VectorStore
    from .unified_index_manager import UnifiedIndexManager
    from .bm25_index import BM25Index
except ImportError:
    # Handle missing dependencies gracefully
    VectorStore = None
    UnifiedIndexManager = None
    BM25Index = None

__all__ = [
    "VectorStore",
    "UnifiedIndexManager",
    "BM25Index"
]