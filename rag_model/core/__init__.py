"""
Core components of the Academic RAG Model.

This module contains the main pipeline, configuration management,
and response models for the RAG system.
"""

from .pipeline import AcademicRAG
from .config import RAGConfig, PipelineType, LLMType
from .exceptions import RAGError, ConfigurationError, ModelNotFoundError, IndexNotFoundError
from .models import RAGResponse, RAGResult, SourceDocument

__all__ = [
    "AcademicRAG",
    "RAGConfig",
    "PipelineType",
    "LLMType",
    "RAGResponse",
    "RAGResult",
    "SourceDocument",
    "RAGError",
    "ConfigurationError",
    "ModelNotFoundError",
    "IndexNotFoundError",
]