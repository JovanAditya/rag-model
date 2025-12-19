"""
Academic RAG Model - Advanced Retrieval-Augmented Generation for Indonesian Academic Information

This package provides a complete RAG pipeline optimized for Indonesian academic documents,
with both baseline (vector-only) and advanced (hybrid search + reranking) approaches.

Main Components:
- AcademicRAG: Main pipeline interface
- RAGConfig: Configuration management
- Evaluation framework for model testing
- Integration examples and utilities

Example Usage:
    from model.rag_model import AcademicRAG, RAGConfig

    # Basic usage
    rag = AcademicRAG()
    result = rag.query("Bagaimana cara pendaftaran mata kuliah?")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")

    # With custom configuration
    config = RAGConfig(
        pipeline_type="baseline",
        llm_type="gemini",
        max_results=10
    )
    rag = AcademicRAG(config=config)

    # Batch processing
    results = rag.batch_query(["Question 1", "Question 2"])
"""

__version__ = "0.2.0"
__author__ = "Jovan Aditya"
__email__ = "jovan@joppanaditya.my.id"

# Core imports
from .core.pipeline import AcademicRAG
from .core.config import RAGConfig, PipelineType, LLMType
from .core.exceptions import (
    RAGError,
    ConfigurationError,
    ModelNotFoundError,
    IndexNotFoundError,
)

# Response models
from .core.models import RAGResponse, RAGResult, SourceDocument

# Evaluation
try:
    from .utils.evaluation import RAGEvaluator, EvaluationMetrics

    _EVALUATION_AVAILABLE = True
except ImportError:
    _EVALUATION_AVAILABLE = False

# Export main components
__all__ = [
    # Core classes
    "AcademicRAG",
    "RAGConfig",
    "PipelineType",
    "LLMType",
    # Response models
    "RAGResponse",
    "RAGResult",
    "SourceDocument",
    # Exceptions
    "RAGError",
    "ConfigurationError",
    "ModelNotFoundError",
    "IndexNotFoundError",
    # Evaluation (if available)
]

if _EVALUATION_AVAILABLE:
    __all__.extend(["RAGEvaluator", "EvaluationMetrics"])


# Package metadata
def get_version():
    """Get package version"""
    return __version__


def get_info():
    """Get package information"""
    return {
        "name": "academic-rag-research",
        "version": __version__,
        "author": __author__,
        "description": "Research prototype for Indonesian academic information retrieval",
        "evaluation_available": _EVALUATION_AVAILABLE,
    }
