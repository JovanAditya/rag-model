"""
Custom exceptions for the Academic RAG Model.
"""


class RAGError(Exception):
    """Base exception for all RAG-related errors."""
    pass


class ConfigurationError(RAGError):
    """Raised when there's an issue with RAG configuration."""
    pass


class ModelNotFoundError(RAGError):
    """Raised when a required model (embedding, LLM, etc.) is not found."""
    pass


class IndexNotFoundError(RAGError):
    """Raised when vector store or search index is not found or empty."""
    pass


class IndexingError(RAGError):
    """Raised when document indexing fails."""
    pass


class DocumentProcessingError(RAGError):
    """Raised when document processing fails."""
    pass


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""
    pass


class GenerationError(RAGError):
    """Raised when LLM text generation fails."""
    pass


class EvaluationError(RAGError):
    """Raised when model evaluation fails."""
    pass


class ServiceUnavailableError(RAGError):
    """Raised when external services (ChromaDB, Ollama) are unavailable."""
    pass