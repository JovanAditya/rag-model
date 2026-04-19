"""
Response models and data structures for the Academic RAG Model.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SourceDocument:
    """Represents a source document used in RAG response."""
    id: str
    text: str
    title: Optional[str] = None
    score: float = 0.0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGResult:
    """Detailed result from RAG pipeline with metadata."""
    answer: str
    question: str
    sources: List[SourceDocument]
    confidence: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0

    # Pipeline information
    pipeline_type: str = "advanced"
    llm_used: str = "gemini"
    embedding_model: str = "indobenchmark/indobert-base-p2"

    # Additional metadata
    documents_retrieved: int = 0
    documents_reranked: int = 0
    context_length: int = 0

    # Error information
    error: Optional[str] = None
    fallback_used: bool = False

    def __post_init__(self):
        if not self.sources:
            self.sources = []


@dataclass
class RAGResponse:
    """Simplified response object for easy integration."""
    result: RAGResult

    @property
    def answer(self) -> str:
        """Get the generated answer."""
        return self.result.answer

    @property
    def sources(self) -> List[SourceDocument]:
        """Get source documents."""
        return self.result.sources

    @property
    def confidence(self) -> float:
        """Get confidence score."""
        return self.result.confidence

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get full metadata."""
        return {
            "pipeline_type": self.result.pipeline_type,
            "retrieval_time": self.result.retrieval_time,
            "generation_time": self.result.generation_time,
            "total_time": self.result.total_time,
            "llm_used": self.result.llm_used,
            "embedding_model": self.result.embedding_model,
            "documents_retrieved": self.result.documents_retrieved,
            "context_length": self.result.context_length,
            "error": self.result.error,
            "fallback_used": self.result.fallback_used,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": [
                {
                    "id": src.id,
                    "text": src.text,
                    "title": src.title,
                    "score": src.score,
                    "relevance_score": src.relevance_score,
                    "metadata": src.metadata
                }
                for src in self.sources
            ],
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat()
        }


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for RAG pipeline performance."""
    mrr: float = 0.0  # Mean Reciprocal Rank
    precision_at_k: Dict[int, float] = None  # P@K for different K values
    recall_at_k: Dict[int, float] = None    # R@K for different K values
    map_score: float = 0.0  # Mean Average Precision
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain

    # RAGAS metrics (if available)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0

    # Performance metrics
    avg_response_time: float = 0.0
    success_rate: float = 0.0

    def __post_init__(self):
        if self.precision_at_k is None:
            self.precision_at_k = {}
        if self.recall_at_k is None:
            self.recall_at_k = {}


@dataclass
class BatchResult:
    """Results from batch query processing."""
    results: List[RAGResponse]
    total_time: float
    avg_time_per_query: float
    successful_queries: int
    failed_queries: int

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = len(self.results)
        return self.successful_queries / total if total > 0 else 0.0

    def get_successful_results(self) -> List[RAGResponse]:
        """Get only successful results."""
        return [r for r in self.results if r.result.error is None]

    def get_failed_results(self) -> List[RAGResponse]:
        """Get only failed results."""
        return [r for r in self.results if r.result.error is not None]