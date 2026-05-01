"""
Main Academic RAG Pipeline implementation (FIXED VERSION).

This module provides the core AcademicRAG class that serves as the main
interface for the RAG system, supporting both baseline and advanced
pipelines with flexible configuration.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union, Awaitable
from pathlib import Path

from .config import RAGConfig, PipelineType
from .models import RAGResponse, RAGResult, SourceDocument, BatchResult
from .exceptions import (
    RAGError,
    ConfigurationError,
    ModelNotFoundError,
    IndexNotFoundError,
    ServiceUnavailableError
)

# Import indexing components
from ..indexing import VectorStore
from ..indexing.unified_index_manager import UnifiedIndexManager

# Import retrieval components
from ..models.baseline_retriever import BaselineRetriever
from ..models.reranker import CrossEncoderReranker

# Import generation components
from ..models.llm_generator import LLMGenerator
from ..models.context_builder import ContextBuilder


class AcademicRAG:
    """
    Main Academic RAG Pipeline for Indonesian academic information retrieval.

    This class provides a unified interface for both baseline (vector search only)
    and advanced (hybrid search + reranking) RAG pipelines.

    Args:
        config: RAGConfig instance or path to config file
        config_path: Path to configuration file (JSON/YAML)
        validate_config: Whether to validate configuration on initialization
        research_mode: Whether to enable research mode with detailed metrics
        response_format: Response format ("simple", "full", "api")

    Example:
        >>> from rag_model import AcademicRAG
        >>> rag = AcademicRAG()
        >>> result = rag.query("Bagaimana cara pendaftaran?")
        >>> print(result.answer)

        >>> # With custom configuration
        >>> config = RAGConfig(pipeline_type="baseline", max_results=10)
        >>> rag = AcademicRAG(config=config)
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        validate_config: bool = True,
        research_mode: bool = False,
        response_format: str = "simple"
    ):
        # Initialize configuration
        if config is None and config_path is None:
            self.config = RAGConfig()
        elif config is not None:
            self.config = config
        elif config_path is not None:
            self.config = RAGConfig.from_file(config_path)
        else:
            raise ConfigurationError("Either config or config_path must be provided")

        # Validate configuration if requested
        if validate_config:
            self.config.validate()

        # Initialize advanced components
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(self.config, 'log_level', logging.INFO))

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # Initialize components
        self._vector_store = None
        self._unified_index_manager = None
        self._baseline_retriever = None
        self._hybrid_retriever = None
        self._reranker = None
        self._llm_generator = None
        self._context_builder = None

        # Performance tracking
        self._query_count = 0
        self._total_time = 0.0
        self._error_count = 0

        # Mode configuration for unified architecture
        self.research_mode = research_mode
        self.response_format = response_format
        self._validate_response_format()

        # Initialize lazy loading
        self._initialized = False

    def _validate_response_format(self) -> None:
        """Validate response format parameter."""
        valid_formats = ["simple", "full", "api"]
        if self.response_format not in valid_formats:
            raise ValueError(f"Invalid response_format: {self.response_format}. "
                           f"Valid options: {valid_formats}")

    def _initialize_components(self):
        """Initialize RAG components on first use."""
        if self._initialized:
            return

        try:
            self.logger.info("Initializing Academic RAG Pipeline components...")

            # Initialize LLM generator first (might be needed for validation)
            from ..models.llm_generator import LLMConfig
            llm_config = LLMConfig(
                model_type=self.config.llm.model_type,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                endpoint=self.config.llm.endpoint
            )
            self._llm_generator = LLMGenerator(config=llm_config)

            # Initialize context builder
            self._context_builder = ContextBuilder(logger=self.logger)

            # Initialize indexing components based on pipeline type
            if self.config.retrieval.pipeline_type == "baseline":
                self._initialize_baseline_components()
            else:
                self._initialize_advanced_components()

            self._initialized = True
            self.logger.info("Academic RAG Pipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise RAGError(f"Pipeline initialization failed: {e}")

    def _initialize_baseline_components(self):
        """Initialize baseline pipeline components."""
        # Initialize vector store
        self._vector_store = VectorStore(
            collection_name=self.config.index.chroma_collection,
            embedding_config=self.config.embedding,
            index_config=self.config.index,
            logger=self.logger
        )

        # Initialize baseline retriever
        self._baseline_retriever = BaselineRetriever(
            vector_store=self._vector_store,
            logger=self.logger
        )

    def _initialize_advanced_components(self):
        """Initialize advanced pipeline components."""
        # Initialize unified index manager
        vector_config = {
            "collection_name": self.config.index.chroma_collection,
            "persist_directory": self.config.index.chroma_dir,
            "embedding_model": self.config.embedding.model_name
        }

        bm25_config = {
            "k1": self.config.bm25.k1,
            "b": self.config.bm25.b,
            "ngram_range": (self.config.bm25.ngram_range_min, self.config.bm25.ngram_range_max)
        }

        self._unified_index_manager = UnifiedIndexManager(
            vector_config=vector_config,
            bm25_config=bm25_config,
            cache_dir=self.config.index.cache_dir
        )

        # Health check for advanced components - only fallback on actual errors
        # Empty/degraded status is OK - indexes will be populated during chunking
        health = self._unified_index_manager.health_check()
        if health["status"] == "error":
            self.logger.warning(f"Advanced pipeline initialization error: {health.get('error', 'unknown')}")
            self.logger.warning("Falling back to baseline pipeline")
            self.config.retrieval.pipeline_type = "baseline"
            self._initialize_baseline_components()
            return
        elif health["status"] != "healthy":
            self.logger.info(f"Advanced pipeline status: {health['status']} (indexes may be empty, will populate on chunking)")

        # Unified manager handles hybrid search directly
        # No need for separate HybridSearchRetriever

        # Initialize reranker if enabled
        if self.config.retrieval.use_reranking:
            try:
                # Create reranker config with appropriate device
                from ..core.config import RerankerConfig
                reranker_config = RerankerConfig(
                    device=self.config.embedding.device
                )
                self._reranker = CrossEncoderReranker(
                    config=reranker_config,
                    logger=self.logger
                )
                self.logger.info("Cross-encoder reranker initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize reranker: {e}")
                self.config.retrieval.use_reranking = False

    def query(
        self,
        question: str,
        pipeline_type: Optional[PipelineType] = None,
        max_results: Optional[int] = None,
        include_sources: bool = True,
        include_metrics: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single query through RAG pipeline with smart formatting.

        Args:
            question: User question in Bahasa Indonesia
            pipeline_type: Override default pipeline type ("baseline" or "advanced")
            max_results: Override default max results
            include_sources: Whether to include source information
            include_metrics: Include detailed metrics (auto-determined from research_mode)
            **kwargs: Additional parameters for the pipeline

        Returns:
            Formatted response dictionary based on current mode

        Raises:
            RAGError: If query processing fails
            IndexNotFoundError: If no documents are found in indexes
        """
        start_time = time.time()
        self._query_count += 1

        try:
            # Initialize components if not already done
            self._initialize_components()

            # Use provided parameters or defaults
            pipeline_type = pipeline_type or self.config.retrieval.pipeline_type
            max_results = max_results or self.config.retrieval.max_results

            self.logger.debug(f"Processing query: {question[:100]}...")

            # Step 1: Document Retrieval
            retrieval_start = time.time()
            retrieved_docs = self._retrieve_documents(
                question, pipeline_type, max_results, **kwargs
            )
            retrieval_time = time.time() - retrieval_start

            if not retrieved_docs:
                self._error_count += 1
                result = self._create_empty_result(question, "No relevant documents found")
                result.retrieval_time = retrieval_time
                result.total_time = time.time() - start_time
                return self._format_response(result, include_metrics)

            # Step 2: Context Building
            context_start = time.time()
            context_data = self._context_builder.build_context(
                retrieved_docs, question
            )
            context_time = time.time() - context_start

            # Step 3: Answer Generation
            generation_start = time.time()
            answer = self._generate_answer(question, context_data["context"])
            generation_time = time.time() - generation_start

            # Step 4: Create Result
            total_time = time.time() - start_time
            result = RAGResult(
                answer=answer,
                question=question,
                sources=retrieved_docs if include_sources else [],
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                pipeline_type=pipeline_type,
                llm_used=self.config.llm.model_type,
                embedding_model=self.config.embedding.model_name,
                documents_retrieved=len(retrieved_docs),
                context_length=len(context_data["context"])
            )

            self._total_time += total_time
            self.logger.debug(f"Query processed in {total_time:.2f}s")

            return self._format_response(result, include_metrics)

        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Query processing failed: {e}")
            result = self._create_error_result(question, str(e))
            result.total_time = time.time() - start_time
            return self._format_response(result, include_metrics)

    async def query_async(
        self,
        question: str,
        pipeline_type: Optional[PipelineType] = None,
        max_results: Optional[int] = None,
        include_sources: bool = True,
        include_metrics: Optional[bool] = None,
        **kwargs
    ) -> Awaitable[Dict[str, Any]]:
        """
        Async version of query method.

        Args:
            question: User question in Bahasa Indonesia
            pipeline_type: Override default pipeline type
            max_results: Override default max results
            include_sources: Whether to include source information
            include_metrics: Include detailed metrics
            **kwargs: Additional parameters

        Returns:
            Formatted response dictionary
        """
        # Run synchronous query in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.query, question, pipeline_type, max_results, include_sources, include_metrics, **kwargs
        )

    def _retrieve_documents(
        self,
        question: str,
        pipeline_type: PipelineType,
        max_results: int,
        **kwargs
    ) -> List[SourceDocument]:
        """Retrieve documents based on pipeline type."""
        if pipeline_type == "baseline":
            if not self._baseline_retriever:
                raise RAGError("Baseline retriever not initialized")

            docs = self._baseline_retriever.retrieve(
                query=question,
                k=max_results,
                **kwargs
            )
        else:  # advanced - use unified index manager
            if not self._unified_index_manager:
                raise RAGError("Unified index manager not initialized")

            # Use unified search with RRF fusion
            search_results = self._unified_index_manager.search_unified(
                query=question,
                k=max_results,
                vector_weight=self.config.retrieval.vector_weight,
                bm25_weight=self.config.retrieval.bm25_weight,
                strategy="rrf"
            )

            docs = search_results["results"]

            # Apply reranking if enabled
            if self.config.retrieval.use_reranking and self._reranker:
                # Get more documents for reranking
                rerank_results = self._unified_index_manager.search_unified(
                    query=question,
                    k=self.config.retrieval.rerank_k,
                    vector_weight=self.config.retrieval.vector_weight,
                    bm25_weight=self.config.retrieval.bm25_weight,
                    strategy="rrf"
                )

                # Rerank documents (rerank expects list of dicts with 'text' field)
                reranked_docs = self._reranker.rerank(question, rerank_results["results"], top_k=max_results)

                # Use reranked documents
                docs = reranked_docs

        # Convert to SourceDocument objects
        source_docs = []
        for doc in docs:
            # Extract ID from multiple possible locations
            metadata = doc.get('metadata', {}) or {}
            doc_id = (
                doc.get('id') or 
                doc.get('chunk_id') or 
                metadata.get('chunk_id') or 
                metadata.get('global_chunk_id') or 
                ''
            )
            
            source_doc = SourceDocument(
                id=doc_id,
                text=doc.get('text', ''),
                title=doc.get('title'),
                score=doc.get('score', doc.get('cross_encoder_score', 0.0)),
                relevance_score=doc.get('relevance_score', doc.get('score', 0.0)),
                metadata=metadata
            )
            source_docs.append(source_doc)

        return source_docs

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM.
        
        Delegates prompt construction to LLMGenerator which applies
        the SIAssist persona and formatting rules consistently.
        """
        if not self._llm_generator:
            raise RAGError("LLM generator not initialized")

        result = self._llm_generator.generate(prompt=question, context=context)
        return result.get('answer', 'Maaf, saya tidak dapat memberikan jawaban yang memadai.')

    def _create_empty_result(self, question: str, reason: str) -> RAGResult:
        """Create empty result when no documents found."""
        return RAGResult(
            answer="Maaf, tidak ada informasi yang relevan ditemukan dalam basis pengetahuan untuk menjawab pertanyaan Anda.",
            question=question,
            sources=[],
            confidence=0.0,
            pipeline_type=self.config.retrieval.pipeline_type,
            error=reason
        )

    def _create_error_result(self, question: str, error_msg: str) -> RAGResult:
        """Create error result."""
        return RAGResult(
            answer="Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi nanti.",
            question=question,
            sources=[],
            confidence=0.0,
            pipeline_type=self.config.retrieval.pipeline_type,
            error=error_msg,
            fallback_used=True
        )

    def _format_response(
        self,
        result: RAGResult,
        include_metrics: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Format response based on current mode.

        Args:
            result: RAGResult from query processing
            include_metrics: Override automatic metric inclusion

        Returns:
            Formatted response dictionary
        """
        # Determine if metrics should be included
        if include_metrics is None:
            include_metrics = self.research_mode

        # Base response data
        response = {
            "question": result.question,
            "answer": result.answer,
            "sources": self._format_sources(result.sources)
        }

        # Add confidence score
        confidence = self._calculate_confidence(result)
        if self.response_format != "simple":
            response["confidence"] = confidence
        else:
            response["confidence"] = round(confidence, 2)

        # Add mode-specific data
        if self.response_format == "api":
            response.update({
                "model_info": self._get_model_info(),
                "timestamp": time.time(),
                "query_id": f"Q{self._query_count:06d}"
            })

        # Add research/production specific data
        if (self.research_mode or include_metrics) and not result.fallback_used:
            response["metadata"] = {
                "retrieval_time": result.retrieval_time,
                "generation_time": result.generation_time,
                "total_time": result.total_time,
                "pipeline_type": result.pipeline_type,
                "documents_retrieved": result.documents_retrieved,
                "context_length": result.context_length,
                "llm_used": result.llm_used,
                "embedding_model": result.embedding_model
            }

        elif not self.research_mode and self.response_format == "full":
            # Production with enhanced details
            response["metadata"] = {
                "total_time": round(result.total_time, 2),
                "sources_count": len(result.sources)
            }

        return response

    def _format_sources(self, sources) -> List[Dict[str, Any]]:
        """Format sources for response."""
        formatted_sources = []
        for idx, source in enumerate(sources):
            # Handle both dict and object sources
            if isinstance(source, dict):
                # DEBUG: Log first source to see structure
                if idx == 0:
                    self.logger.info(f"[DEBUG] First source keys: {list(source.keys())}")
                    self.logger.info(f"[DEBUG] First source metadata: {source.get('metadata')}")
                
                # Extract ID from metadata if not at top level
                metadata = source.get('metadata', {}) or {}
                source_id = (
                    source.get('id') or 
                    source.get('chunk_id') or 
                    metadata.get('chunk_id') or 
                    metadata.get('global_chunk_id') or 
                    metadata.get('id') or
                    ''
                )
                
                text = source.get('text', source.get('content', ''))
                formatted_source = {
                    "id": source_id,
                    "content": text[:200] + "..." if len(text) > 200 else text,
                    "score": source.get('score', source.get('cross_encoder_score', 0.0)),
                    "metadata": metadata  # Always include metadata for filename extraction
                }
            else:
                # Object sources (SourceDocument)
                metadata = getattr(source, 'metadata', {}) or {}
                formatted_source = {
                    "id": getattr(source, 'id', getattr(source, 'chunk_id', '')),
                    "content": getattr(source, 'text', '')[:200] + "..." if hasattr(source, 'text') and len(getattr(source, 'text', '')) > 200 else getattr(source, 'text', ''),
                    "score": getattr(source, 'score', 0.0),
                    "metadata": metadata  # Always include metadata
                }

            # Add additional metadata for research mode
            if self.research_mode and isinstance(source, dict):
                formatted_source.update({
                    "title": source.get('title', f'Chunk {formatted_source["id"]}'),
                    "document_id": source.get('document_id') or (source.get('metadata', {}) or {}).get('document_id', ''),
                    "position": source.get('position', source.get('chunk_index', 0))
                })

            formatted_sources.append(formatted_source)

        return formatted_sources

    def _calculate_confidence(self, result: RAGResult) -> float:
        """
        Calculate confidence score for the response.

        Args:
            result: RAGResult with sources and metadata

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not result.sources:
            return 0.0

        # Average score of top 3 sources
        top_sources = sorted(result.sources, key=lambda x: getattr(x, 'score', 0.0), reverse=True)[:3]
        avg_score = sum(getattr(source, 'score', 0.0) for source in top_sources) / len(top_sources)

        # Adjust based on retrieval time (faster = more confident)
        time_factor = min(1.0, 2.0 / max(result.total_time, 0.5))

        # Adjust based on number of sources (more sources = more confident)
        source_factor = min(1.0, len(result.sources) / 5.0)

        # Combine factors
        confidence = avg_score * 0.7 + time_factor * 0.15 + source_factor * 0.15

        return round(min(max(confidence, 0.0), 1.0), 3)

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information for API responses."""
        return {
            "pipeline_type": self.config.retrieval.pipeline_type,
            "llm_model": self.config.llm.model_type,
            "embedding_model": self.config.embedding.model_name,
            "reranking": self.config.retrieval.use_reranking,
            "hybrid_search": (
                self.config.retrieval.pipeline_type == "advanced" and
                getattr(self.config.retrieval, 'use_bm25', False) and
                self.config.retrieval.use_vector_search
            )
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        avg_time = self._total_time / self._query_count if self._query_count > 0 else 0
        success_rate = (self._query_count - self._error_count) / self._query_count if self._query_count > 0 else 0

        return {
            "total_queries": self._query_count,
            "successful_queries": self._query_count - self._error_count,
            "failed_queries": self._error_count,
            "success_rate": success_rate,
            "average_response_time": avg_time,
            "total_time": self._total_time,
            "pipeline_type": self.config.retrieval.pipeline_type,
            "llm_model": self.config.llm.model_type,
            "embedding_model": self.config.embedding.model_name,
            "components_initialized": self._initialized
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all pipeline components.

        Returns:
            Dictionary with health status of components
        """
        # Initialize components if not already done to ensure accurate health check
        self._initialize_components()

        health = {
            "overall": {"healthy": True, "issues": []},
            "components": {}
        }

        try:
            # Check LLM
            if self._llm_generator:
                health["components"]["llm"] = {"status": "healthy", "model": self.config.llm.model_type}
            else:
                health["components"]["llm"] = {"status": "not_initialized"}
                health["overall"]["issues"].append("LLM generator not initialized")

            # Check vector store (always available in both baseline and advanced modes)
            vector_store_accessible = False
            doc_count = 0

            if self.config.retrieval.pipeline_type == "baseline" and self._vector_store:
                stats = self._vector_store.get_collection_stats()
                doc_count = stats.get("document_count", 0)
                vector_store_accessible = True
                health["components"]["vector_store"] = {
                    "status": "healthy" if doc_count > 0 else "empty",
                    "documents": doc_count
                }
                if doc_count == 0:
                    health["overall"]["issues"].append("Vector store is empty")

            # Check advanced components
            elif self.config.retrieval.pipeline_type == "advanced" and self._unified_index_manager:
                unified_health = self._unified_index_manager.health_check()
                health["components"]["unified_index_manager"] = unified_health.get("overall", {})

                # Get comprehensive stats to extract document counts
                try:
                    comprehensive_stats = self._unified_index_manager.get_unified_stats()
                    vector_stats = comprehensive_stats.get("vector_store", {})
                    bm25_stats = comprehensive_stats.get("bm25_index", {})

                    doc_count = vector_stats.get("document_count", 0)
                    bm25_doc_count = bm25_stats.get("documents_count", 0)
                    vector_store_accessible = unified_health.get("overall", {}).get("healthy", False)

                    if doc_count == 0:
                        health["overall"]["issues"].append("Vector store is empty")

                except Exception as e:
                    self.logger.warning(f"Failed to get comprehensive stats: {e}")
                    doc_count = 0
                    bm25_doc_count = 0
                    vector_store_accessible = False

                # First update with unified_health components
                health["components"].update(unified_health.get("components", {}))

                if not unified_health.get("overall", {}).get("healthy", False):
                    health["overall"]["issues"].extend(unified_health.get("overall", {}).get("issues", []))
                
                # THEN override with explicit dict format (after update to prevent overwrite)
                health["components"]["vector_store"] = {
                    "status": "healthy" if doc_count > 0 else "empty",
                    "documents": doc_count
                }
                health["components"]["bm25_index"] = {
                    "status": "healthy" if bm25_doc_count > 0 else "empty",
                    "documents": bm25_doc_count
                }

            # Ensure vector_store component exists for validation compatibility
            if "vector_store" not in health["components"]:
                health["components"]["vector_store"] = {
                    "status": "not_accessible" if not vector_store_accessible else "empty",
                    "documents": doc_count
                }

            # Overall health
            health["overall"]["healthy"] = len(health["overall"]["issues"]) == 0

        except Exception as e:
            health["overall"]["healthy"] = False
            health["overall"]["issues"].append(f"Health check failed: {e}")

        # Add backward compatibility for legacy 'status' key
        health["status"] = health["overall"]["healthy"]
        health["ready"] = health["overall"]["healthy"]

        return health

    def set_mode(
        self,
        research_mode: Optional[bool] = None,
        response_format: Optional[str] = None
    ) -> None:
        """
        Dynamically switch between research and production modes.

        Args:
            research_mode: Enable research mode with detailed metrics
            response_format: Response format ("simple", "full", "api")
        """
        if research_mode is not None:
            self.research_mode = research_mode
            self.logger.info(f"Research mode set to: {research_mode}")

        if response_format is not None:
            self.response_format = response_format
            self._validate_response_format()
            self.logger.info(f"Response format set to: {response_format}")

    def get_mode_info(self) -> Dict[str, Any]:
        """
        Get current mode configuration.

        Returns:
            Dictionary with current mode settings
        """
        return {
            "research_mode": self.research_mode,
            "response_format": self.response_format,
            "description": self._get_mode_description()
        }

    def _get_mode_description(self) -> str:
        """Get description of current mode."""
        if self.research_mode:
            if self.response_format == "full":
                return "Research mode with full metrics and detailed analysis"
            elif self.response_format == "api":
                return "Research mode with API format + detailed metrics"
            else:
                return "Research mode with simple format + basic metrics"
        else:
            if self.response_format == "full":
                return "Production mode with enhanced response details"
            elif self.response_format == "api":
                return "Production mode optimized for API integration"
            else:
                return "Production mode optimized for chatbot responses"

    def refresh_indexes(self) -> bool:
        """
        Refresh indexes by reinitializing the unified index manager.
        
        Call this after external reindexing to load newly indexed data from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.config.retrieval.pipeline_type == "advanced":
                self.logger.info("Refreshing indexes by reinitializing unified index manager...")
                self._initialize_advanced_components()
                self.logger.info("Indexes refreshed successfully")
                return True
            elif self.config.retrieval.pipeline_type == "baseline":
                # Baseline uses vector_store which should auto-load from ChromaDB
                self.logger.info("Refreshing baseline vector store...")
                self._initialize_baseline_components()
                self.logger.info("Vector store refreshed successfully")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to refresh indexes: {e}")
            return False

    def batch_query(
        self,
        questions: List[str],
        max_results: int = 5,
        include_metrics: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.

        Args:
            questions: List of questions to process
            max_results: Maximum number of sources per question
            include_metrics: Override automatic metric inclusion

        Returns:
            List of responses for each question
        """
        results = []

        for i, question in enumerate(questions):
            self.logger.debug(f"Processing batch query {i+1}/{len(questions)}")

            result = self.query(
                question,
                max_results=max_results,
                include_metrics=include_metrics
            )
            results.append(result)

        return results

    def get_supported_query_types(self) -> Dict[str, str]:
        """
        Get list of supported query types for documentation.

        Returns:
            Dictionary with query type descriptions
        """
        return {
            "procedural": "Questions about how to do something (e.g., 'How to register for courses?')",
            "informational": "Questions seeking information (e.g., 'What are the scholarship requirements?')",
            "factual": "Questions with specific answers (e.g., 'How much is tuition fee?')",
            "deadline": "Questions about dates and times (e.g., 'When is the registration deadline?')",
            "directory": "Questions about people/offices (e.g., 'Who are the thesis supervisors?')",
            "facilities": "Questions about facilities and services (e.g., 'What labs are available?')"
        }

    def get_model_performance_stats(self) -> Dict[str, Any]:
        """
        Get model performance statistics for reference.

        Returns:
            Dictionary with performance information
        """
        return {
            "pipeline_comparison": {
                "baseline": "Vector-only search with IndoBERT embeddings",
                "advanced": "Hybrid BM25 + Vector search with cross-encoder reranking",
                "expected_improvement": "5-15% better retrieval performance"
            },
            "performance_metrics": {
                "average_response_time": "1-3 seconds per query",
                "supported_languages": ["Indonesian (Bahasa Indonesia)"],
                "document_types": ["PDF academic documents", "Faculty regulations", "Course materials"],
                "current_mode": self.get_mode_info()
            },
            "research_validation": {
                "evaluation_dataset": "75 Indonesian academic queries",
                "statistical_significance": "Validated with p < 0.05",
                "quality_metrics": ["MRR", "MAP", "RAGAS Faithfulness", "RAGAS Answer Relevancy"]
            },
            "usage_stats": {
                "total_queries": self._query_count,
                "total_time": self._total_time,
                "error_count": self._error_count,
                "average_time": self._total_time / max(1, self._query_count)
            }
        }

    def enhanced_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check with test query.

        Returns:
            Dictionary with detailed health status and model information
        """
        try:
            # Test query
            test_question = "Apa itu teknologi informasi?"
            start_time = time.time()
            test_result = self.query(test_question, max_results=1, include_metrics=True)
            test_time = time.time() - start_time

            # Get basic health
            basic_health = self.health_check()

            # Enhanced health information
            health_status = {
                "status": "healthy",
                "model_loaded": True,
                "test_query_success": test_result.get("error") is None,
                "test_response_time": test_time,
                "mode_info": self.get_mode_info(),
                "model_info": self._get_model_info(),
                "basic_health": basic_health,
                "performance_stats": {
                    "retrieval_time": test_result.get("metadata", {}).get("retrieval_time", 0),
                    "generation_time": test_result.get("metadata", {}).get("generation_time", 0),
                    "total_time": test_result.get("metadata", {}).get("total_time", 0)
                }
            }

        except Exception as e:
            health_status = {
                "status": "unhealthy",
                "model_loaded": False,
                "test_query_success": False,
                "error": str(e),
                "mode_info": self.get_mode_info(),
                "model_info": self._get_model_info(),
                "test_response_time": 0
            }

        return health_status

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._query_count = 0
        self._total_time = 0.0
        self._error_count = 0
        self.logger.info("Usage statistics reset")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources if needed
        pass
