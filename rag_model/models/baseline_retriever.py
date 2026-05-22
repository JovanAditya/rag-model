"""Baseline retriever using vector search only for Advanced RAG Pipeline."""

import logging
import time
from typing import List, Dict, Any, Optional

from ..indexing.vector_store import VectorStore
from ..utils.logging import PipelineLogger
from ..utils.helpers import format_retrieval_results, validate_query
from ..utils.query_optimizer import QueryOptimizer
from ..utils.result_fusion import ResultFusion


class BaselineRetriever:
    """
    Standard RAG retrieval using vector search only.

    Serves as the baseline for comparison with the advanced hybrid retrieval approach.
    Uses only IndoBERT vector search with ChromaDB for document retrieval.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        logger: Optional[PipelineLogger] = None,
        use_query_optimization: bool = True
    ):
        """
        Initialize baseline retriever.

        Args:
            vector_store: ChromaDB vector store instance
            logger: Optional logger instance
            use_query_optimization: Whether to use multi-query optimization
        """
        self.vector_store = vector_store
        self.use_query_optimization = use_query_optimization
        self.query_optimizer = QueryOptimizer() if use_query_optimization else None

        # Handle different logger types
        if logger and hasattr(logger, 'log_error'):
            # PipelineLogger
            self.logger = logger
            self._is_pipeline_logger = True
        elif logger:
            # Standard Python logger
            self.logger = logger
            self._is_pipeline_logger = False
        else:
            # Create a PipelineLogger instance for consistency
            self.logger = PipelineLogger("BaselineRetriever")
            self._is_pipeline_logger = True

        if self._is_pipeline_logger:
            self.logger.logger.info(f"BaselineRetriever initialized (vector search only, query_optimization={use_query_optimization})")
        else:
            self.logger.info(f"BaselineRetriever initialized (vector search only, query_optimization={use_query_optimization})")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using vector search with optional query optimization.

        Args:
            query: Search query string
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters

        Returns:
            List of retrieved documents with scores and metadata

        Raises:
            ValueError: If query is invalid
        """
        # Validate query
        validate_query(query)

        start_time = time.time()
        query_id = f"baseline_{int(start_time * 1000)}"

        self.logger.info(f"[BaselineRetriever] Processing query: {query[:50]}...")

        try:
            if self.use_query_optimization and self.query_optimizer:
                # Use multi-query optimization
                return self._multi_query_retrieve(query, k, filter_dict, query_id, start_time)
            else:
                # Single query retrieval
                return self._single_query_retrieve(query, k, filter_dict, query_id, start_time)

        except Exception as e:
            latency = time.time() - start_time
            if self._is_pipeline_logger:
                self.logger.log_error("BaselineRetriever", e, {"query": query, "latency": latency})
            else:
                self.logger.error(f"BaselineRetriever error: {e}")
            raise

    def _single_query_retrieve(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict[str, Any]],
        query_id: str,
        start_time: float
    ) -> List[Dict[str, Any]]:
        """Perform single query retrieval."""
        # Perform vector search
        raw_results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict
        )

        # Format results
        formatted_results = format_retrieval_results(
            raw_results,
            add_rank=True
        )

        # Calculate retrieval latency
        latency = time.time() - start_time

        # Log retrieval operation
        if self._is_pipeline_logger:
            self.logger.log_retrieval(
                query_id=query_id,
                query=query,
                num_retrieved=len(formatted_results),
                latency=latency,
                component="BaselineRetriever"
            )
        else:
            self.logger.info(f"[BaselineRetriever] Query {query_id}: \"{query[:50]}...\" | Retrieved: {len(formatted_results)} docs | Latency: {latency:.3f}s")

        # Add additional metadata
        for result in formatted_results:
            result['retrieval_method'] = 'vector_search'
            result['query_id'] = query_id
            result['retrieval_latency'] = latency

        if self._is_pipeline_logger:
            self.logger.logger.info(f"[BaselineRetriever] Retrieved {len(formatted_results)} documents in {latency:.3f}s")
        else:
            self.logger.info(f"[BaselineRetriever] Retrieved {len(formatted_results)} documents in {latency:.3f}s")

        return formatted_results

    def _multi_query_retrieve(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict[str, Any]],
        query_id: str,
        start_time: float
    ) -> List[Dict[str, Any]]:
        """Perform multi-query optimization with result fusion."""

        # Generate query variations
        query_variations = self.query_optimizer.optimize_query(query)
        self.logger.info(f"[BaselineRetriever] Generated {len(query_variations)} query variations")

        # Limit variations for performance
        max_variations = 5  # Configurable constant for query optimization
        query_variations = query_variations[:max_variations]

        # Retrieve results for each query variation
        all_results = []
        for i, var_query in enumerate(query_variations):
            try:
                var_results = self.vector_store.similarity_search(
                    query=var_query,
                    k=k,
                    filter_dict=filter_dict
                )
                all_results.append(var_results)
                self.logger.debug(f"[BaselineRetriever] Query variation {i+1}: {len(var_results)} results")
            except Exception as e:
                self.logger.warning(f"[BaselineRetriever] Query variation {i+1} failed: {e}")

        if not all_results:
            return []

        # Fuse results using result fusion utility
        fused_results = ResultFusion.fuse_results(all_results, query, strategy="rrf")
        
        # Batasi hasil akhir sesuai nilai k yang diminta
        fused_results = fused_results[:k]

        # Add metadata
        latency = time.time() - start_time
        for result in fused_results:
            result['retrieval_method'] = 'enhanced_vector_search'
            result['query_id'] = query_id
            result['retrieval_latency'] = latency
            result['query_variations_used'] = len(query_variations)

        # Log retrieval operation
        if self._is_pipeline_logger:
            self.logger.log_retrieval(
                query_id=query_id,
                query=query,
                num_retrieved=len(fused_results),
                latency=latency,
                component="BaselineRetriever-Enhanced"
            )
        else:
            self.logger.info(f"[BaselineRetriever-Enhanced] Query {query_id}: \"{query[:50]}...\" | Retrieved: {len(fused_results)} docs | Latency: {latency:.3f}s")

        if self._is_pipeline_logger:
            self.logger.logger.info(f"[BaselineRetriever-Enhanced] Retrieved {len(fused_results)} documents in {latency:.3f}s using {len(query_variations)} query variations")
        else:
            self.logger.info(f"[BaselineRetriever-Enhanced] Retrieved {len(fused_results)} documents in {latency:.3f}s using {len(query_variations)} query variations")

        return fused_results

    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of search queries
            k: Number of documents to retrieve per query
            filter_dict: Optional metadata filters

        Returns:
            List of retrieval results for each query
        """
        self.logger.info(f"[BaselineRetriever] Processing {len(queries)} queries in batch")

        results = []
        total_start_time = time.time()

        for i, query in enumerate(queries):
            try:
                query_results = self.retrieve(query, k, filter_dict)
                results.append(query_results)

                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"[BaselineRetriever] Processed {i + 1}/{len(queries)} queries")

            except Exception as e:
                self.logger.error(f"[BaselineRetriever] Failed to process query {i + 1}: {e}")
                results.append([])  # Empty results for failed query

        total_latency = time.time() - total_start_time
        avg_latency = total_latency / len(queries) if queries else 0

        self.logger.info(f"[BaselineRetriever] Batch completed in {total_latency:.3f}s (avg: {avg_latency:.3f}s per query)")

        return results

    def get_retrieval_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for retrieval results.

        Args:
            results: Retrieval results

        Returns:
            Dictionary with retrieval statistics
        """
        if not results:
            return {
                "num_results": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "score_distribution": {}
            }

        scores = [result.get('score', 0.0) for result in results]

        # Calculate score distribution
        score_ranges = {
            "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
            "0.6-0.8": 0, "0.8-1.0": 0
        }

        for score in scores:
            if score < 0.2:
                score_ranges["0.0-0.2"] += 1
            elif score < 0.4:
                score_ranges["0.2-0.4"] += 1
            elif score < 0.6:
                score_ranges["0.4-0.6"] += 1
            elif score < 0.8:
                score_ranges["0.6-0.8"] += 1
            else:
                score_ranges["0.8-1.0"] += 1

        return {
            "num_results": len(results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": score_ranges
        }

    def explain_retrieval(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide explanation for retrieval of a specific document.

        Args:
            query: Search query
            document: Retrieved document

        Returns:
            Explanation dictionary
        """
        return {
            "query": query,
            "document_id": document.get('metadata', {}).get('global_chunk_id', ''),
            "score": document.get('score', 0.0),
            "rank": document.get('rank', 0),
            "retrieval_method": "vector_search",
            "similarity_type": "cosine_similarity",
            "embedding_model": self.vector_store.embedding_config.model_name,
            "source": document.get('metadata', {}).get('source', ''),
            "explanation": "Document retrieved based on semantic similarity using IndoBERT embeddings"
        }

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information for the retriever.

        Returns:
            Dictionary with configuration details
        """
        return {
            "retriever_type": "baseline",
            "retrieval_method": "vector_search_only",
            "vector_store": {
                "collection_name": self.vector_store.collection_name,
                "embedding_model": self.vector_store.embedding_config.model_name,
                "embedding_dimensions": self.vector_store.embedding_config.dimensions,
                "similarity_metric": "cosine"
            },
            "capabilities": [
                "Semantic search with IndoBERT",
                "Cosine similarity scoring",
                "Metadata filtering",
                "Batch processing"
            ],
            "limitations": [
                "No keyword matching",
                "No reranking",
                "Dependent on embedding quality"
            ]
        }