"""Hybrid retriever combining BM25 and vector search with RRF fusion."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..indexing.unified_index_manager import UnifiedIndexManager
from ..utils.logging import PipelineLogger
from ..utils.helpers import format_retrieval_results, validate_query, calculate_rerf_score
from ..utils.query_optimizer import QueryOptimizer
from ..utils.result_fusion import ResultFusion


class HybridSearchRetriever:
    """
    Combine Python BM25 and ChromaDB vector search using Reciprocal Rank Fusion (RRF).

    Implements advanced retrieval that combines the strengths of both lexical (BM25) and
    semantic (vector) search approaches using RRF to merge and rerank results.
    """

    def __init__(
        self,
        unified_index_manager: UnifiedIndexManager,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        logger: Optional[PipelineLogger] = None,
        use_query_optimization: bool = True
    ):
        """
        Initialize hybrid retriever with unified indexes.

        Args:
            unified_index_manager: Manager for unified ChromaDB and BM25 indexes
            rrf_k: RRF constant (default: 60)
            bm25_weight: Weight for BM25 scores in fusion
            vector_weight: Weight for vector scores in fusion
            logger: Optional logger instance
            use_query_optimization: Whether to use multi-query optimization
        """
        self.unified_index_manager = unified_index_manager
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
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
            self.logger = PipelineLogger("HybridSearchRetriever")
            self._is_pipeline_logger = True

        if self._is_pipeline_logger:
            self.logger.logger.info(f"HybridSearchRetriever initialized: RRF_k={rrf_k}, BM25_weight={bm25_weight}, Vector_weight={vector_weight}, query_optimization={use_query_optimization}")
        else:
            self.logger.info(f"HybridSearchRetriever initialized: RRF_k={rrf_k}, BM25_weight={bm25_weight}, Vector_weight={vector_weight}, query_optimization={use_query_optimization}")

    def retrieve(
        self,
        query: str,
        k: int = 10,
        bm25_k: int = 50,
        vector_k: int = 50,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_reranking: bool = False,
        reranker: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval workflow using RRF fusion with optional query optimization.

        Args:
            query: Search query string
            k: Final number of documents to return
            bm25_k: Number of documents to retrieve from BM25
            vector_k: Number of documents to retrieve from vector search
            filter_dict: Optional metadata filters
            use_reranking: Whether to apply reranking
            reranker: Optional reranker instance

        Returns:
            List of retrieved documents with combined scores and metadata

        Raises:
            ValueError: If query is invalid
        """
        # Validate query
        validate_query(query)

        start_time = time.time()
        query_id = f"hybrid_{int(start_time * 1000)}"

        self.logger.info(f"[HybridRetriever] Processing query: {query[:50]}...")

        try:
            if self.use_query_optimization and self.query_optimizer:
                # Use enhanced retrieval with query optimization
                return self._enhanced_hybrid_retrieve(
                    query, k, bm25_k, vector_k, filter_dict,
                    use_reranking, reranker, query_id, start_time
                )
            else:
                # Standard hybrid retrieval
                return self._standard_hybrid_retrieve(
                    query, k, bm25_k, vector_k, filter_dict,
                    use_reranking, reranker, query_id, start_time
                )

        except Exception as e:
            latency = time.time() - start_time
            if self._is_pipeline_logger:
                self.logger.log_error("HybridRetriever", e, {"query": query, "latency": latency})
            else:
                self.logger.error(f"HybridRetriever error: {e}")
            raise

    def _standard_hybrid_retrieve(
        self,
        query: str,
        k: int,
        bm25_k: int,
        vector_k: int,
        filter_dict: Optional[Dict[str, Any]],
        use_reranking: bool,
        reranker: Optional[Any],
        query_id: str,
        start_time: float
    ) -> List[Dict[str, Any]]:
        """Standard hybrid retrieval without query optimization."""
        # Step 1: Parallel retrieval from both indexes
        retrieval_start = time.time()
        parallel_results = self.dual_index_manager.search_parallel(
            query=query,
            k=max(bm25_k, vector_k),
            vector_k=vector_k,
            bm25_k=bm25_k
        )
        retrieval_time = time.time() - retrieval_start

        bm25_results = parallel_results["bm25"]
        vector_results = parallel_results["vector"]

        self.logger.info(f"[HybridRetriever] Retrieved {len(bm25_results)} from BM25, {len(vector_results)} from vector search in {retrieval_time:.3f}s")

        # Step 2: Apply RRF fusion
        fusion_start = time.time()
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            vector_results
        )
        fusion_time = time.time() - fusion_start

        self.logger.info(f"[HybridRetriever] RRF fusion completed in {fusion_time:.3f}s")

        # Step 3: Apply reranking if requested
        if use_reranking and reranker and len(fused_results) > k:
            rerank_start = time.time()
            fused_results = reranker.rerank(query, fused_results[:max(20, k * 2)])
            rerank_time = time.time() - rerank_start
            self.logger.info(f"[HybridRetriever] Reranking completed in {rerank_time:.3f}s")
        else:
            rerank_time = 0

        # Step 4: Final formatting and ranking
        final_results = self._format_hybrid_results(
            fused_results[:k],
            query_id,
            start_time
        )

        total_latency = time.time() - start_time

        # Log retrieval operation
        if self._is_pipeline_logger:
            self.logger.log_retrieval(
                query_id=query_id,
                query=query,
                num_retrieved=len(final_results),
                latency=total_latency,
                component="HybridRetriever"
            )
        else:
            self.logger.info(f"[HybridRetriever] Query {query_id}: \"{query[:50]}...\" | Retrieved: {len(final_results)} docs | Latency: {total_latency:.3f}s")

        if self._is_pipeline_logger:
            self.logger.logger.info(f"[HybridRetriever] Final results: {len(final_results)} documents in {total_latency:.3f}s (retrieval: {retrieval_time:.3f}s, fusion: {fusion_time:.3f}s, rerank: {rerank_time:.3f}s)")
        else:
            self.logger.info(f"[HybridRetriever] Final results: {len(final_results)} documents in {total_latency:.3f}s (retrieval: {retrieval_time:.3f}s, fusion: {fusion_time:.3f}s, rerank: {rerank_time:.3f}s)")

        return final_results

    def _enhanced_hybrid_retrieve(
        self,
        query: str,
        k: int,
        bm25_k: int,
        vector_k: int,
        filter_dict: Optional[Dict[str, Any]],
        use_reranking: bool,
        reranker: Optional[Any],
        query_id: str,
        start_time: float
    ) -> List[Dict[str, Any]]:
        """Enhanced hybrid retrieval with multi-query optimization."""

        # Generate query variations
        query_variations = self.query_optimizer.optimize_query(query)
        self.logger.info(f"[HybridRetriever-Enhanced] Generated {len(query_variations)} query variations")

        # Limit variations for performance
        max_variations = 5  # Configurable constant for query optimization
        query_variations = query_variations[:max_variations]

        # Retrieve results for each query variation
        all_results_sets = []
        for i, var_query in enumerate(query_variations):
            try:
                # Direct retrieval using dual index manager for this variation
                parallel_results = self.dual_index_manager.search_parallel(
                    query=var_query,
                    k=max(bm25_k, vector_k),
                    vector_k=vector_k,
                    bm25_k=bm25_k
                )

                var_fused = self._reciprocal_rank_fusion(
                    parallel_results["bm25"],
                    parallel_results["vector"]
                )

                all_results_sets.append(var_fused)
                self.logger.debug(f"[HybridRetriever-Enhanced] Query variation {i+1}: {len(var_fused)} results")

            except Exception as e:
                self.logger.warning(f"[HybridRetriever-Enhanced] Query variation {i+1} failed: {e}")

        if not all_results_sets:
            return []

        # Fuse all result sets using result fusion utility
        final_fused = ResultFusion.fuse_results(all_results_sets, query, strategy="rrf")

        # Apply reranking if requested
        if use_reranking and reranker and len(final_fused) > k:
            rerank_start = time.time()
            final_fused = reranker.rerank(query, final_fused[:max(20, k * 2)])
            rerank_time = time.time() - rerank_start
            self.logger.info(f"[HybridRetriever-Enhanced] Reranking completed in {rerank_time:.3f}s")
        else:
            rerank_time = 0

        # Final formatting
        final_results = self._format_hybrid_results(
            final_fused[:k],
            query_id,
            start_time
        )

        # Add enhanced retrieval metadata
        total_latency = time.time() - start_time
        for result in final_results:
            result['query_variations_used'] = len(query_variations)
            result['enhanced_retrieval'] = True

        # Log retrieval operation
        if self._is_pipeline_logger:
            self.logger.log_retrieval(
                query_id=query_id,
                query=query,
                num_retrieved=len(final_results),
                latency=total_latency,
                component="HybridRetriever-Enhanced"
            )
        else:
            self.logger.info(f"[HybridRetriever-Enhanced] Query {query_id}: \"{query[:50]}...\" | Retrieved: {len(final_results)} docs | Latency: {total_latency:.3f}s")

        if self._is_pipeline_logger:
            self.logger.logger.info(f"[HybridRetriever-Enhanced] Final results: {len(final_results)} documents in {total_latency:.3f}s using {len(query_variations)} query variations")
        else:
            self.logger.info(f"[HybridRetriever-Enhanced] Final results: {len(final_results)} documents in {total_latency:.3f}s using {len(query_variations)} query variations")

        return final_results

    def search_vector_only(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using vector store only (for enhanced retriever compatibility).

        Args:
            query: Search query string
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters

        Returns:
            List of retrieved documents
        """
        try:
            raw_results = self.dual_index_manager.vector_store.similarity_search(
                query=query,
                k=k,
                filter_dict=filter_dict
            )
            return format_retrieval_results(raw_results, add_rank=True)
        except Exception as e:
            self.logger.error(f"Vector-only search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge BM25 and vector search results using RRF algorithm.

        RRF Formula: score(d) = Σ(w_i / (k + rank_i(d)))
        where w_i is the weight for retrieval method i, and rank_i(d) is the rank of document d in method i.

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search

        Returns:
            Fused and reranked results
        """
        # Create document lookup tables
        bm25_lookup = {}
        vector_lookup = {}

        # BM25 lookup with normalized scores
        for rank, result in enumerate(bm25_results, 1):
            doc_id = self._get_document_id(result)
            bm25_lookup[doc_id] = {
                'result': result,
                'rank': rank,
                'normalized_score': self._normalize_score(result.get('score', 0), bm25_results)
            }

        # Vector lookup with normalized scores
        for rank, result in enumerate(vector_results, 1):
            doc_id = self._get_document_id(result)
            vector_lookup[doc_id] = {
                'result': result,
                'rank': rank,
                'normalized_score': self._normalize_score(result.get('score', 0), vector_results)
            }

        # Combine documents from both result sets
        all_doc_ids = set(bm25_lookup.keys()) | set(vector_lookup.keys())
        fused_results = []

        for doc_id in all_doc_ids:
            # Get RRF score components
            rrf_score = 0
            components = {
                'bm25_score': 0,
                'vector_score': 0,
                'bm25_rank': None,
                'vector_rank': None
            }

            # BM25 contribution
            if doc_id in bm25_lookup:
                bm25_data = bm25_lookup[doc_id]
                rrf_component = self.bm25_weight / (self.rrf_k + bm25_data['rank'])
                rrf_score += rrf_component
                components['bm25_score'] = rrf_component
                components['bm25_rank'] = bm25_data['rank']

            # Vector contribution
            if doc_id in vector_lookup:
                vector_data = vector_lookup[doc_id]
                rrf_component = self.vector_weight / (self.rrf_k + vector_data['rank'])
                rrf_score += rrf_component
                components['vector_score'] = rrf_component
                components['vector_rank'] = vector_data['rank']

            # Get the best available document (prefer the one with higher original score)
            best_result = None
            if doc_id in bm25_lookup and doc_id in vector_lookup:
                best_result = bm25_lookup[doc_id]['result'] if bm25_lookup[doc_id]['normalized_score'] > vector_lookup[doc_id]['normalized_score'] else vector_lookup[doc_id]['result']
            elif doc_id in bm25_lookup:
                best_result = bm25_lookup[doc_id]['result']
            elif doc_id in vector_lookup:
                best_result = vector_lookup[doc_id]['result']

            if best_result:
                fused_result = best_result.copy()
                fused_result['rrf_score'] = rrf_score
                fused_result['rrf_components'] = components
                fused_results.append(fused_result)

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)

        return fused_results

    def _get_document_id(self, result: Dict[str, Any]) -> str:
        """
        Get unique document identifier from result.

        Args:
            result: Retrieval result

        Returns:
            Document identifier
        """
        metadata = result.get('metadata', {})
        return metadata.get('global_chunk_id') or metadata.get('chunk_id', '') or result.get('id', '')

    def _normalize_score(self, score: float, all_results: List[Dict[str, Any]]) -> float:
        """
        Normalize score to [0, 1] range.

        Args:
            score: Original score
            all_results: All results for normalization

        Returns:
            Normalized score
        """
        if not all_results:
            return 0.0

        scores = [r.get('score', 0) for r in all_results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return 1.0  # All scores are the same

        return (score - min_score) / (max_score - min_score)

    def _format_hybrid_results(
        self,
        fused_results: List[Dict[str, Any]],
        query_id: str,
        start_time: float
    ) -> List[Dict[str, Any]]:
        """
        Format hybrid results with consistent structure.

        Args:
            fused_results: Fused results from RRF
            query_id: Query identifier
            start_time: Retrieval start time

        Returns:
            Formatted results
        """
        formatted_results = []
        total_latency = time.time() - start_time

        for rank, result in enumerate(fused_results, 1):
            formatted_result = {
                'text': result.get('text', ''),
                'score': result.get('rrf_score', 0.0),
                'rank': rank,
                'metadata': result.get('metadata', {}),
                'retrieval_method': 'hybrid_search',
                'query_id': query_id,
                'retrieval_latency': total_latency,
                'rrf_components': result.get('rrf_components', {}),
                'original_scores': {
                    'bm25_score': result.get('rrf_components', {}).get('bm25_score', 0),
                    'vector_score': result.get('rrf_components', {}).get('vector_score', 0)
                }
            }

            formatted_results.append(formatted_result)

        return formatted_results

    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 10,
        use_reranking: bool = False,
        reranker: Optional[Any] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries using hybrid search.

        Args:
            queries: List of search queries
            k: Number of documents to retrieve per query
            use_reranking: Whether to apply reranking
            reranker: Optional reranker instance

        Returns:
            List of retrieval results for each query
        """
        self.logger.info(f"[HybridRetriever] Processing {len(queries)} queries in batch")

        results = []
        total_start_time = time.time()

        for i, query in enumerate(queries):
            try:
                query_results = self.retrieve(
                    query=query,
                    k=k,
                    use_reranking=use_reranking,
                    reranker=reranker
                )
                results.append(query_results)

                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"[HybridRetriever] Processed {i + 1}/{len(queries)} queries")

            except Exception as e:
                self.logger.error(f"[HybridRetriever] Failed to process query {i + 1}: {e}")
                results.append([])  # Empty results for failed query

        total_latency = time.time() - total_start_time
        avg_latency = total_latency / len(queries) if queries else 0

        self.logger.info(f"[HybridRetriever] Batch completed in {total_latency:.3f}s (avg: {avg_latency:.3f}s per query)")

        return results

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information for the hybrid retriever.

        Returns:
            Dictionary with configuration details
        """
        return {
            "retriever_type": "hybrid",
            "retrieval_method": "bm25_vector_fusion",
            "rrf_parameters": {
                "k": self.rrf_k,
                "bm25_weight": self.bm25_weight,
                "vector_weight": self.vector_weight
            },
            "indexes": {
                "bm25": {
                    "cache_dir": self.unified_index_manager.cache_dir,
                    "vocabulary_size": getattr(self.unified_index_manager.bm25_index, 'vocabulary_size', 'N/A')
                },
                "chromadb": {
                    "collection_name": self.unified_index_manager.vector_store.collection_name,
                    "embedding_model": self.unified_index_manager.vector_store.embedding_config.model_name
                }
            },
            "capabilities": [
                "BM25 lexical search",
                "Vector semantic search",
                "Reciprocal Rank Fusion",
                "Parallel retrieval",
                "Score normalization",
                "Hybrid scoring"
            ]
        }