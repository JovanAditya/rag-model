"""Result fusion utilities for combining multiple retrieval results."""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import logging


class ResultFusion:
    """Utility class for fusing results from multiple retrievals."""

    @staticmethod
    def fuse_results(all_results: List[Dict], query: str, strategy: str = "rrf",
                    rrf_k: int = 60, weight_vector: float = 0.7, weight_bm25: float = 0.3) -> List[Dict]:
        """
        Fuse results from multiple retrieval strategies.

        Args:
            all_results: List of result dictionaries
            query: Original query
            strategy: Fusion strategy ("rrf", "weighted", "max")
            rrf_k: RRF parameter
            weight_vector: Weight for vector search results
            weight_bm25: Weight for BM25 search results

        Returns:
            Fused and ranked results
        """
        if not all_results:
            return []

        if strategy == "rrf":
            return ResultFusion._rrf_fusion(all_results, query, rrf_k)
        elif strategy == "weighted":
            return ResultFusion._weighted_fusion(all_results, query, weight_vector, weight_bm25)
        elif strategy == "max":
            return ResultFusion._max_fusion(all_results)
        else:
            # Default to simple scoring
            return ResultFusion._simple_fusion(all_results)

    @staticmethod
    def _rrf_fusion(all_results: List[Dict], query: str, k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) for combining results.

        Args:
            all_results: List of result dictionaries
            query: Original query
            k: RRF parameter

        Returns:
            RRF-fused results
        """
        if not all_results:
            return []

        # Track scores for each document
        doc_scores = defaultdict(float)
        doc_sources = defaultdict(list)

        for results in all_results:
            if not results:
                continue

            for rank, doc in enumerate(results, 1):
                doc_id = doc.get('id', hash(str(doc.get('content', ''))))
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += rrf_score

                # Store source information
                source_info = {
                    'retrieval_type': doc.get('retrieval_type', 'unknown'),
                    'original_score': doc.get('score', 0.0),
                    'rank': rank
                }
                doc_sources[doc_id].append(source_info)

        # Create fused results
        fused_results = []
        for doc_id, rrf_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            # Find original document
            original_doc = None
            for results in all_results:
                for doc in results:
                    current_id = doc.get('id', hash(str(doc.get('content', ''))))
                    if current_id == doc_id:
                        original_doc = doc
                        break
                if original_doc:
                    break

            if original_doc:
                fused_doc = original_doc.copy()
                fused_doc['score'] = rrf_score
                fused_doc['sources'] = doc_sources.get(doc_id, [])
                fused_results.append(fused_doc)

        return fused_results

    @staticmethod
    def _weighted_fusion(all_results: List[Dict], query: str,
                         weight_vector: float = 0.7, weight_bm25: float = 0.3) -> List[Dict]:
        """
        Weighted fusion for combining vector and BM25 results.

        Args:
            all_results: List of result dictionaries (vector, BM25)
            query: Original query
            weight_vector: Weight for vector search results
            weight_bm25: Weight for BM25 search results

        Returns:
            Weighted-fused results
        """
        if len(all_results) < 2:
            return all_results[0] if all_results else []

        # Normalize weights
        total_weight = weight_vector + weight_bm25
        w_vector = weight_vector / total_weight
        w_bm25 = weight_bm25 / total_weight

        # Extract results (assuming first is vector, second is BM25)
        vector_results = all_results[0] if len(all_results) > 0 else []
        bm25_results = all_results[1] if len(all_results) > 1 else []

        # Create document lookup
        vector_docs = {doc.get('id', hash(str(doc.get('content', '')))): doc for doc in vector_results}
        bm25_docs = {doc.get('id', hash(str(doc.get('content', '')))): doc for doc in bm25_results}

        # Combine results
        all_doc_ids = set(vector_docs.keys()) | set(bm25_docs.keys())
        fused_results = []

        for doc_id in all_doc_ids:
            vector_doc = vector_docs.get(doc_id)
            bm25_doc = bm25_docs.get(doc_id)

            # Calculate weighted score
            vector_score = vector_doc.get('score', 0.0) if vector_doc else 0.0
            bm25_score = bm25_doc.get('score', 0.0) if bm25_doc else 0.0

            weighted_score = w_vector * vector_score + w_bm25 * bm25_score

            # Use the document with higher original score as base
            base_doc = vector_doc if vector_score >= bm25_score else bm25_doc

            if base_doc:
                fused_doc = base_doc.copy()
                fused_doc['score'] = weighted_score
                fused_doc['sources'] = []

                if vector_doc:
                    fused_doc['sources'].append({'type': 'vector', 'score': vector_score})
                if bm25_doc:
                    fused_doc['sources'].append({'type': 'bm25', 'score': bm25_score})

                fused_results.append(fused_doc)

        # Sort by final score
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        return fused_results

    @staticmethod
    def _max_fusion(all_results: List[Dict]) -> List[Dict]:
        """
        Max fusion - keep highest score for each document.

        Args:
            all_results: List of result dictionaries

        Returns:
            Max-fused results
        """
        if not all_results:
            return []

        # Track best scores for each document
        best_docs = {}

        for results in all_results:
            for doc in results:
                doc_id = doc.get('id', hash(str(doc.get('content', ''))))
                current_score = doc.get('score', 0.0)

                if doc_id not in best_docs or current_score > best_docs[doc_id]['score']:
                    best_docs[doc_id] = doc.copy()

        # Return sorted results
        fused_results = list(best_docs.values())
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        return fused_results

    @staticmethod
    def _simple_fusion(all_results: List[Dict]) -> List[Dict]:
        """
        Simple fusion - combine all results and sort by score.

        Args:
            all_results: List of result dictionaries

        Returns:
            Simply fused results
        """
        if not all_results:
            return []

        # Flatten all results
        all_docs = []
        doc_ids = set()  # Track seen documents to avoid duplicates

        for results in all_results:
            for doc in results:
                doc_id = doc.get('id', hash(str(doc.get('content', ''))))
                if doc_id not in doc_ids:
                    all_docs.append(doc.copy())
                    doc_ids.add(doc_id)
                else:
                    # If duplicate, keep the one with higher score
                    for i, existing_doc in enumerate(all_docs):
                        existing_id = existing_doc.get('id', hash(str(existing_doc.get('content', ''))))
                        if existing_id == doc_id and doc.get('score', 0.0) > existing_doc.get('score', 0.0):
                            all_docs[i] = doc.copy()
                            break

        # Sort by score
        all_docs.sort(key=lambda x: x['score'], reverse=True)
        return all_docs