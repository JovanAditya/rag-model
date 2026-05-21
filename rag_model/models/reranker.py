"""Cross-encoder reranker for Advanced RAG Pipeline."""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    logging.error("sentence-transformers not available. Install with: pip install sentence-transformers")
    raise

from ..core.config import RerankerConfig
from ..utils.logging import PipelineLogger
from ..utils.helpers import format_retrieval_results, safe_gpu_memory_cleanup


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval results.

    Uses a cross-encoder model to rerank retrieved documents by scoring
    the relevance of each document to the specific query, rather than
    relying on static embeddings.
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        logger: Optional[any] = None
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            config: Reranker configuration
            logger: Optional logger instance (PipelineLogger or standard Logger)
        """
        self.config = config or RerankerConfig()

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
            from ..utils.logging import PipelineLogger
            self.logger = PipelineLogger("CrossEncoderReranker")
            self._is_pipeline_logger = True

        # Initialize cross-encoder model
        self.model = self._load_model()

        if self._is_pipeline_logger:
            self.logger.logger.info(f"CrossEncoderReranker initialized: {self.config.model_name}")
        else:
            self.logger.info(f"CrossEncoderReranker initialized: {self.config.model_name}")

    def _load_model(self) -> CrossEncoder:
        """
        Load cross-encoder model.

        Returns:
            Loaded CrossEncoder model
        """
        try:
            if self._is_pipeline_logger:
                self.logger.logger.info(f"Loading cross-encoder model: {self.config.model_name}")
            else:
                self.logger.info(f"Loading cross-encoder model: {self.config.model_name}")

            # CrossEncoder parameters - device handling depends on version
            model_kwargs = {
                "max_length": self.config.max_length
            }

            # Only add device parameter if not CPU (some versions don't support device param)
            if self.config.device != "cpu":
                try:
                    model_kwargs["device"] = self.config.device
                except:
                    # If device parameter not supported, fallback to CPU
                    if self._is_pipeline_logger:
                        self.logger.logger.warning(f"Device parameter not supported, using CPU")
                    else:
                        self.logger.warning(f"Device parameter not supported, using CPU")

            model = CrossEncoder(
                self.config.model_name,
                **model_kwargs
            )

            if self._is_pipeline_logger:
                self.logger.logger.info("Cross-encoder model loaded successfully")
            else:
                self.logger.info("Cross-encoder model loaded successfully")

            return model
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return (if None, return all)

        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            self.logger.warning("No documents to rerank")
            return []

        if len(documents) == 1:
            self.logger.debug("Only one document, no reranking needed")
            return documents

        start_time = time.time()
        self.logger.info(f"[CrossEncoderReranker] Reranking {len(documents)} documents for query: {query[:50]}...")

        try:
            # Prepare query-document pairs for scoring
            # Support both 'text' and 'content' keys
            doc_texts = [doc.get('text', doc.get('content', '')) for doc in documents]
            query_doc_pairs = [(query, doc_text) for doc_text in doc_texts]

            # Predict scores in batches
            scores = self._predict_scores_batched(query_doc_pairs)

            # Update document scores
            reranked_docs = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                reranked_doc = doc.copy()
                reranked_doc['original_score'] = doc.get('score', 0.0)
                reranked_doc['cross_encoder_score'] = float(score)
                reranked_doc['rerank_rank'] = None  # Will be set after sorting
                reranked_doc['rerank_improvement'] = float(score) - doc.get('score', 0.0)
                reranked_docs.append(reranked_doc)

            # Post-reranking heuristic boost for acronyms and specific terms
            import re
            acronyms = re.findall(r'\b[A-Z]{2,5}\b', query)
            if acronyms:
                self.logger.info(f"[CrossEncoderReranker] Boosting documents containing acronyms: {acronyms}")
                for doc in reranked_docs:
                    content = doc.get('content', doc.get('text', '')).lower()
                    original_content = doc.get('content', doc.get('text', ''))
                    
                    # Check for exact acronym match (case-sensitive) or word-boundary match in content
                    matched_acros = []
                    for acro in acronyms:
                        if acro in original_content or re.search(rf'\b{acro}\b', content, re.IGNORECASE):
                            matched_acros.append(acro)
                    
                    if matched_acros:
                        # Scale boost based on number of matches to reward docs that hit multiple acronyms (e.g. MPTI and IPK)
                        boost_val = len(matched_acros) * 2.5
                        doc['cross_encoder_score'] += boost_val 
                        if 'metadata' not in doc: doc['metadata'] = {}
                        doc['metadata']['acronym_boosted'] = True
                        doc['metadata']['matched_acronyms'] = matched_acros

            # Faculty and domain specific boosting
            query_lower = query.lower()
            fasilkom_terms = ['fasilkom', 'teknik informatika', 'sistem informasi', 'ilmu komputer', 'fakultas ilmu komputer']
            fasilkom_triggers = ['ta', 'kp', 'skpi', 'mpti', 'klik', 'sidang', 'tugas akhir', 'kerja praktek', 'yudisium']
            is_fasilkom_query = any(term in query_lower for term in fasilkom_triggers)
            
            # Authoritative Document Boost (Sync with UnifiedIndexManager)
            is_ta_trigger = any(term in query_lower for term in ['tugas akhir', 'skripsi', 'sidang akhir', 'yudisium'])
            is_kp_trigger = any(term in query_lower for term in ['kerja praktek', 'magang', 'mbkm', 'sosialisasi kp']) or bool(re.search(r'\bkp\b', query_lower))
            is_mpti_trigger = any(term in query_lower for term in ['mpti', 'metodologi penelitian', 'proposal', 'sempro', 'apa style']) or bool(re.search(r'\bmpti\b', query_lower))
            is_ult_trigger = any(term in query_lower for term in ['ult', 'unit layanan terpadu', 'akun', 'password', 'login', 'pendaftaran']) or bool(re.search(r'\bult\b', query_lower))
            is_pasca_trigger = any(term in query_lower for term in ['pasca sidang', 'setelah sidang', 'setelah lulus', 'revisi', '14 hari', 'toefl', 'toeic', 'skpi', 'yudisium'])

            for doc in reranked_docs:
                metadata = doc.get('metadata', {}) or {}
                source_file = (metadata.get('original_filename') or metadata.get('source') or metadata.get('source_document') or '').lower()
                
                authoritative_boost = 0.0
                
                is_pasca_doc = any(kw in source_file for kw in ['arahan pascasidang', 'press arahan'])
                is_mpti_doc = any(kw in source_file for kw in ['panduan mpti', 'mpti 2023'])
                is_ta_doc = any(kw in source_file for kw in ['panduan tugas akhir', 'skripsi'])
                is_kp_doc = any(kw in source_file for kw in ['sosialisasi kp', 'kerja praktek', 'mbkm'])
                is_ult_doc = any(kw in source_file for kw in ['buku panduan ult', 'panduan ult mahasiswa'])
                
                if is_pasca_trigger or is_ta_trigger:
                    if is_pasca_doc or is_ta_doc:
                        authoritative_boost = 5.0
                    elif is_mpti_doc or is_kp_doc or is_ult_doc:
                        authoritative_boost = -10.0
                if is_mpti_trigger:
                    if is_mpti_doc:
                        authoritative_boost = 5.0
                    elif is_ta_doc or is_kp_doc or is_ult_doc or is_pasca_doc:
                        authoritative_boost = -10.0
                if is_kp_trigger:
                    if is_kp_doc:
                        authoritative_boost = 5.0
                    elif is_ta_doc or is_mpti_doc or is_ult_doc or is_pasca_doc:
                        authoritative_boost = -10.0
                if is_ult_trigger:
                    if is_ult_doc:
                        authoritative_boost = 5.0
                    elif is_ta_doc or is_mpti_doc or is_kp_doc or is_pasca_doc:
                        authoritative_boost = -10.0
                
                if authoritative_boost != 0.0:
                    doc['cross_encoder_score'] += authoritative_boost
                    if 'metadata' not in doc: doc['metadata'] = {}
                    doc['metadata']['authoritative_boosted'] = True

            # Numeric density boost for quantitative questions
            if any(term in query_lower for term in ['berapa', 'jumlah', 'minimal', 'maksimal', 'durasi', 'skor']):
                for doc in reranked_docs:
                    content_lower = doc.get('content', doc.get('text', '')).lower()
                    if re.search(r'\d+', content_lower):
                        # Stronger boost if doc hits numbers and specific quantitative words
                        hits = sum(1 for term in ['minimal', 'kali', 'bulan', 'hari', 'skor', 'ipk', 'sks'] if term in content_lower and term in query_lower)
                        if hits > 0:
                            doc['cross_encoder_score'] += (hits * 3.0)
                        elif any(term in content_lower for term in ['minimal', 'kali', 'bulan', 'hari', 'skor', 'ipk', 'sks']):
                            doc['cross_encoder_score'] += 1.0

            # Semantic Alias Boost for Edge Cases (e.g., Form Yudisium)
            # The actual list of documents for Form Yudisium is usually split across chunks that don't contain the word "yudisium".
            if 'form yudisium' in query_lower or 'berkas yudisium' in query_lower:
                for doc in reranked_docs:
                    content_lower = doc.get('content', doc.get('text', '')).lower()
                    # Check for signature items in the "Contoh Berkas" list
                    is_yudisium_list = (
                        'tanda terima penyerahan tugas akhir dari tata usaha' in content_lower or
                        'bukti transfer sumbangan buku alumni' in content_lower or
                        ('verifikasi data mahasiswa' in content_lower and 'sudah sesuai' in content_lower) or
                        ('berkas haki' in content_lower and 'lampirkan ktp' in content_lower)
                    )
                    if is_yudisium_list:
                        doc['cross_encoder_score'] += 20.0
                        if 'metadata' not in doc:
                            doc['metadata'] = {}
                        doc['metadata']['semantic_alias_boosted'] = True

            # Sort by cross-encoder score descending
            reranked_docs.sort(key=lambda x: x['cross_encoder_score'], reverse=True)

            # Update ranks
            for rank, doc in enumerate(reranked_docs, 1):
                doc['rerank_rank'] = rank

            # Limit to top_k if specified
            if top_k is not None and len(reranked_docs) > top_k:
                reranked_docs = reranked_docs[:top_k]

            # Check if all scores are exactly 0.0 (indicates prediction failure)
            all_zeros = all(doc['cross_encoder_score'] == 0.0 for doc in reranked_docs)
            
            # Update main score with hybrid tie-breaker and retrieval-stage boosts
            # We incorporate the boost_factor from UnifiedIndexManager to ensure
            # that domain isolation penalties (like KILL:7_DAYS) are preserved.
            scored_docs = []
            for doc in reranked_docs:
                metadata = doc.get('metadata', {}) or {}
                boost_factor = metadata.get('boost_factor', 1.0)
                
                # HARD KILL: If retrieval stage explicitly killed this chunk (factor < 0.1),
                # drop it entirely to prevent it from leaking into the LLM context.
                if boost_factor < 0.1:
                    continue
                
                if all_zeros:
                    doc['score'] = doc.get('original_score', 0.0)
                else:
                    # Apply the boost/penalty factor from retrieval stage to the CE score
                    base_ce_score = doc['cross_encoder_score']
                    effective_ce = base_ce_score + 10.0 # Shift to positive range
                    
                    orig_signal = doc.get('original_score', 0.0) / 10000.0
                    doc['score'] = (effective_ce * boost_factor) + orig_signal
                
                scored_docs.append(doc)
            
            reranked_docs = scored_docs
            
            # If all zeros, restore original order
            if all_zeros:
                reranked_docs.sort(key=lambda x: x.get('original_score', 0.0), reverse=True)
            else:
                # Re-sort after hybrid tie-breaker and boost application
                reranked_docs.sort(key=lambda x: x['score'], reverse=True)

            latency = time.time() - start_time
            
            # Log top 3 results for debugging
            self.logger.info(f"[CrossEncoderReranker] Top 3 reranked docs:")
            for i, doc in enumerate(reranked_docs[:3]):
                self.logger.info(f"  {i+1}. ID: {doc.get('metadata', {}).get('chunk_id')} | CE Score: {doc.get('cross_encoder_score', 0.0):.4f} | Orig Score: {doc.get('original_score', 0.0):.4f}")

            self.logger.info(f"[CrossEncoderReranker] Reranking completed in {latency:.3f}s. Failed: {all_zeros}")

            return reranked_docs

        except Exception as e:
            latency = time.time() - start_time
            if self._is_pipeline_logger:
                self.logger.log_error("CrossEncoderReranker", e, {"query": query, "latency": latency})
            else:
                self.logger.error(f"[CrossEncoderReranker] Reranking failed: {e}")
            # Return original documents if reranking fails
            return documents

    def _predict_scores_batched(
        self,
        query_doc_pairs: List[tuple]
    ) -> np.ndarray:
        """
        Predict scores in batches to avoid memory issues.

        Args:
            query_doc_pairs: List of (query, document) tuples

        Returns:
            Array of scores
        """
        batch_size = self.config.batch_size
        all_scores = []

        for i in range(0, len(query_doc_pairs), batch_size):
            batch_pairs = query_doc_pairs[i:i + batch_size]

            try:
                batch_scores = self.model.predict(
                    batch_pairs,
                    show_progress_bar=False,
                    batch_size=batch_size
                )
                all_scores.extend(batch_scores)

            except Exception as e:
                self.logger.error(f"Failed to process batch {i//batch_size}: {e}")
                # Use default scores for this batch
                batch_scores = [0.0] * len(batch_pairs)
                all_scores.extend(batch_scores)

            # Cleanup GPU memory if needed
            if i % (batch_size * 2) == 0:
                safe_gpu_memory_cleanup()

        return np.array(all_scores)

    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank documents for multiple queries.

        Args:
            queries: List of search queries
            documents_list: List of document lists for each query
            top_k: Number of top documents to return per query

        Returns:
            List of reranked document lists
        """
        if len(queries) != len(documents_list):
            raise ValueError("Number of queries must match number of document lists")

        self.logger.info(f"[CrossEncoderReranker] Reranking {len(queries)} queries in batch")

        results = []
        total_start_time = time.time()

        for i, (query, documents) in enumerate(zip(queries, documents_list)):
            try:
                reranked_docs = self.rerank(query, documents, top_k)
                results.append(reranked_docs)

                # Log progress
                if (i + 1) % 5 == 0:
                    self.logger.info(f"[CrossEncoderReranker] Reranked {i + 1}/{len(queries)} queries")

            except Exception as e:
                self.logger.error(f"[CrossEncoderReranker] Failed to rerank query {i + 1}: {e}")
                results.append(documents)  # Return original if reranking fails

        total_latency = time.time() - total_start_time
        avg_latency = total_latency / len(queries) if queries else 0

        self.logger.info(f"[CrossEncoderReranker] Batch reranking completed in {total_latency:.3f}s (avg: {avg_latency:.3f}s per query)")

        return results

    def get_reranking_stats(
        self,
        original_docs: List[Dict[str, Any]],
        reranked_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about the reranking process.

        Args:
            original_docs: Documents before reranking
            reranked_docs: Documents after reranking

        Returns:
            Dictionary with reranking statistics
        """
        if not original_docs or not reranked_docs:
            return {"error": "Empty document lists"}

        # Calculate score improvements
        improvements = [
            doc.get('rerank_improvement', 0)
            for doc in reranked_docs
            if 'rerank_improvement' in doc
        ]

        # Calculate rank changes
        original_scores = {i: doc.get('score', 0) for i, doc in enumerate(original_docs)}
        reranked_ranks = {doc.get('text', ''): doc.get('rerank_rank', 0) for doc in reranked_docs}

        rank_changes = []
        for i, doc in enumerate(original_docs):
            original_rank = i + 1
            reranked_rank = reranked_ranks.get(doc.get('text', ''), original_rank)
            change = original_rank - reranked_rank
            rank_changes.append(change)

        # Calculate average position change
        avg_position_change = sum(abs(change) for change in rank_changes) / len(rank_changes)

        return {
            "num_documents": len(original_docs),
            "avg_score_improvement": np.mean(improvements) if improvements else 0,
            "max_score_improvement": max(improvements) if improvements else 0,
            "min_score_improvement": min(improvements) if improvements else 0,
            "avg_position_change": avg_position_change,
            "max_position_improvement": max(rank_changes) if rank_changes else 0,
            "max_position_decline": min(rank_changes) if rank_changes else 0,
            "documents_improved": sum(1 for imp in improvements if imp > 0),
            "documents_declined": sum(1 for imp in improvements if imp < 0),
            "documents_unchanged": sum(1 for imp in improvements if abs(imp) < 0.001)
        }

    def explain_reranking(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide explanation for reranking of a specific document.

        Args:
            query: Search query
            document: Reranked document

        Returns:
            Explanation dictionary
        """
        return {
            "query": query,
            "document_id": document.get('metadata', {}).get('global_chunk_id', ''),
            "original_score": document.get('original_score', 0.0),
            "cross_encoder_score": document.get('cross_encoder_score', 0.0),
            "score_improvement": document.get('rerank_improvement', 0.0),
            "original_rank": document.get('rank', 0),
            "rerank_rank": document.get('rerank_rank', 0),
            "rank_change": document.get('rank', 0) - document.get('rerank_rank', 0),
            "reranker_model": self.config.model_name,
            "explanation": "Document reranked using cross-encoder model that scores query-document relevance directly"
        }

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information for the reranker.

        Returns:
            Dictionary with configuration details
        """
        return {
            "reranker_type": "cross_encoder",
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "capabilities": [
                "Query-document relevance scoring",
                "Batch processing",
                "GPU acceleration",
                "Score improvement tracking",
                "Rank change analysis"
            ]
        }
