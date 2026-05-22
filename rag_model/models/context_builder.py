"""Context builder for Advanced RAG Pipeline."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..utils.logging import PipelineLogger
from ..utils.helpers import truncate_context


@dataclass
class ContextConfig:
    """Configuration for context building."""
    max_context_length: int = 15000  # Maximum characters in context
    max_documents: int = 5  # Maximum number of documents to include
    include_metadata: bool = True  # Whether to include document metadata
    include_scores: bool = True  # Whether to include relevance scores
    separator: str = "\n\n---\n\n"  # Separator between documents


class ContextBuilder:
    """
    Build and manage context for LLM generation from retrieved documents.

    Handles context construction, truncation, and formatting to provide
    optimal input for language model generation.
    """

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize context builder.

        Args:
            config: Context building configuration
            logger: Optional logger instance
        """
        self.config = config or ContextConfig()
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"ContextBuilder initialized: max_length={self.config.max_context_length}, max_docs={self.config.max_documents}")

    def build_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context from retrieved documents.

        Args:
            retrieved_docs: List of retrieved documents with scores and metadata
            query: Optional query for reference

        Returns:
            Dictionary with built context and metadata
        """
        if not retrieved_docs:
            self.logger.warning("No documents provided for context building")
            return {
                "context": "",
                "documents_used": 0,
                "context_length": 0,
                "truncated": False
            }

        self.logger.info(f"[ContextBuilder] Building context from {len(retrieved_docs)} documents")

        # Sort documents by relevance score (highest first)
        # Handle both dictionary and SourceDocument objects
        if retrieved_docs and hasattr(retrieved_docs[0], 'score'):
            # SourceDocument objects
            sorted_docs = sorted(
                retrieved_docs,
                key=lambda x: x.score,
                reverse=True
            )
        else:
            # Dictionary objects
            sorted_docs = sorted(
                retrieved_docs,
                key=lambda x: x.get('score', 0),
                reverse=True
            )

        # Domain-aware prioritization (TA vs KP)
        if query:
            query_lower = query.lower()
            is_ta_query = any(term in query_lower for term in ['tugas akhir', 'skripsi', 'ta ', ' ta', '(ta)'])
            is_kp_query = any(term in query_lower for term in ['kerja praktek', 'magang', ' kp', 'kp ', '(kp)'])
            is_skpi_query = any(term in query_lower for term in ['skpi', 'sertifikat', 'prestasi', 'ijazah', 'yudisium'])
            is_mpti_query = any(term in query_lower for term in ['mpti', 'metodologi penelitian', 'proposal', 'sempro'])
            is_ult_query = any(term in query_lower for term in ['ult', 'unit layanan terpadu', 'akun', 'password', 'login', 'bantuan'])
            is_pasca_sidang_query = any(term in query_lower for term in ['pasca sidang', 'setelah sidang', 'revisi ta', '14 hari'])
            is_count_query = any(term in query_lower for term in ['berapa', 'jumlah', 'minimal', 'maksimal', 'limit', 'total'])
            
            if is_ta_query or is_kp_query or is_skpi_query or is_mpti_query or is_ult_query or is_pasca_sidang_query or is_count_query:
                import re
                def domain_priority_score(doc):
                    metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
                    source = metadata.get('original_filename', '').lower()
                    content = (doc.text if hasattr(doc, 'text') else doc.get('text', '')).lower()
                    
                    # Dynamic domain detection
                    def is_domain_match(keywords, text, min_matches=2):
                        return sum(text.count(kw) for kw in keywords) >= min_matches

                    ta_kws = ['tugas akhir', 'skripsi', 'sidang akhir', 'pengesahan ta']
                    kp_kws = ['kerja praktek', 'magang', 'kp ']
                    skpi_kws = ['skpi', 'sertifikat', 'prestasi', 'yudisium']
                    mpti_kws = ['mpti', 'metodologi penelitian', 'proposal', 'sempro', 'apa style']
                    pasca_sidang_kws = ['pasca sidang', 'revisi', '14 hari', 'setelah sidang']
                    ult_kws = ['ult', 'layanan terpadu', 'jam operasional']
                    fasilkom_kws = ['fasilkom', 'teknik informatika', 'sistem informasi', 'ilmu komputer']
                    
                    score = 0
                    # Primary boosts
                    if is_ta_query and (any(kw in source for kw in ta_kws) or is_domain_match(ta_kws, content)):
                        score += 5
                    if is_pasca_sidang_query and (any(kw in source for kw in pasca_sidang_kws) or is_domain_match(pasca_sidang_kws, content)):
                        score += 6
                    if is_kp_query and (any(kw in source for kw in kp_kws) or is_domain_match(kp_kws, content)):
                        score += 4
                    if is_mpti_query and (any(kw in source for kw in mpti_kws) or is_domain_match(mpti_kws, content)):
                        score += 4
                    if is_ult_query and (any(kw in source for kw in ult_kws) or is_domain_match(ult_kws, content)):
                        score += 4
                    
                    # Penalties for mismatched domains
                    if (is_ta_query or is_pasca_sidang_query) and (is_domain_match(mpti_kws, content) or is_domain_match(kp_kws, content)):
                        score -= 5
                    
                    if any(kw in source for kw in fasilkom_kws):
                        score += 2
                        
                    if is_count_query and re.search(r'\d+', content):
                        score += 1
                        
                    return score

                # Stable sort to keep original score ranking within domain priority
                sorted_docs = sorted(
                    sorted_docs,
                    key=domain_priority_score,
                    reverse=True
                )

        # Limit number of documents
        docs_to_use = sorted_docs[:self.config.max_documents]

        # Format documents
        formatted_docs = []
        for i, doc in enumerate(docs_to_use, 1):
            formatted_doc = self._format_document(doc, i)
            formatted_docs.append(formatted_doc)

        # Join documents with separator
        full_context = self.config.separator.join(formatted_docs)

        # Truncate if necessary
        truncated = False
        if len(full_context) > self.config.max_context_length:
            self.logger.info(f"[ContextBuilder] Truncating context from {len(full_context)} to {self.config.max_context_length} characters")
            full_context = truncate_context([full_context], self.config.max_context_length)
            truncated = True

        # Prepare metadata
        context_metadata = {
            "query": query,
            "documents_used": len(docs_to_use),
            "total_documents_retrieved": len(retrieved_docs),
            "context_length": len(full_context),
            "truncated": truncated,
            "average_score": sum(doc.score if hasattr(doc, 'score') else doc.get('score', 0) for doc in docs_to_use) / len(docs_to_use),
            "retrieval_methods": list(set(doc.get('retrieval_method', 'unknown') if hasattr(doc, 'get') else getattr(doc, 'retrieval_method', 'unknown') for doc in docs_to_use)),
            "score_range": {
                "min": min(doc.score if hasattr(doc, 'score') else doc.get('score', 0) for doc in docs_to_use),
                "max": max(doc.score if hasattr(doc, 'score') else doc.get('score', 0) for doc in docs_to_use)
            }
        }

        self.logger.info(f"[ContextBuilder] Context built: {context_metadata['documents_used']} docs, {context_metadata['context_length']} chars")

        return {
            "context": full_context,
            "metadata": context_metadata
        }

    def _format_document(self, doc: Dict[str, Any], doc_num: int) -> str:
        """
        Format a single document for context.

        Args:
            doc: Document dictionary
            doc_num: Document number for reference

        Returns:
            Formatted document string
        """
        # Handle both dictionary and SourceDocument objects
        if hasattr(doc, 'text'):
            # SourceDocument object
            text = doc.text.strip()
            title = doc.title if doc.title else ''
            metadata = doc.metadata if doc.metadata else {}
            score = doc.score
        else:
            # Dictionary object
            text = doc.get('text', '').strip()
            metadata = doc.get('metadata', {})
            title = metadata.get('title', '')
            score = doc.get('score', 0)

        if not text:
            return ""

        # Use generic document label to prevent LLM from referencing
        # internal filenames or metadata in its answers
        formatted_parts = [f"[Dokumen Referensi {doc_num}]"]

        # Add page info if available (inline with doc label)
        page_numbers = metadata.get('page_numbers', []) or metadata.get('page', None)
        if page_numbers:
            if isinstance(page_numbers, list) and page_numbers:
                formatted_parts[0] += f" (Halaman {', '.join(map(str, page_numbers))})"
            elif isinstance(page_numbers, int):
                formatted_parts[0] += f" (Halaman {page_numbers})"

        # NOTE: Relevance scores are intentionally NOT included in the
        # LLM-facing context. They are internal metadata used for ranking
        # and should not leak into the model's answers. Scores are still
        # available in the pipeline's metadata/sources output.

        # Add the actual text
        formatted_parts.append(f"\n{text}")

        return "\n".join(formatted_parts)

    @staticmethod
    def _is_hashed_filename(filename: str) -> bool:
        """
        Check if a filename is a system-generated hash (not human-readable).
        
        Hashed filenames from Laravel upload follow the pattern:
        <hex_uniqid>_<timestamp>.<ext>  (e.g., 69f37b44dae49_1777564484.pdf)
        
        These should NOT be shown to the LLM as they are internal identifiers.
        """
        import re
        # Pattern: hex characters (10+) _ digits . extension
        return bool(re.match(r'^[0-9a-f]{10,}_\d+\.\w+$', filename, re.IGNORECASE))

    def _get_document_name(self, metadata: Dict[str, Any], title: str, doc_num: int) -> str:
        """
        Extract best document name from metadata.
        
        Returns a generic label ("Dokumen Referensi N") if the filename is a
        system-generated hash, since the pipeline does not have access to the
        Laravel database where the human-readable original filename is stored.
        
        Priority:
        1. original_filename — only if NOT a hashed/system filename
        2. filename — only if NOT a hashed/system filename
        3. source basename — only if NOT a hashed/system filename
        4. title
        5. Fallback to generic numbering
        
        Args:
            metadata: Document metadata dictionary
            title: Document title if available
            doc_num: Fallback document number
            
        Returns:
            Best available document name
        """
        import os
        
        # Priority 1: original_filename (skip if hashed)
        original_filename = metadata.get('original_filename', '')
        if original_filename and not self._is_hashed_filename(original_filename):
            return original_filename
        
        # Priority 2: filename (skip if hashed)
        filename = metadata.get('filename', '')
        if filename and not self._is_hashed_filename(filename):
            return filename
        
        # Priority 3: source basename (skip if hashed)
        source = metadata.get('source', '')
        if source:
            basename = os.path.basename(source)
            if basename and not self._is_hashed_filename(basename):
                return basename
        
        # Priority 4: title
        if title:
            return title
        
        # Fallback: generic label (safe, no internal metadata leaked)
        return f"Dokumen Referensi {doc_num}"

    def build_comparison_context(
        self,
        baseline_docs: List[Dict[str, Any]],
        advanced_docs: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build comparison context showing results from both baseline and advanced retrieval.

        Args:
            baseline_docs: Documents from baseline retrieval
            advanced_docs: Documents from advanced retrieval
            query: Optional query for reference

        Returns:
            Dictionary with comparison context
        """
        self.logger.info("[ContextBuilder] Building comparison context")

        # Build individual contexts
        baseline_context_data = self.build_context(baseline_docs, query)
        advanced_context_data = self.build_context(advanced_docs, query)

        # Combine into comparison format
        comparison_context = f"""Konteks Baseline (Pencarian Vektor Saja):
{baseline_context_data['context']}

Konteks Lanjutan (Pencarian Hibrid + Reranking):
{advanced_context_data['context']}"""

        # Create comparison metadata
        comparison_metadata = {
            "query": query,
            "baseline": baseline_context_data['metadata'],
            "advanced": advanced_context_data['metadata'],
            "context_type": "comparison",
            "total_context_length": len(comparison_context)
        }

        return {
            "context": comparison_context,
            "metadata": comparison_metadata
        }

    def extract_key_information(self, context: str, max_sentences: int = 3) -> str:
        """
        Extract key information from context for quick summarization.

        Args:
            context: Full context text
            max_sentences: Maximum number of sentences to extract

        Returns:
            Key information summary
        """
        if not context:
            return ""

        # Split into sentences (simple approach)
        sentences = []
        for part in context.split('.'):
            part = part.strip()
            if part and len(part) > 20:  # Filter out very short parts
                sentences.append(part + '.')

        if not sentences:
            return context[:200] + "..." if len(context) > 200 else context

        # Take first few sentences
        key_sentences = sentences[:max_sentences]
        return " ".join(key_sentences)

    def get_context_statistics(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed statistics about the built context.

        Args:
            context_data: Context data from build_context

        Returns:
            Dictionary with context statistics
        """
        context = context_data.get('context', '')
        metadata = context_data.get('metadata', {})

        # Basic statistics
        char_count = len(context)
        word_count = len(context.split())
        sentence_count = context.count('.') + context.count('!') + context.count('?')

        # Document overlap analysis (if available)
        doc_overlaps = []
        if 'documents_used' in metadata:
            doc_overlaps.append({
                "documents_in_context": metadata['documents_used'],
                "total_retrieved": metadata.get('total_documents_retrieved', 0),
                "context_ratio": metadata['documents_used'] / metadata.get('total_documents_retrieved', 1)
            })

        return {
            "text_statistics": {
                "character_count": char_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "average_sentence_length": word_count / max(sentence_count, 1),
                "estimated_tokens": word_count * 1.3  # Rough estimate
            },
            "context_efficiency": {
                "truncated": metadata.get('truncated', False),
                "max_length_utilization": char_count / self.config.max_context_length,
                "documents_per_character": metadata.get('documents_used', 0) / max(char_count, 1)
            },
            "quality_indicators": {
                "average_relevance_score": metadata.get('average_score', 0),
                "score_variance": self._calculate_score_variance(context_data),
                "retrieval_method_diversity": len(metadata.get('retrieval_methods', []))
            },
            "overlap_analysis": doc_overlaps
        }

    def _calculate_score_variance(self, context_data: Dict[str, Any]) -> float:
        """Calculate score variance from context data if available."""
        # This is a simplified version - could be expanded based on actual data structure
        metadata = context_data.get('metadata', {})
        score_range = metadata.get('score_range', {})

        if isinstance(score_range, dict) and 'min' in score_range and 'max' in score_range:
            return (score_range['max'] - score_range['min']) / 2  # Simple variance estimate

        return 0.0

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information for the context builder.

        Returns:
            Dictionary with configuration details
        """
        return {
            "max_context_length": self.config.max_context_length,
            "max_documents": self.config.max_documents,
            "include_metadata": self.config.include_metadata,
            "include_scores": self.config.include_scores,
            "separator": self.config.separator,
            "capabilities": [
                "Document ranking",
                "Context truncation",
                "Metadata preservation",
                "Comparison context building",
                "Key information extraction"
            ]
        }
