"""Python BM25 Index Implementation for AcademicRAG.

Lightweight BM25 implementation using scikit-learn.
Optimized for <1000 documents with Indonesian text processing.
"""

import logging
import math
import time
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("❌ scikit-learn not available. Install with: pip install scikit-learn")
    raise

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Lightweight BM25 implementation using scikit-learn.

    Optimized for academic Indonesian text with <1000 documents.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, ngram_range: Tuple[int, int] = (1, 2), cache_dir: str = "./cache"):
        """
        Initialize BM25 index.

        Args:
            k1: Controls term frequency saturation (typical: 1.2-2.0)
            b: Controls document length normalization (typical: 0.75)
            ngram_range: Range of n-grams for tokenization
            cache_dir: Directory to store cache files
        """
        self.k1 = k1
        self.b = b
        self.ngram_range = ngram_range
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Indonesian stop words (minimal set for academic text)
        self.indonesian_stop_words = {
            'dan', 'di', 'ke', 'pada', 'untuk', 'dengan', 'dari', 'yang',
            'adalah', 'sebuah', 'atau', 'dalam', 'tersebut', 'ini', 'itu',
            'yaitu', 'tersebut', 'adalah', 'yaitu', 'oleh', 'sebagai', 'bagi',
            'dapat', 'akan', 'telah', 'sudah', 'banyak', 'terdapat', 'tersebut',
            'juga', 'yaitu', 'yakni', 'adalah', 'melalui', 'dengan', 'menggunakan'
        }

        # Initialize TF-IDF vectorizer with Indonesian settings
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(self.indonesian_stop_words),
            ngram_range=self.ngram_range,
            min_df=1,  # Include terms that appear in at least 1 document
            max_df=0.95,  # Exclude terms that appear in >95% of documents
            sublinear_tf=True  # Apply sublinear TF scaling
        )

        # Index data
        self.documents = []
        self.doc_freqs = None  # Document frequencies
        self.idf = None        # Inverse document frequencies
        self.doc_lengths = None  # Document lengths
        self.avg_doc_length = None
        self.vocabulary = None
        self.term_freq_matrix = None

        logger.info(f"BM25Index initialized (k1={k1}, b={b})")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index document chunks for BM25 search.

        Args:
            documents: List of document chunks with text and metadata
        """
        start_time = time.time()
        logger.info(f"Indexing {len(documents)} documents for BM25 search")

        # Extract text content
        texts = []
        self.documents = []

        for doc in documents:
            text = doc.get('text', '').strip()
            if text:
                texts.append(text)
                self.documents.append(doc)

        if not texts:
            logger.warning("No valid texts to index")
            return

        # Fit TF-IDF vectorizer and transform
        logger.info("Building TF-IDF matrix...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Calculate document statistics
        self.doc_lengths = np.array([len(text.split()) for text in texts])
        self.avg_doc_length = np.mean(self.doc_lengths)

        # Get vocabulary and document frequencies
        self.vocabulary = self.vectorizer.vocabulary_
        self.term_freq_matrix = tfidf_matrix.toarray()

        # Calculate document frequencies (how many docs contain each term)
        self.doc_freqs = np.sum((self.term_freq_matrix > 0), axis=0)

        # Calculate IDF (Inverse Document Frequency)
        n_docs = len(texts)
        self.idf = np.log((n_docs + 1) / (self.doc_freqs + 1)) + 1

        index_time = time.time() - start_time
        logger.info(f"BM25 indexing completed in {index_time:.2f}s")
        logger.info(f"Vocabulary size: {len(self.vocabulary)}")
        logger.info(f"Average doc length: {self.avg_doc_length:.1f} words")

    def _calculate_bm25_score(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document against query terms.

        Args:
            query_terms: List of query terms
            doc_idx: Index of document in corpus

        Returns:
            BM25 score
        """
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]

        for term in query_terms:
            if term in self.vocabulary:
                term_idx = self.vocabulary[term]

                # Term frequency in document
                tf = self.term_freq_matrix[doc_idx, term_idx]

                # Document frequency
                df = self.doc_freqs[term_idx]
                idf = self.idf[term_idx]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))

                term_score = idf * (numerator / denominator)
                score += term_score

        return score

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents using BM25 scoring.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of scored documents
        """
        if not self.documents:
            logger.warning("No indexed documents available for search")
            return []

        # Process query
        start_time = time.time()

        # Tokenize and filter query terms
        query_processed = self.vectorizer.build_tokenizer()(query.lower())
        query_terms = [term for term in query_processed
                      if term not in self.indonesian_stop_words
                      and term in self.vocabulary]

        if not query_terms:
            logger.warning("No valid query terms found after filtering")
            return []

        # Calculate BM25 scores for all documents
        scores = []
        for doc_idx in range(len(self.documents)):
            score = self._calculate_bm25_score(query_terms, doc_idx)
            scores.append((doc_idx, score))

        # Sort by score (descending) and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:k]

        # Prepare results
        results = []
        for doc_idx, score in top_scores:
            doc = self.documents[doc_idx].copy()
            doc['bm25_score'] = float(score)
            doc['rank'] = len(results) + 1
            results.append(doc)

        search_time = time.time() - start_time
        logger.debug(f"BM25 search completed in {search_time:.3f}s for {len(query_terms)} query terms")

        return results

    def get_document_count(self) -> int:
        """Get total number of indexed documents."""
        return len(self.documents)

    def get_vocabulary_size(self) -> int:
        """Get size of vocabulary."""
        return len(self.vocabulary) if self.vocabulary else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "documents_count": len(self.documents),
            "vocabulary_size": len(self.vocabulary) if self.vocabulary else 0,
            "avg_doc_length": float(self.avg_doc_length) if self.avg_doc_length else 0,
            "avg_term_frequency": float(np.mean(self.doc_freqs)) if self.doc_freqs is not None and len(self.doc_freqs) > 0 else 0,
            "k1": self.k1,
            "b": self.b,
            "ngram_range": self.ngram_range,
            "stop_words_count": len(self.indonesian_stop_words)
        }

    def save_cache(self, cache_name: str = "bm25_index") -> None:
        """
        Save BM25 index to cache file.

        Args:
            cache_name: Name for cache file (without extension)
        """
        import pickle
        import gzip

        cache_file = self.cache_dir / f"{cache_name}.pkl.gz"

        try:
            cache_data = {
                'documents': self.documents,
                'doc_freqs': self.doc_freqs,
                'idf': self.idf,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'vocabulary': self.vocabulary,
                'term_freq_matrix': self.term_freq_matrix,
                'vectorizer': self.vectorizer,
                'k1': self.k1,
                'b': self.b,
                'ngram_range': self.ngram_range,
                'created_at': time.time()
            }

            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"✅ BM25 index saved to cache: {cache_file}")

        except Exception as e:
            logger.error(f"Failed to save BM25 cache: {e}")

    def load_cache(self, cache_name: str = "bm25_index") -> bool:
        """
        Load BM25 index from cache file.

        Args:
            cache_name: Name for cache file (without extension)

        Returns:
            True if cache loaded successfully, False otherwise
        """
        import pickle
        import gzip

        cache_file = self.cache_dir / f"{cache_name}.pkl.gz"

        if not cache_file.exists():
            logger.debug(f"BM25 cache file not found: {cache_file}")
            return False

        try:
            with gzip.open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Restore index data
            self.documents = cache_data['documents']
            self.doc_freqs = cache_data['doc_freqs']
            self.idf = cache_data['idf']
            self.doc_lengths = cache_data['doc_lengths']
            self.avg_doc_length = cache_data['avg_doc_length']
            self.vocabulary = cache_data['vocabulary']
            self.term_freq_matrix = cache_data['term_freq_matrix']
            self.vectorizer = cache_data['vectorizer']
            self.k1 = cache_data['k1']
            self.b = cache_data['b']
            self.ngram_range = cache_data['ngram_range']

            logger.info(f"✅ BM25 index loaded from cache: {cache_file}")
            logger.info(f"   Loaded {len(self.documents)} documents with {len(self.vocabulary)} vocabulary terms")

            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 cache: {e}")
            return False