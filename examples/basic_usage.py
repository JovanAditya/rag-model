#!/usr/bin/env python3
"""
Basic Usage Example - Academic RAG

Basic usage example for Academic RAG question-answering system.
Questions are based on actual knowledge base content (UMB academic guides).
Shows detailed per-stage output for pipeline transparency.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_model import AcademicRAG
from rag_model.core.config import RAGConfig


def print_header(title: str):
    """Print a simple section header."""
    print(f"\n=== {title} ===")


def print_doc_list(header: str, docs: list, score_key: str = "score"):
    """Print a linear list of documents with preview text."""
    print(f"\n{header}")
    for i, doc in enumerate(docs[:5]):
        score = doc.get(score_key, doc.get('score', 0.0))
        doc_id = doc.get('id') or doc.get('chunk_id') or doc.get('metadata', {}).get('chunk_id') or '?'
        content = doc.get('text', doc.get('content', '')).strip().replace('\n', ' ')
        preview = content[:150] + ("..." if len(content) > 150 else "")
        print(f"[Dokumen {i+1}] ID: {doc_id} | Skor: {score:.4f}")
        print(f"Preview: \"{preview}\"")
        print("-" * 50)


def run_verbose_query(query: str):
    """Run a single query with verbose per-stage output."""

    print(f"\nACADEMIC RAG - VERBOSE QUERY PIPELINE")
    print(f"Query: \"{query}\"")

    # Initialize RAG
    print_header("INISIALISASI MODEL")
    init_start = time.time()
    rag = AcademicRAG(research_mode=True, response_format="full")
    rag._initialize_components()
    init_time = time.time() - init_start
    print(f"Model berhasil diinisialisasi ({init_time:.2f}s)")
    print(f"Pipeline: {rag.config.retrieval.pipeline_type}")
    print(f"LLM: {rag.config.llm.model_type}")
    print(f"Embedding: {rag.config.embedding.model_name}")
    print(f"Reranking: {'Ya' if rag.config.retrieval.use_reranking else 'Tidak'}")

    # === TAHAP 4: HYBRID SEARCH ===
    print_header("HYBRID SEARCH")

    retrieval_start = time.time()

    if rag._unified_index_manager:
        manager = rag._unified_index_manager

        if manager.vector_store is None or manager.bm25_index is None:
            manager.initialize_indexes()

        # BM25 results
        bm25_start = time.time()
        try:
            bm25_results = manager.bm25_index.search(query, k=5)
            bm25_time = time.time() - bm25_start
            print_doc_list(f"Hasil BM25 (Leksikal) - {bm25_time:.3f}s:", bm25_results, "score")
        except Exception as e:
            bm25_time = time.time() - bm25_start
            print(f"⚠️ BM25 search error: {e}")
            bm25_results = []

        # Vector results
        vector_start = time.time()
        try:
            vector_results = manager.vector_store.similarity_search(query, k=5)
            vector_time = time.time() - vector_start
            print_doc_list(f"\nHasil Vector Search (Semantik) - {vector_time:.3f}s:", vector_results, "score")
        except Exception as e:
            vector_time = time.time() - vector_start
            print(f"⚠️ Vector search error: {e}")
            vector_results = []

        # Unified (RRF) results
        rrf_start = time.time()
        rrf_results = manager.search_unified(
            query=query, k=5,
            vector_weight=rag.config.retrieval.vector_weight,
            bm25_weight=rag.config.retrieval.bm25_weight,
            strategy="rrf"
        )
        rrf_time = time.time() - rrf_start
        fused_docs = rrf_results.get("results", [])
        print_doc_list(f"\nHasil RRF Fusion (Gabungan) - {rrf_time:.3f}s:", fused_docs, "score")

    retrieval_time = time.time() - retrieval_start

    # === TAHAP 5: RERANKING ===
    print_header("RERANKING (Cross-Encoder)")

    if rag.config.retrieval.use_reranking and rag._reranker:
        rerank_candidates = rag._unified_index_manager.search_unified(
            query=query, k=rag.config.retrieval.rerank_k,
            vector_weight=rag.config.retrieval.vector_weight,
            bm25_weight=rag.config.retrieval.bm25_weight,
            strategy="rrf"
        )
        candidate_docs = rerank_candidates.get("results", [])

        print(f"Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"Kandidat masuk: {len(candidate_docs)} dokumen")

        print(f"\nUrutan SEBELUM Reranking:")
        for i, doc in enumerate(candidate_docs[:5]):
            doc_id = str(doc.get('id', doc.get('chunk_id', '?')))
            print(f"[Kandidat {i+1}] Skor RRF: {doc.get('score', 0.0):.4f} | ID: {doc_id}")

        # Rerank
        rerank_start = time.time()
        reranked_docs = rag._reranker.rerank(query, candidate_docs, top_k=5)
        rerank_time = time.time() - rerank_start

        print(f"\nUrutan SESUDAH Reranking ({rerank_time:.3f}s):")
        for i, doc in enumerate(reranked_docs[:5]):
            doc_id = str(doc.get('id', doc.get('chunk_id', '?')))
            score = doc.get('cross_encoder_score', doc.get('score', 0.0))
            content = doc.get('text', doc.get('content', '')).strip().replace('\n', ' ')
            preview = content[:200] + ("..." if len(content) > 200 else "")
            print(f"\n[Peringkat {i+1}] Skor CE: {score:.4f} | ID: {doc_id}")
            print(f"Preview: \"{preview}\"")
            print("-" * 50)

        final_docs = reranked_docs
    else:
        print("(Reranking dinonaktifkan, menggunakan hasil RRF)")
        final_docs = fused_docs
        rerank_time = 0.0

    # === TAHAP 6: CONTEXT BUILDING ===
    print_header("CONTEXT BUILDING")

    context_start = time.time()
    context_data = rag._context_builder.build_context(final_docs, query)
    context_time = time.time() - context_start
    context_text = context_data.get("context", "")

    print(f"Dokumen digunakan: {len(final_docs[:5])}")
    print(f"Panjang total konteks: {len(context_text):,} karakter")
    print(f"Waktu: {context_time:.3f}s")
    print(f"\nKonteks yang Dikirim ke LLM:\n")
    print(context_text.strip())
    print("-" * 50)

    # === TAHAP 7: GENERASI JAWABAN ===
    print_header("GENERATE JAWABAN (LLM)")

    print(f"Model LLM: {rag.config.llm.model_type}")
    print(f"Mengirim query + konteks ke LLM...\n")

    gen_start = time.time()
    answer = rag._generate_answer(query, context_text)
    gen_time = time.time() - gen_start

    total_time = init_time + retrieval_time + rerank_time + context_time + gen_time

    print(f"Jawaban LLM ({gen_time:.2f}s):\n")
    print(answer.strip())
    print("-" * 50)

    # === RINGKASAN WAKTU ===
    print_header("RINGKASAN WAKTU EKSEKUSI")
    print(f"Inisialisasi: {init_time:.2f}s")
    print(f"Retrieval (Hybrid Search): {retrieval_time:.2f}s")
    print(f"Reranking (Cross-Encoder): {rerank_time:.2f}s")
    print(f"Context Building: {context_time:.3f}s")
    print(f"Generasi LLM: {gen_time:.2f}s")
    print("-" * 30)
    print(f"TOTAL WAKTU: {total_time:.2f}s")

    print(f"\nPipeline selesai.\n")


def main():
    """Basic usage example with real knowledge base questions."""
    # Default query — can be overridden via command line
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Apa syarat untuk mengikuti Kerja Praktek (KP)?"

    run_verbose_query(query)


if __name__ == "__main__":
    main()
