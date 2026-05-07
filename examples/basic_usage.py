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


def print_header(title: str, char: str = "═"):
    """Print a formatted section header."""
    width = 64
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def print_box(title: str, rows: list):
    """Print a formatted box with key-value rows."""
    width = 58
    print(f"\n   ╔{'═'*width}╗")
    print(f"   ║ {title:<{width-2}s} ║")
    print(f"   ╠{'═'*width}╣")
    for key, value in rows:
        print(f"   ║ {key:<30s} : {str(value):<{width-34}s} ║")
    print(f"   ╚{'═'*width}╝")


def print_doc_table(header: str, docs: list, score_key: str = "score"):
    """Print a table of documents with scores."""
    print(f"\n   {header}")
    print(f"   {'No':>3s} │ {'Skor':>8s} │ {'Chunk ID':<25s} │ Konten (preview)")
    print(f"   {'───':>3s}─┼─{'────────':>8s}─┼─{'─'*25}─┼─{'─'*30}")
    for i, doc in enumerate(docs[:5]):
        score = doc.get(score_key, doc.get('score', 0.0))
        doc_id = doc.get('id', doc.get('chunk_id', '?'))
        if len(str(doc_id)) > 25:
            doc_id = str(doc_id)[-25:]
        content = doc.get('text', doc.get('content', ''))[:50].replace('\n', ' ')
        print(f"   {i+1:>3d} │ {score:>8.4f} │ {str(doc_id):<25s} │ {content}...")


def run_verbose_query(query: str):
    """Run a single query with verbose per-stage output."""

    print("\n" + "═" * 64)
    print("  🎓 ACADEMIC RAG — VERBOSE QUERY PIPELINE")
    print("═" * 64)
    print(f"\n  Query: \"{query}\"")

    # Initialize RAG
    print_header("INISIALISASI MODEL")
    init_start = time.time()
    rag = AcademicRAG(research_mode=True, response_format="full")
    rag._initialize_components()
    init_time = time.time() - init_start
    print(f"   ✅ Model berhasil diinisialisasi ({init_time:.2f}s)")
    print(f"   Pipeline : {rag.config.retrieval.pipeline_type}")
    print(f"   LLM      : {rag.config.llm.model_type}")
    print(f"   Embedding: {rag.config.embedding.model_name}")
    print(f"   Reranking: {'Ya' if rag.config.retrieval.use_reranking else 'Tidak'}")

    # === TAHAP 4: HYBRID SEARCH ===
    print_header("🔍 TAHAP 4: HYBRID SEARCH")

    retrieval_start = time.time()

    # Get results from unified index manager with details
    if rag._unified_index_manager:
        manager = rag._unified_index_manager

        # BM25 results
        bm25_start = time.time()
        try:
            bm25_results = manager._bm25_index.search(query, k=5)
            bm25_time = time.time() - bm25_start
            print_doc_table(f"Hasil BM25 (Leksikal) — {bm25_time:.3f}s:", bm25_results, "score")
        except Exception as e:
            bm25_time = time.time() - bm25_start
            print(f"   ⚠️  BM25 search error: {e}")
            bm25_results = []

        # Vector results
        vector_start = time.time()
        try:
            vector_results = manager._vector_store.search(query, k=5)
            vector_time = time.time() - vector_start
            print_doc_table(f"\nHasil Vector Search (Semantik) — {vector_time:.3f}s:", vector_results, "score")
        except Exception as e:
            vector_time = time.time() - vector_start
            print(f"   ⚠️  Vector search error: {e}")
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
        print_doc_table(f"\nHasil RRF Fusion (Gabungan) — {rrf_time:.3f}s:", fused_docs, "score")

    retrieval_time = time.time() - retrieval_start

    # === TAHAP 5: RERANKING ===
    print_header("🔄 TAHAP 5: RERANKING (Cross-Encoder)")

    if rag.config.retrieval.use_reranking and rag._reranker:
        # Get more candidates for reranking
        rerank_candidates = rag._unified_index_manager.search_unified(
            query=query, k=rag.config.retrieval.rerank_k,
            vector_weight=rag.config.retrieval.vector_weight,
            bm25_weight=rag.config.retrieval.bm25_weight,
            strategy="rrf"
        )
        candidate_docs = rerank_candidates.get("results", [])

        print(f"   Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"   Kandidat masuk: {len(candidate_docs)} dokumen")

        # Show pre-reranking order
        print(f"\n   Urutan SEBELUM Reranking:")
        print(f"   {'No':>3s} │ {'Skor RRF':>10s} │ Chunk ID")
        print(f"   {'───':>3s}─┼─{'──────────':>10s}─┼─{'─'*35}")
        for i, doc in enumerate(candidate_docs[:5]):
            doc_id = str(doc.get('id', doc.get('chunk_id', '?')))
            if len(doc_id) > 35:
                doc_id = doc_id[-35:]
            print(f"   {i+1:>3d} │ {doc.get('score', 0.0):>10.4f} │ {doc_id}")

        # Rerank
        rerank_start = time.time()
        reranked_docs = rag._reranker.rerank(query, candidate_docs, top_k=5)
        rerank_time = time.time() - rerank_start

        # Show post-reranking order
        print(f"\n   Urutan SESUDAH Reranking ({rerank_time:.3f}s):")
        print(f"   {'No':>3s} │ {'Skor CE':>10s} │ Chunk ID")
        print(f"   {'───':>3s}─┼─{'──────────':>10s}─┼─{'─'*35}")
        for i, doc in enumerate(reranked_docs[:5]):
            doc_id = str(doc.get('id', doc.get('chunk_id', '?')))
            if len(doc_id) > 35:
                doc_id = doc_id[-35:]
            print(f"   {i+1:>3d} │ {doc.get('cross_encoder_score', doc.get('score', 0.0)):>10.4f} │ {doc_id}")

        final_docs = reranked_docs
    else:
        print("   (Reranking dinonaktifkan, menggunakan hasil RRF)")
        final_docs = fused_docs
        rerank_time = 0.0

    # === TAHAP 6: CONTEXT BUILDING ===
    print_header("📋 TAHAP 6: CONTEXT BUILDING")

    context_start = time.time()
    context_data = rag._context_builder.build_context(final_docs, query)
    context_time = time.time() - context_start
    context_text = context_data.get("context", "")

    print(f"   Dokumen digunakan : {len(final_docs[:5])}")
    print(f"   Panjang konteks   : {len(context_text):,} karakter")
    print(f"   Waktu             : {context_time:.3f}s")
    print(f"\n   Preview konteks (200 karakter pertama):")
    print(f"   ┌{'─'*56}┐")
    for line in context_text[:200].split('\n')[:5]:
        print(f"   │ {line[:54]:<54s} │")
    print(f"   └{'─'*56}┘")

    # === TAHAP 7: GENERASI JAWABAN ===
    print_header("🤖 TAHAP 7: GENERASI JAWABAN (LLM)")

    print(f"   Model LLM: {rag.config.llm.model_type}")
    print(f"   Mengirim query + konteks ke LLM...")

    gen_start = time.time()
    answer = rag._generate_answer(query, context_text)
    gen_time = time.time() - gen_start

    total_time = init_time + retrieval_time + rerank_time + context_time + gen_time

    print(f"\n   Jawaban LLM ({gen_time:.2f}s):")
    print(f"   ╔{'═'*56}╗")
    # Word-wrap the answer
    words = answer.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 54:
            print(f"   ║ {line:<54s} ║")
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        print(f"   ║ {line:<54s} ║")
    print(f"   ╚{'═'*56}╝")

    # === RINGKASAN WAKTU ===
    print_header("⏱️  RINGKASAN WAKTU EKSEKUSI")
    print_box("Breakdown Waktu per Tahap", [
        ("Inisialisasi", f"{init_time:.2f}s"),
        ("Retrieval (Hybrid Search)", f"{retrieval_time:.2f}s"),
        ("Reranking (Cross-Encoder)", f"{rerank_time:.2f}s"),
        ("Context Building", f"{context_time:.3f}s"),
        ("Generasi LLM", f"{gen_time:.2f}s"),
        ("─" * 28, "─" * 20),
        ("TOTAL", f"{total_time:.2f}s"),
    ])

    print("\n" + "═" * 64)
    print("  ✅ Pipeline selesai.")
    print("═" * 64 + "\n")


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
