#!/usr/bin/env python3
"""
Basic Usage Example - Academic RAG

Basic usage example for Academic RAG question-answering system.
Questions are based on actual knowledge base content (UMB academic guides).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_model import AcademicRAG


def main():
    """Basic usage example with real knowledge base questions."""
    print("=" * 60)
    print("Academic RAG - Basic Usage Example")
    print("=" * 60)
    
    # 1. Initialize RAG model
    print("\n[1] Initializing Academic RAG...")
    rag = AcademicRAG()
    print("    Model initialized successfully!")
    
    # 2. Test queries based on actual knowledge base content
    # Documents: KP guidelines, Tugas Akhir, Academic guides, etc.
    test_questions = [
        "Apa syarat untuk mengikuti Kerja Praktek (KP)?",
        "Bagaimana prosedur pendaftaran sidang tugas akhir?",
        "Apa saja dokumen yang diperlukan untuk ujian sidang?",
        "Berapa jumlah SKS minimal untuk mengikuti tugas akhir?",
        "Apa yang dimaksud dengan program RPL?"
    ]
    
    print("\n[2] Running queries on knowledge base...\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"--- Query {i} ---")
        print(f"Q: {question}")
        
        result = rag.query(question)
        
        # Show truncated answer
        answer = result.get('answer', 'No answer')
        if len(answer) > 300:
            answer = answer[:300] + "..."
        
        print(f"A: {answer}")
        print(f"Sources: {len(result.get('sources', []))} documents")
        print()
    
    # 3. Compare baseline vs advanced pipeline
    print("\n[3] Comparing pipelines...")
    question = "Apa syarat untuk mengikuti Kerja Praktek?"
    
    print(f"\nQuestion: {question}")
    
    # Baseline (vector only)
    result_baseline = rag.query(question, pipeline_type="baseline")
    print(f"\nBaseline Pipeline:")
    print(f"  - Sources: {len(result_baseline.get('sources', []))}")
    print(f"  - Answer length: {len(result_baseline.get('answer', ''))} chars")
    
    # Advanced (hybrid + reranking)
    result_advanced = rag.query(question, pipeline_type="advanced")
    print(f"\nAdvanced Pipeline:")
    print(f"  - Sources: {len(result_advanced.get('sources', []))}")
    print(f"  - Answer length: {len(result_advanced.get('answer', ''))} chars")
    
    # 4. Health check
    print("\n[4] Health check...")
    health = rag.health_check()
    print(f"    Status: {'Ready' if health.get('ready') else 'Not Ready'}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
