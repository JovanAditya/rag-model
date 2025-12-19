#!/usr/bin/env python3
"""
Evaluation Example - Academic RAG

Evaluation guide for RAG model research/thesis.
Questions are based on actual knowledge base content.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Show evaluation example with actual knowledge base questions."""
    print("=" * 60)
    print("Academic RAG - Evaluation Example")
    print("=" * 60)
    
    print("\n## For complete evaluation, use thesis-evaluation/ folder\n")
    
    print("The thesis-evaluation/ folder contains:")
    print("  - Standard RAG vs Advanced RAG comparison")
    print("  - Retrieval metrics (Recall, Precision, MRR, MAP, NDCG)")
    print("  - RAGAS metrics (Faithfulness, Answer Relevancy)")
    print("  - Result visualization for thesis")
    
    print("\n## Quick Evaluation (here)\n")
    print("Run a simple evaluation with questions from actual knowledge base:")
    
    print("""
from rag_model import AcademicRAG

# Initialize
rag = AcademicRAG()

# Test queries based on UMB knowledge base content:
# - Panduan KP dan MBKM
# - Panduan Tugas Akhir
# - Buku Panduan Akademik
# - Panduan Pasca Sidang
test_questions = [
    "Apa syarat untuk mengikuti Kerja Praktek (KP)?",
    "Bagaimana prosedur pendaftaran sidang tugas akhir?",
    "Apa saja dokumen yang diperlukan untuk ujian sidang?",
    "Berapa jumlah SKS minimal untuk mengikuti tugas akhir?",
    "Apa yang dimaksud dengan program RPL?",
    "Bagaimana cara mengisi SKPI?",
    "Apa saja layanan yang tersedia di BAK?",
    "Bagaimana prosedur MBKM di UMB?"
]

# Run evaluation
results = []
for question in test_questions:
    result = rag.query(question)
    results.append({
        "question": question,
        "answer_length": len(result.get("answer", "")),
        "num_sources": len(result.get("sources", [])),
        "confidence": result.get("confidence", 0)
    })
    print(f"[OK] {question[:50]}...")

# Summary
avg_confidence = sum(r["confidence"] for r in results) / len(results)
avg_sources = sum(r["num_sources"] for r in results) / len(results)
print(f"\\nAverage confidence: {avg_confidence:.2f}")
print(f"Average sources: {avg_sources:.1f}")
""")
    
    print("\n## Full Evaluation\n")
    print("For complete baseline vs advanced comparison:")
    print()
    print("  cd d:/UMB/SKRIPSI/RAG/thesis-evaluation")
    print("  python experiments/run_baseline.py")
    print("  python experiments/run_advanced.py")
    print("  python experiments/run_comparison.py")
    print("  python visualizations/plot_results.py")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
