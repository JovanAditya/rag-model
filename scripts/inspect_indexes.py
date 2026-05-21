import sys
import os
from pathlib import Path
import json

# Add rag_model to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_model.indexing import UnifiedIndexManager
import chromadb

def inspect_indexes():
    # Hardcode to point to the actual thesis data directory (from rag-deploy root)
    chroma_dir = "data/chroma_db"
    cache_dir = "data/cache"
    collection_name = "academic_docs"
    
    print("="*80)
    print("🔍 INSPEKSI STRUKTUR INDEKS")
    print("="*80)

    # 1. Inspect BM25
    print("\n" + "-"*60)
    print("1. BM25 INVERTED INDEX (scikit-learn)")
    print("-"*60)
    
    try:
        import pickle
        import gzip
        import glob
        
        # Find the latest cache file
        cache_files = glob.glob(f"{cache_dir}/bm25_{collection_name}.pkl.gz")
        if cache_files:
            with gzip.open(cache_files[0], 'rb') as f:
                bm25_data = pickle.load(f)
            
            vocab = bm25_data['vocabulary']
            documents = bm25_data['documents']
            
            print(f"Total Dokumen     : {len(documents)}")
            print(f"Total Vocabulary  : {len(vocab)}")
            print(f"Parameter         : k1={bm25_data['k1']}, b={bm25_data['b']}")
            
            print("\n[Contoh Pemetaan Kosakata ke ID (Inverted Index Mapping)]:")
            # Sort vocab by ID and take a sample
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            # Print some random/interesting words, e.g., around the middle
            mid = len(sorted_vocab) // 2
            sample_vocab = sorted_vocab[mid:mid+5]
            for word, term_id in sample_vocab:
                print(f"  - '{word}' -> Term ID: {term_id}")
                
            print("\n[Contoh Representasi Teks Ter-Tokenisasi]:")
            print("  Teks Asli: 'Syarat minimal SKS Kerja Praktek'")
            if 'vectorizer' in bm25_data:
                vectorizer = bm25_data['vectorizer']
                transformed = vectorizer.transform(["Syarat minimal SKS Kerja Praktek"])
                print(f"  Sparse Vector   : {transformed.shape}")
                print(f"  Non-zero terms  : {transformed.nnz}")
        else:
            print("❌ BM25 cache tidak ditemukan di", cache_dir)
    except Exception as e:
        print(f"Error inspecting BM25: {e}")

    # 2. Inspect Vector Store
    print("\n" + "-"*60)
    print("2. CHROMADB VECTOR INDEX (IndoBERT 768-D)")
    print("-"*60)
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        print(f"Total Embedding   : {count}")
        print(f"Dimensi Vektor    : 768 (IndoBERT Base)")
        
        if count > 0:
            results = collection.get(limit=1, include=["embeddings", "metadatas", "documents"])
            
            chunk_id = results['ids'][0]
            metadata = results['metadatas'][0]
            document = results['documents'][0]
            embedding = results['embeddings'][0]
            
            print("\n[Contoh Struktur Chunk yang Tersimpan]:")
            print(f"  Chunk ID       : {chunk_id}")
            print(f"  Metadata       : {{")
            for k, v in list(metadata.items())[:4]: # just show a few
                print(f"    '{k}': '{str(v)[:50]}'")
            if len(metadata) > 4:
                print(f"    ... (+{len(metadata)-4} fields)")
            print(f"  }}")
            print(f"  Dokumen Teks   : {document[:100].replace(chr(10), ' ')}...")
            
            print("\n[Representasi Vektor (Dense Embedding)]:")
            print(f"  Tipe Data      : {type(embedding).__name__} (Panjang: {len(embedding)})")
            # Show first 5 and last 5 elements to illustrate a dense vector
            vec_preview = ", ".join([f"{x:.4f}" for x in embedding[:4]]) + " ... " + ", ".join([f"{x:.4f}" for x in embedding[-4:]])
            print(f"  Vektor Tensor  : [{vec_preview}]")
            
    except Exception as e:
        print(f"Error inspecting ChromaDB: {e}")

if __name__ == "__main__":
    inspect_indexes()
