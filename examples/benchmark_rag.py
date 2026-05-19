#!/usr/bin/env python3
import os
import sys
import time
import psutil
import threading
import torch
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_model import AcademicRAG
from rag_model.core.config import RAGConfig

# List of realistic questions to test
TEST_QUERIES = [
    "Apa syarat untuk mengikuti Kerja Praktek (KP)?",
    "Bagaimana prosedur pendaftaran wisuda?",
    "Berapa batas waktu revisi setelah sidang tugas akhir?",
    "Apa saja syarat pengajuan cuti akademik?",
    "Bagaimana aturan konversi nilai magang?"
]

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.1, monitor_ollama=False):
        super().__init__()
        self.interval = interval
        self.running = True
        self.process = psutil.Process(os.getpid())
        self.monitor_ollama = monitor_ollama
        
        # Stats history
        self.cpu_percentages = []
        self.ram_mb_samples = []
        
        # Ollama stats history
        self.ollama_cpu_percentages = []
        self.ollama_ram_mb_samples = []
        
        # Combined stats
        self.combined_cpu_percentages = []
        self.combined_ram_mb_samples = []
        
        self.ram_system_used = []

    def get_ollama_processes(self):
        p_list = []
        if not self.monitor_ollama:
            return p_list
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info['name'].lower()
                if 'ollama' in name or 'llama' in name:
                    p_list.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return p_list

    def run(self):
        while self.running:
            try:
                # 1. RAG python process stats
                rag_cpu = psutil.cpu_percent(interval=None)
                rag_ram = self.process.memory_info().rss / 1024 / 1024
                
                self.cpu_percentages.append(rag_cpu)
                self.ram_mb_samples.append(rag_ram)
                
                # 2. Ollama processes stats (if enabled)
                ollama_cpu = 0.0
                ollama_ram = 0.0
                
                if self.monitor_ollama:
                    ollama_procs = self.get_ollama_processes()
                    for p in ollama_procs:
                        try:
                            # cpu_percent(interval=None) on process might return 0.0 initially, but accumulates
                            ollama_cpu += p.cpu_percent(interval=None)
                            ollama_ram += p.memory_info().rss / 1024 / 1024
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
                self.ollama_cpu_percentages.append(ollama_cpu)
                self.ollama_ram_mb_samples.append(ollama_ram)
                
                # 3. Combined stats (RAG + Ollama)
                self.combined_cpu_percentages.append(rag_cpu + ollama_cpu)
                self.combined_ram_mb_samples.append(rag_ram + ollama_ram)
                
                # 4. System wide RAM
                self.ram_system_used.append(psutil.virtual_memory().used / 1024 / 1024)
                
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False

def run_benchmark():
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark Tool")
    parser.add_argument("--provider", type=str, choices=["gemini", "ollama"], default="gemini",
                        help="LLM Provider to use (gemini or ollama)")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model name (e.g. llama3.2:latest, qwen2.5:7b)")
    
    args = parser.parse_args()
    
    provider = args.provider
    model_name = args.model
    
    if provider == "ollama":
        if not model_name:
            model_name = "llama3.2:latest"  # Default to lightweight 3B model
        # Set environment variable for RAGConfig loading
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_MODEL"] = model_name
        print(f"⚙️ Configured local LLM: Ollama | Model: {model_name}")
    else:
        os.environ["LLM_PROVIDER"] = "gemini"
        print("⚙️ Configured Cloud LLM: Gemini")
        
    print("==================================================")
    print("🧪 RAG PIPELINE BENCHMARK & RESOURCE USAGE TEST")
    print("==================================================")
    
    # 1. Start resource monitor (monitor Ollama process resource usage if provider is ollama)
    monitor = ResourceMonitor(interval=0.05, monitor_ollama=(provider == "ollama"))
    monitor.start()
    
    time.sleep(1.0) # Let it capture some baseline idle stats
    idle_ram = sum(monitor.ram_mb_samples) / len(monitor.ram_mb_samples)
    print(f"📊 Process RAG idle RAM   : {idle_ram:.2f} MB")
    if provider == "ollama":
        idle_ollama_ram = sum(monitor.ollama_ram_mb_samples) / len(monitor.ollama_ram_mb_samples)
        print(f"📊 Process Ollama idle RAM: {idle_ollama_ram:.2f} MB")
    
    # 2. Initialize the pipeline
    print(f"\n[1/3] Inisialisasi AcademicRAG ({provider.upper()})...")
    start_init = time.perf_counter()
    
    config = RAGConfig()
    config.llm.model_type = provider
    config.retrieval.pipeline_type = "advanced"
    config.retrieval.use_reranking = True
    
    rag = AcademicRAG(config=config, research_mode=True, response_format="full")
    rag._initialize_components()
    
    init_time = time.perf_counter() - start_init
    print(f"✅ Inisialisasi selesai dalam: {init_time:.2f} detik")
    
    # Capture RAM after initialization
    post_init_ram = monitor.ram_mb_samples[-1]
    print(f"📊 Process RAG RAM after init: {post_init_ram:.2f} MB")
    
    # 3. Run queries sequentially (Latency & Throughput test)
    print(f"\n[2/3] Menjalankan {len(TEST_QUERIES)} kueri uji secara berurutan...")
    latencies = []
    
    # Limit to 3 queries for Ollama CPU to save time if it's very slow
    num_queries = 3 if provider == "ollama" else len(TEST_QUERIES)
    
    for i in range(num_queries):
        query = TEST_QUERIES[i]
        print(f"   👉 Kueri {i+1}: \"{query}\"")
        start_query = time.perf_counter()
        
        try:
            result = rag.query(query)
            end_query = time.perf_counter()
            duration = end_query - start_query
            latencies.append(duration)
            print(f"      ✓ Berhasil ({duration:.2f}s)")
        except Exception as e:
            print(f"      ❌ Gagal: {e}")
            
    # Stop monitoring
    monitor.stop()
    monitor.join()
    
    # 4. Process and print benchmark results
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    throughput = 1 / avg_latency if avg_latency > 0 else 0
    
    print("\n==================================================")
    print("📊 HASIL BEBAN KERJA & LATENCY")
    print("==================================================")
    print(f"• Jumlah Kueri Uji       : {len(latencies)}")
    print(f"• Rata-rata Latency      : {avg_latency:.2f} detik")
    print(f"• Latency Maksimum       : {max_latency:.2f} detik")
    print(f"• Latency Minimum        : {min_latency:.2f} detik")
    print(f"• Throughput (req/s)     : {throughput:.2f} req/s")
    
    print("\n==================================================")
    print("📈 HASIL RESOURCE USAGE (PENGGUNAAN MEMORI & CPU)")
    print("==================================================")
    
    # Calculate CPU stats
    valid_combined_cpus = [c for c in monitor.combined_cpu_percentages if c > 0]
    avg_cpu = sum(valid_combined_cpus) / len(valid_combined_cpus) if valid_combined_cpus else 0
    peak_cpu = max(monitor.combined_cpu_percentages) if monitor.combined_cpu_percentages else 0
    
    peak_ram_rag = max(monitor.ram_mb_samples) if monitor.ram_mb_samples else 0
    peak_ram_combined = max(monitor.combined_ram_mb_samples) if monitor.combined_ram_mb_samples else 0
    peak_ram_system = max(monitor.ram_system_used) if monitor.ram_system_used else 0
    
    print(f"• CPU Rata-rata (Total)  : {avg_cpu:.1f}%")
    print(f"• CPU Peak (Total Puncak): {peak_cpu:.1f}%")
    print(f"• RAM Peak Proses RAG    : {peak_ram_rag:.2f} MB")
    if provider == "ollama":
        peak_ram_ollama = max(monitor.ollama_ram_mb_samples) if monitor.ollama_ram_mb_samples else 0
        print(f"• RAM Peak Proses Ollama : {peak_ram_ollama:.2f} MB")
        print(f"• RAM Peak Gabungan      : {peak_ram_combined:.2f} MB")
    print(f"• RAM Peak Total Sistem  : {peak_ram_system:.2f} MB")
    
    if torch.cuda.is_available():
        # Clean CUDA memory stats
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"• VRAM Peak (CUDA GPU)   : {vram_mb:.2f} MB")
    else:
        print("• Penggunaan GPU/VRAM    : Tidak terdeteksi (CPU-only)")
        
    print("\n==================================================")
    print("💡 ANALISIS & REKOMENDASI UNTUK MODEL LOKAL:")
    print("==================================================")
    
    if provider == "ollama":
        print(f"1. Model yang Diuji      : {model_name}")
        print(f"2. RAM Gabungan (Peak)   : {peak_ram_combined / 1024:.2f} GB")
        print(f"3. Latency Generasi CPU  : {avg_latency:.2f} detik per kueri")
        
        # RAM Recommendation
        rec_ram = (peak_ram_combined * 2.2) / 1024
        print(f"4. Rekomendasi RAM VPS   : Minimal {max(4.0, rec_ram):.0f} GB RAM.")
        if avg_latency > 15:
            print("5. Catatan Performa      : Latency model lokal di CPU sangat lambat (>15 detik).")
            print("                           Sangat disarankan menggunakan GPU VPS atau tetap menggunakan Gemini API.")
        else:
            print("5. Catatan Performa      : Latency masih dalam batas wajar untuk pengujian terbatas.")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()
