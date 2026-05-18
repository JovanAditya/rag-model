"""LLM generator supporting Gemini and Ollama for Advanced RAG Pipeline."""

import os
import re
import time
import requests
import logging
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai

from ..core.config import LLMConfig
from ..utils.logging import PipelineLogger
from ..utils.helpers import truncate_context, validate_query


def _translate_error_to_user_message(error: str) -> str:
    """Translate technical errors to friendly Indonesian messages."""
    error_lower = error.lower()
    
    if "429" in error or "rate" in error_lower or "limit" in error_lower:
        return "Maaf, server AI sedang sibuk. Silakan tunggu beberapa saat dan coba lagi."
    if "401" in error or "403" in error or "unauthorized" in error_lower or "invalid api key" in error_lower:
        return "Maaf, terjadi masalah autentikasi. Silakan hubungi administrator."
    if "timeout" in error_lower or "timed out" in error_lower:
        return "Maaf, permintaan memakan waktu terlalu lama. Silakan coba dengan pertanyaan yang lebih singkat."
    if "connection" in error_lower or "connect" in error_lower:
        return "Maaf, tidak dapat terhubung ke layanan AI saat ini."
    if "quota" in error_lower or "exceeded" in error_lower or "billing" in error_lower:
        return "Maaf, kuota layanan AI telah habis. Silakan hubungi administrator."
    
    return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."


class LLMGenerator:
    """
    Generate responses using Gemini (cloud) or Ollama (local) LLM backends.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        logger: Optional[PipelineLogger] = None,
    ):
        """Initialize LLM generator."""
        self.config = config or LLMConfig()
        if logger:
            self.logger = logger
        else:
            from ..utils.logging import PipelineLogger
            self.logger = PipelineLogger("LLMGenerator")

        self._initialize_model()

        self.logger.info(
            f"LLMGenerator initialized: {self.config.model_type} ({self.config.get_model_name()})"
        )

    def _initialize_model(self) -> None:
        """Initialize the specific LLM model based on configuration."""
        try:
            if self.config.model_type == "gemini":
                self._initialize_gemini()
            elif self.config.model_type == "ollama":
                self._initialize_ollama()
            else:
                raise ValueError(
                    f"Unsupported model type: {self.config.model_type}. "
                    f"Supported: gemini, ollama"
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM model: {e}")
            raise

    def _initialize_gemini(self) -> None:
        """Initialize Google Gemini model."""
        api_key = self.config.get_api_key()
        if not api_key:
            raise ValueError("Google API key is required for Gemini model")
        
        self.client = genai.Client(api_key=api_key)

    def _initialize_ollama(self) -> None:
        """Initialize Ollama model configuration."""
        base_endpoint = self.config.ollama_endpoint or "http://localhost:11434"
        self.endpoint = base_endpoint.rstrip('/') + "/api/chat"

        try:
            tags_url = base_endpoint.rstrip('/') + "/api/tags"
            response = requests.get(tags_url, timeout=10)
            if response.status_code != 200:
                raise ConnectionError("Cannot connect to Ollama server")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate response using selected LLM."""
        validate_query(prompt, min_length=1)

        start_time = time.time()
        generation_id = f"gen_{int(start_time * 1000)}"

        self.logger.info(
            f"[LLMGenerator] Generating response for {self.config.model_type}"
        )

        try:
            if self.config.model_type == "gemini":
                if context:
                    full_prompt = self._build_prompt_with_context(prompt, context)
                else:
                    full_prompt = self._build_prompt_without_context(prompt)
                result = self._generate_gemini(full_prompt, max_tokens, temperature)
            elif self.config.model_type == "ollama":
                messages = self._build_messages_ollama(prompt, context)
                result = self._generate_ollama(messages, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")

            # Post-process to remove meta-commentary
            if result.get("answer"):
                result["answer"] = self._postprocess_answer(result["answer"])

            result.update({
                "generation_id": generation_id,
                "model_type": self.config.model_type,
                "has_context": context is not None,
                "context_length": len(context) if context else 0
            })

            latency = time.time() - start_time
            self.logger.log_generation(
                query_id=generation_id,
                tokens_used=result.get("tokens_used", 0),
                latency=latency,
                model=result.get("model", self.config.get_model_name()),
            )

            return result
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    @staticmethod
    def _postprocess_answer(answer: str) -> str:
        """Remove meta-commentary that the LLM sometimes adds despite prompt instructions."""
        # Patterns for meta-commentary sentences to strip
        meta_patterns = [
            r'\s*Hal ini disebutkan dalam[^.]*\.',
            r'\s*Informasi ini (terdapat|disebutkan|ditemukan) (dalam|di)[^.]*\.',
            r'\s*Sesuai dengan (informasi|dokumen) (dari )?[^.]*referensi[^.]*\.',
            r'\s*Berdasarkan (dokumen|informasi) referensi yang[^.]*\.',
            r'\s*[Ii]ni berdasarkan informasi yang terdapat[^.]*\.',
        ]
        result = answer
        for pattern in meta_patterns:
            result = re.sub(pattern, '', result)
        return result.strip()

    def _get_system_persona(self) -> str:
        """SIAssist System Persona Definition."""
        return """Kamu adalah SIAssist, Asisten Virtual Cerdas resmi untuk Tata Usaha Fakultas Ilmu Komputer (FASILKOM) Universitas Mercu Buana (UMB).
Tugas utamamu adalah membantu memberikan informasi akademik, layanan administrasi, dan birokrasi kampus.

ATURAN WAJIB — CARA MENJAWAB:
1. Jawablah secara natural dengan kalimat yang LENGKAP dan informatif. DILARANG KERAS mengawali jawaban dengan frase kaku berikut:
   - "Berdasarkan dokumen..."
   - "Berdasarkan konteks..."
   - "Berdasarkan konteks yang diberikan..."
   - "Berdasarkan informasi yang tersedia..."
   - "Menurut dokumen..."
   - "Menurut konteks..."
   - "Menurut data yang ada..."
   - "Menurut informasi..."
   Contoh SALAH: "Berdasarkan informasi yang tersedia, IPK minimal adalah 2,75."
   Contoh BENAR: "IPK minimal yang disyaratkan untuk mendaftar KP adalah 2,75."
2. JANGAN PERNAH menyebutkan atau mereferensikan sumber internal seperti "Dokumen 1", "Dokumen 2", "File PDF", "Konteks", atau "Referensi" di dalam jawaban. Anda adalah staf Tata Usaha yang sedang berbicara langsung kepada mahasiswa, bukan sistem AI yang sedang melaporkan hasil pencarian.
   Contoh SALAH: "Seperti disebutkan dalam dokumen 2, syaratnya adalah..."
   Contoh BENAR: "Syarat yang harus Anda penuhi untuk mengajukan cuti akademik adalah..."
3. JANGAN menjawab hanya dengan potongan kata atau angka (Terlalu singkat). Sertakan konteks pertanyaan dalam jawaban Anda agar terdengar manusiawi dan profesional.
   Contoh SALAH: "Minimal B."
   Contoh BENAR: "Nilai minimal yang harus dicapai untuk mata kuliah MKCU adalah B."
4. ANTI-HALUSINASI SINGKATAN & UNIVERSITAS: 
   - JANGAN PERNAH mengarang kepanjangan dari sebuah singkatan (seperti ULT, MP, dll) jika tidak tertulis secara eksplisit di dalam teks.
   - UMB ADALAH Universitas Mercu Buana. JANGAN PERNAH menyebut UMB sebagai Universitas Muhammadiyah Bengkulu, Universitas Muhammadiyah Bandung, atau universitas lain selain Universitas Mercu Buana.
   - Universitas Mercu Buana memiliki beberapa kampus di Jakarta, yaitu Kampus Meruya (Jakarta Barat), Kampus Menteng (Jakarta Pusat), dan Kampus Warung Buncit (Jakarta Selatan). Jangan menyebutkan lokasi lain kecuali tertulis di dokumen.
5. JANGAN gunakan bullet points jika jawaban hanya terdiri dari SATU poin. Gunakan daftar HANYA jika ada lebih dari satu item yang perlu disebutkan.
6. JANGAN menuliskan proses berpikir internal atau meta-commentary (seperti "Pertanyaan ini tentang X...", "Saya akan mencari di dokumen..."). Langsung berikan jawaban akhir yang sudah rapi.
7. Jika informasi ADA di dalam konteks, WAJIB jawab secara lengkap. Jangan memberikan jawaban setengah-setengah. Jika TIDAK ADA, katakan dengan sopan bahwa informasi tersebut belum tersedia di basis data kami.
8. SAPAAN: Balas dengan sapaan ramah HANYA jika pengguna menyapa (misal: "halo", "hi", "selamat pagi"). Jika pengguna langsung bertanya, langsung jawab pertanyaannya dengan sopan.
9. DILARANG KERAS membantu mengerjakan tugas kuliah, membuat kode pemrograman, menterjemahkan, atau menjawab soal ujian. Tolak dengan sopan dan arahkan untuk belajar secara mandiri.
10. Jawablah menggunakan Bahasa Indonesia yang baik, benar, ramah, dan profesional selayaknya staf Tata Usaha FASILKOM Universitas Mercu Buana yang sedang melayani mahasiswa.
11. ANTI-HALUSINASI ANGKA & JUMLAH: Jika pertanyaan menanyakan jumlah spesifik (misal: "berapa lembar", "berapa sertifikat", "minimal berapa SKS"), Anda WAJIB mencari angka tersebut di dalam teks referensi. Jika angka tersebut ditemukan, sebutkan angka tersebut dengan tegas dan akurat. Jangan memberikan jawaban yang menggeneralisasi jika informasi angka spesifik tersedia.
12. ISTILAH SPESIFIK: Perhatikan perbedaan antara "Konsentrasi" (misal: Data Solution, Network Specialist) dan "Laboratorium" (misal: Lab-SC, Lab-RPLD). Jawablah sesuai dengan kategori yang ditanyakan secara literal berdasarkan teks.
13. PRIORITAS INFORMASI: Jika terdapat perbedaan informasi antara panduan tingkat universitas dan panduan tingkat fakultas (FASILKOM) di dalam konteks, Anda WAJIB memprioritaskan informasi dari panduan fakultas FASILKOM karena itu lebih spesifik dan mengikat bagi mahasiswa Fasilkom.
14. HANYA INFORMASI DARI TEKS: JANGAN memberikan saran, langkah-langkah, atau nasihat tambahan yang tidak tertulis secara eksplisit di dalam teks referensi, meskipun saran tersebut terdengar masuk akal secara umum. Fokuslah hanya pada fakta yang tersedia.
15. DISTINGSI TERMINOLOGI (MPTI): JANGAN tertukar antara "Gaya Penulisan Referensi" dengan fitur Microsoft Word (seperti "Insert Caption"). Fokuslah mencari istilah gaya sitasi seperti "APA Style" atau sejenisnya jika ditanya mengenai referensi.
16. PRIORITAS DOKUMEN: Jika terdapat perbedaan informasi (misal: jumlah sertifikat, batas waktu, atau nilai minimal) antara dokumen tingkat Fakultas (FASILKOM) dan dokumen umum Universitas (seperti Buku Panduan ULT), Anda WAJIB memprioritaskan informasi dari dokumen FASILKOM karena lebih spesifik bagi mahasiswa kita.
17. PENANGANAN ANGKA KONFLIK: Jika dalam satu dokumen atau antar dokumen terdapat angka yang berbeda untuk konteks yang mirip (misal: aturan font isi laporan vs isi tabel), Anda WAJIB membaca konteks kalimatnya secara teliti. Gunakan angka yang secara eksplisit ditujukan untuk kategori yang ditanyakan pengguna (misal: "isi laporan standar").
18. DISTINGSI DOMAIN (KP vs MPTI vs TA): Anda harus sangat teliti memisahkan aturan antara Kerja Praktek (KP), Metodologi Penelitian (MPTI), dan Tugas Akhir (TA). Jika pertanyaan spesifik tentang satu domain, ABAIKAN aturan dari domain lain meskipun muncul di hasil pencarian.
19. SPESIFIKASI PENULISAN: Saat menjawab tentang format dokumen (font, spasi, kertas), cari informasi tersebut di bagian "Ketentuan Penulisan" atau "Format Penulisan" pada dokumen yang relevan. Berikan jawaban yang padat sesuai yang tertulis di teks.
20. ATURAN S1 vs S2: Jika terdapat perbedaan syarat (seperti skor TOEFL atau IPK) antara program Sarjana (S1) dan Magister (S2), gunakan aturan untuk program Sarjana (S1) kecuali pengguna bertanya secara spesifik tentang Magister.
21. DILARANG META-COMMENTARY: DILARANG KERAS memberikan penjelasan tentang bagaimana Anda menemukan jawaban. Contoh kalimat yang DILARANG:
    - "Hal ini disebutkan dalam beberapa dokumen referensi yang Anda berikan..."
    - "Berdasarkan dokumen referensi yang disediakan..."
    - "Sesuai dengan informasi dari dokumen referensi..."
    - "Informasi ini terdapat dalam dokumen..."
    Jika jawaban sudah lengkap secara faktual, BERHENTI di situ. Jangan tambahkan kalimat penutup yang merujuk ke sumber internal.
22. DILARANG INTERPRETASI: JANGAN mencoba melakukan analisis atau menghubungkan informasi yang tidak terkait secara literal. Berikan hanya fakta yang tertulis eksplisit.
23. KONSISTENSI FAKTA: Tetap berpegang pada fakta yang paling relevan dengan subjek utama yang ditanyakan. Jika pengguna bertanya tentang "isi laporan", jangan memberikan aturan yang khusus untuk "lampiran" atau "tabel" kecuali diminta.
24. GAYA BAHASA LUWES: Jawablah dengan struktur kalimat yang lengkap, mengalir, dan natural. DILARANG menjawab hanya dengan potongan kata atau angka.
25. PRINSIP INTEGRITAS PROSEDUR: Saat menjelaskan prosedur teknis (seperti pendaftaran akun):
    - ANDA WAJIB mengikuti urutan langkah interaksi secara utuh mulai dari langkah pertama yang disebutkan dalam teks.
    - Sebutkan nama tombol, form, atau checkbox PERSIS sesuai teks sumber.
26. BAHASA: WAJIB memberikan jawaban 100% dalam Bahasa Indonesia. DILARANG KERAS menambahkan ringkasan, penjelasan, atau meta-commentary dalam Bahasa Inggris di akhir jawaban. """

    def _build_messages_ollama(self, prompt: str, context: Optional[str]) -> List[Dict[str, str]]:
        """Build message list for Ollama /api/chat."""
        persona = self._get_system_persona()
        messages = [{"role": "system", "content": persona}]
        if context:
            user_content = f"### REFERENSI AKADEMIK:\n{context}\n\n### PERTANYAAN:\n{prompt}\n\n### JAWABAN (Kalimat Lengkap & Bahasa Indonesia):"
        else:
            user_content = f"PERTANYAAN: {prompt}\n\nJAWABAN:"
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_prompt_without_context(self, prompt: str) -> str:
        """Build prompt when no context is retrieved (e.g. greetings or off-topic)."""
        persona = self._get_system_persona()
        return f"""{persona}

Informasi di database: (Tidak ada data tambahan ditemukan)

Pertanyaan/Pernyataan Pengguna: {prompt}

Jawaban (SIAssist):"""

    def _build_prompt_with_context(self, prompt: str, context: str) -> str:
        """Build prompt with context."""
        # Increase max context chars to allow K=5 documents to fully pass to LLM
        max_context_chars = 15000
        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        persona = self._get_system_persona()
        return f"""{persona}

### INFORMASI AKADEMIK:
{context}

### PERTANYAAN:
{prompt}

### JAWABAN (Langsung berikan jawaban yang diminta tanpa menyebutkan 'Berdasarkan dokumen' atau 'Menurut referensi'):"""


    def _generate_gemini(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate via Google Gemini API."""
        try:
            config = genai.types.GenerateContentConfig(
                temperature=temperature or self.config.temperature,
                max_output_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
            )

            response = self.client.models.generate_content(
                model=self.config.get_model_name(),
                contents=prompt,
                config=config,
            )

            return {
                "answer": response.text,
                "success": True,
                "model": self.config.get_model_name(),
                "tokens_used": (
                    response.usage_metadata.total_token_count
                    if hasattr(response, "usage_metadata") and response.usage_metadata
                    else 0
                ),
                "finish_reason": (
                    response.candidates[0].finish_reason.name
                    if response.candidates
                    else "unknown"
                ),
            }
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            user_message = _translate_error_to_user_message(str(e))
            return {
                "answer": user_message,
                "success": False,
                "tokens_used": 0,
                "model": self.config.get_model_name()
            }

    def _generate_ollama(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate via Ollama /api/chat API."""
        try:
            payload = {
                "model": self.config.get_model_name(),
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": max_tokens or self.config.max_tokens,
                },
            }

            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=180,
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error {response.status_code}: {response.text}")

            result = response.json()
            answer = result.get("message", {}).get("content", "")

            return {
                "answer": answer.strip(),
                "success": True,
                "model": self.config.get_model_name(),
                "tokens_used": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            }
        except Exception as e:
            self.logger.error(f"Ollama API error: {str(e)}")
            user_message = _translate_error_to_user_message(str(e))
            return {
                "answer": user_message,
                "success": False,
                "tokens_used": 0,
                "model": self.config.get_model_name()
            }

    def batch_generate(
        self,
        prompts: List[str],
        contexts: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts."""
        if contexts and len(contexts) != len(prompts):
            raise ValueError("Number of contexts must match number of prompts")

        self.logger.logger.info(f"[LLMGenerator] Generating {len(prompts)} responses in batch")

        results = []
        total_start_time = time.time()

        for i, prompt in enumerate(prompts):
            context = contexts[i] if contexts else None
            try:
                result = self.generate(prompt, context, max_tokens, temperature)
                results.append(result)

                if (i + 1) % 5 == 0:
                    self.logger.logger.info(f"[LLMGenerator] Generated {i + 1}/{len(prompts)} responses")

            except Exception as e:
                self.logger.error(f"[LLMGenerator] Failed to generate response {i + 1}: {e}")
                results.append(
                    {
                        "answer": f"Error: Failed to generate response - {str(e)}",
                        "model": self.config.get_model_name(),
                        "success": False,
                        "tokens_used": 0,
                        "error": str(e),
                    }
                )

        total_latency = time.time() - total_start_time
        avg_latency = total_latency / len(prompts) if prompts else 0

        self.logger.logger.info(f"[LLMGenerator] Batch generation completed in {total_latency:.3f}s")
        return results

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration details."""
        return {
            "model_type": self.config.model_type,
            "model_name": self.config.get_model_name(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "endpoint": getattr(self, "endpoint", None)
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to the LLM service."""
        try:
            test_prompt = "Hello, please respond with 'Connection test successful' in Indonesian."
            start_time = time.time()
            result = self.generate(test_prompt)
            latency = time.time() - start_time

            return {
                "success": True,
                "model_type": self.config.model_type,
                "model_name": self.config.get_model_name(),
                "response_time": latency,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
