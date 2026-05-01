"""LLM generator supporting Gemini and Ollama for Advanced RAG Pipeline."""

import os
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
        self.endpoint = base_endpoint.rstrip('/') + "/api/generate"

        try:
            tags_url = self.endpoint.replace('/api/generate', '/api/tags')
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

        if context:
            full_prompt = self._build_prompt_with_context(prompt, context)
        else:
            full_prompt = self._build_prompt_without_context(prompt)

        self.logger.info(
            f"[LLMGenerator] Generating response for {self.config.model_type}"
        )

        try:
            if self.config.model_type == "gemini":
                result = self._generate_gemini(full_prompt, max_tokens, temperature)
            elif self.config.model_type == "ollama":
                result = self._generate_ollama(full_prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")

            result.update({
                "generation_id": generation_id,
                "model_type": self.config.model_type,
                "prompt_length": len(full_prompt),
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
            latency = time.time() - start_time
            self.logger.log_error(
                "LLMGenerator",
                e,
                {"prompt_length": len(full_prompt), "latency": latency},
            )
            raise

    def _get_system_persona(self) -> str:
        """SIAssist System Persona Definition."""
        return """Kamu adalah SIAssist, Asisten Virtual Cerdas resmi untuk Tata Usaha Fakultas Ilmu Komputer (FASILKOM).
Tugas utamamu adalah membantu memberikan informasi akademik, layanan administrasi, dan birokrasi kampus.

BATASAN & ATURAN KETAT:
1. SAPAAN: Balas dengan sapaan ramah HANYA jika pengguna menyapa (misal: "halo", "hi", "selamat pagi", "hey"). Jika pengguna langsung bertanya tentang akademik, JANGAN awali dengan sapaan — langsung jawab pertanyaannya.
2. JANGAN PERNAH membantu mengerjakan tugas kuliah, membuat kode pemrograman, menterjemahkan atau menjawab soal ujian. Jika diminta, tolak dengan sopan dan ingatkan bahwa kamu adalah chatbot khusus informasi administrasi/akademik.
3. Tolak pertanyaan yang sama sekali tidak ada hubungannya dengan perkuliahan, Fasilkom, atau kampus.
4. Jawab secara natural, informatif, dan tidak kaku. DILARANG KERAS menggunakan frase pembuka berikut:
   - "Berdasarkan dokumen..."
   - "Berdasarkan konteks yang diberikan..."
   - "Berdasarkan informasi yang tersedia..."
   - "Menurut dokumen..."
   - "Menurut data yang ada..."
   Langsung saja jawab inti pertanyaannya tanpa frase pembuka apapun."""

    def _build_prompt_without_context(self, prompt: str) -> str:
        """Build prompt when no context is retrieved (e.g. greetings or off-topic)."""
        persona = self._get_system_persona()
        return f"""{persona}

Informasi di database: (Tidak ada data tambahan ditemukan)

Pertanyaan/Pernyataan Pengguna: {prompt}

Jawaban (SIAssist):"""

    def _build_prompt_with_context(self, prompt: str, context: str) -> str:
        """Build prompt with context."""
        max_context_chars = 3000
        if len(context) > max_context_chars:
            context = truncate_context([context], max_context_chars)

        persona = self._get_system_persona()
        return f"""{persona}

Informasi dari Arsip TU yang tersedia:
{context}

Pertanyaan/Pernyataan Pengguna: {prompt}

Jawaban (SIAssist):"""


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
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate via Ollama API."""
        try:
            payload = {
                "model": self.config.get_model_name(),
                "prompt": prompt,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "top_p": self.config.top_p,
                "stream": False,
            }

            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=180,
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error {response.status_code}: {response.text}")

            result = response.json()

            return {
                "answer": result.get("response", ""),
                "success": True,
                "model": self.config.get_model_name(),
                "tokens_used": result.get("eval_count", 0),
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "total_time": result.get("total_duration", 0) / 1e9,
                "load_time": result.get("load_duration", 0) / 1e9,
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
