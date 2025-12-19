"""LLM generator supporting multiple models for Advanced RAG Pipeline."""

import os
import json
import logging
import time
import requests
from typing import Literal, Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

from ..core.config import LLMConfig
from ..utils.logging import PipelineLogger
from ..utils.helpers import truncate_context, validate_query


class LLMGenerator:
    """
    Generate responses using configurable LLM backends.

    Supports multiple LLM providers including local models (via Ollama)
    and cloud APIs (Google Gemini) for flexible generation.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        logger: Optional[PipelineLogger] = None,
    ):
        """
        Initialize LLM generator.

        Args:
            config: LLM configuration
            logger: Optional logger instance
        """
        self.config = config or LLMConfig()
        # Use PipelineLogger if provided, otherwise create a basic logger
        if logger:
            self.logger = logger
        else:
            # Create a PipelineLogger instance for consistency
            from ..utils.logging import PipelineLogger
            self.logger = PipelineLogger("LLMGenerator")

        # Initialize model based on type
        self._initialize_model()

        self.logger.info(
            f"LLMGenerator initialized: {self.config.model_type} ({self.config.get_model_name()})"
        )

    def _initialize_model(self) -> None:
        """Initialize the specific LLM model based on configuration."""
        try:
            if self.config.model_type == "gemini":
                self._initialize_gemini()
            elif self.config.model_type == "openai":
                self._initialize_openai()
            elif self.config.model_type == "anthropic":
                self._initialize_anthropic()
            elif self.config.model_type in ["ollama", "qwen", "llama"]:
                self._initialize_ollama()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM model: {e}")
            raise

    def _initialize_gemini(self) -> None:
        """Initialize Google Gemini model."""
        api_key = self.config.get_api_key()
        if not api_key:
            raise ValueError("Google API key is required for Gemini model")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config.get_model_name())

    def _initialize_openai(self) -> None:
        """Initialize OpenAI model."""
        if openai is None:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

        api_key = self.config.get_api_key()
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI model")

        self.client = openai.OpenAI(api_key=api_key)

    def _initialize_anthropic(self) -> None:
        """Initialize Anthropic Claude model."""
        if anthropic is None:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")

        api_key = self.config.get_api_key()
        if not api_key:
            raise ValueError("Anthropic API key is required for Claude model")

        self.client = anthropic.Anthropic(api_key=api_key)

    def _initialize_ollama(self) -> None:
        """Initialize Ollama model configuration."""
        # Set default endpoint if not configured
        base_endpoint = self.config.ollama_endpoint or "http://localhost:11434"
        self.endpoint = base_endpoint.rstrip('/') + "/api/generate"

        # Test Ollama connection
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
        """
        Generate response using selected LLM.

        Args:
            prompt: Generation prompt
            context: Optional context to include
            max_tokens: Override max tokens if provided
            temperature: Override temperature if provided

        Returns:
            Dictionary with generation results
        """
        # Validate inputs
        validate_query(prompt, min_length=1)

        start_time = time.time()
        generation_id = f"gen_{int(start_time * 1000)}"

        # Combine context and prompt
        if context:
            full_prompt = self._build_prompt_with_context(prompt, context)
        else:
            full_prompt = prompt

        self.logger.info(
            f"[LLMGenerator] Generating response for {self.config.model_type}"
        )

        try:
            if self.config.model_type == "gemini":
                result = self._generate_gemini(full_prompt, max_tokens, temperature)
            elif self.config.model_type == "openai":
                result = self._generate_openai(full_prompt, max_tokens, temperature)
            elif self.config.model_type == "anthropic":
                result = self._generate_anthropic(full_prompt, max_tokens, temperature)
            else:  # qwen or llama via Ollama
                result = self._generate_ollama(full_prompt, max_tokens, temperature)

            # Add metadata
            result["generation_id"] = generation_id
            result["model_type"] = self.config.model_type
            result["prompt_length"] = len(full_prompt)
            result["has_context"] = context is not None
            result["context_length"] = len(context) if context else 0

            # Log generation
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

    def _build_prompt_with_context(self, prompt: str, context: str) -> str:
        """
        Build prompt with context.

        Args:
            prompt: Original prompt
            context: Context to include

        Returns:
            Combined prompt with context
        """
        # Truncate context if needed
        max_context_chars = 3000
        if len(context) > max_context_chars:
            context = truncate_context([context], max_context_chars)

        return f"""Kamu adalah asisten akademik. Jawab pertanyaan berikut secara LANGSUNG dan JELAS.

ATURAN PENTING:
- JANGAN gunakan kata pembuka seperti "Berdasarkan konteks...", "Menurut dokumen...", atau sejenisnya
- Langsung jawab poinnya
- Gunakan numbered list jika ada beberapa poin

Informasi yang tersedia:
{context}

Pertanyaan: {prompt}

Jawaban:"""

    def _generate_gemini(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate via Google Gemini API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generation result
        """
        generation_config = {
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        response = self.model.generate_content(
            prompt, generation_config=generation_config
        )

        return {
            "answer": response.text,
            "model": self.config.get_model_name(),
            "tokens_used": (
                response.usage_metadata.total_token_count
                if hasattr(response, "usage_metadata")
                else 0
            ),
            "finish_reason": (
                response.candidates[0].finish_reason.name
                if response.candidates
                else "unknown"
            ),
        }

    def _generate_openai(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate via OpenAI API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generation result
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.get_model_name(),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
            )

            return {
                "answer": response.choices[0].message.content,
                "success": True,
                "tokens_used": response.usage.total_tokens,
                "model": self.config.get_model_name()
            }

        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return {
                "answer": f"Error: OpenAI API failed - {str(e)}",
                "success": False,
                "tokens_used": 0,
                "model": self.config.get_model_name()
            }

    def _generate_anthropic(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate via Anthropic Claude API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generation result
        """
        try:
            response = self.client.messages.create(
                model=self.config.get_model_name(),
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return {
                "answer": response.content[0].text,
                "success": True,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "model": self.config.get_model_name()
            }

        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            return {
                "answer": f"Error: Anthropic API failed - {str(e)}",
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
        """
        Generate via Ollama API (for Qwen and Llama).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generation result
        """
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
            timeout=60,  # Increased timeout for local models
        )

        if response.status_code != 200:
            raise Exception(
                f"Ollama API error: {response.status_code} - {response.text}"
            )

        result = response.json()

        return {
            "answer": result.get("response", ""),
            "model": self.config.get_model_name(),
            "tokens_used": result.get("eval_count", 0),
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "total_time": result.get("total_duration", 0)
            / 1e9,  # Convert nanoseconds to seconds
            "load_time": result.get("load_duration", 0) / 1e9,
        }

    def batch_generate(
        self,
        prompts: List[str],
        contexts: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            contexts: List of contexts (optional)
            max_tokens: Override max tokens if provided
            temperature: Override temperature if provided

        Returns:
            List of generation results
        """
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

                # Log progress
                if (i + 1) % 5 == 0:
                    self.logger.logger.info(
                        f"[LLMGenerator] Generated {i + 1}/{len(prompts)} responses"
                    )

            except Exception as e:
                self.logger.error(
                    f"[LLMGenerator] Failed to generate response {i + 1}: {e}"
                )
                # Add error result
                results.append(
                    {
                        "answer": f"Error: Failed to generate response - {str(e)}",
                        "model": self.config.get_model_name(),
                        "tokens_used": 0,
                        "error": str(e),
                    }
                )

        total_latency = time.time() - total_start_time
        avg_latency = total_latency / len(prompts) if prompts else 0

        self.logger.logger.info(
            f"[LLMGenerator] Batch generation completed in {total_latency:.3f}s (avg: {avg_latency:.3f}s per response)"
        )

        return results

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information for the LLM generator.

        Returns:
            Dictionary with configuration details
        """
        return {
            "model_type": self.config.model_type,
            "model_name": self.config.get_model_name(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "endpoint": getattr(self, "endpoint", None),
            "capabilities": [
                "Context-aware generation",
                "Batch processing",
                "Retry mechanism",
                "Temperature control",
                "Token limit management",
            ],
        }

    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to the LLM service.

        Returns:
            Dictionary with connection test results
        """
        try:
            test_prompt = (
                "Hello, please respond with 'Connection test successful' in Indonesian."
            )
            start_time = time.time()

            result = self.generate(test_prompt)

            latency = time.time() - start_time

            return {
                "success": True,
                "model_type": self.config.model_type,
                "model_name": self.config.get_model_name(),
                "response_time": latency,
                "tokens_used": result.get("tokens_used", 0),
                "test_response": result.get("answer", "")[:100],
            }

        except Exception as e:
            return {
                "success": False,
                "model_type": self.config.model_type,
                "model_name": self.config.get_model_name(),
                "error": str(e),
            }
