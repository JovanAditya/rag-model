"""Prompt templates for Academic RAG Pipeline."""

from typing import Dict, Any, List, Optional
from ..core.config import LLMConfig


class PromptTemplate:
    """Template manager for RAG prompts."""

    def __init__(self, language: str = "id"):
        """
        Initialize prompt template manager.

        Args:
            language: Language code (id for Indonesian, en for English)
        """
        self.language = language
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates based on language."""
        if self.language == "id":
            return {
                "qa_context": """Kamu adalah asisten akademik yang membantu mahasiswa. Jawab pertanyaan berikut secara langsung, jelas, dan lengkap.

PENTING:
- Langsung jawab inti pertanyaannya tanpa frase pembuka apapun
- Jawab langsung tanpa kata pembuka seperti "Berdasarkan konteks...", "Berdasarkan informasi yang tersedia...", atau "Menurut dokumen..."
- Gunakan format yang mudah dibaca (numbered list, bullet points jika perlu)
- Jika ada beberapa poin, gunakan penomoran

Informasi yang tersedia:
{context}

Pertanyaan: {question}

Jawaban:""",

                "qa_no_context": """Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan tersebut.

Pertanyaan: {question}

Silakan coba dengan pertanyaan yang lebih spesifik atau hubungi administrator.""",

                "research_mode": """Jawab pertanyaan berikut secara komprehensif dan terstruktur menggunakan informasi akademik di bawah ini.

Informasi yang tersedia:
{context}

Pertanyaan: {question}

Jawaban:""",

                "simple_mode": """{context}

Pertanyaan: {question}
Jawaban:"""
            }
        else:  # English
            return {
                "qa_context": """Based on the following context, answer the question accurately and completely.

Context:
{context}

Question: {question}

Answer:""",

                "qa_no_context": """Sorry, I couldn't find relevant information in the knowledge base to answer that question.

Question: {question}

Please try again with a more specific question or contact the system administrator.""",

                "research_mode": """Based on the following academic documents, provide a comprehensive answer with source references.

Context:
{context}

Question: {question}

Answer (with sources):""",

                "simple_mode": """{context}

Question: {question}
Answer:"""
            }

    def format_prompt(
        self,
        context: str,
        question: str,
        template_type: str = "qa_context"
    ) -> str:
        """
        Format prompt with context and question.

        Args:
            context: Retrieved document context
            question: User question
            template_type: Type of template to use

        Returns:
            Formatted prompt string
        """
        if template_type not in self.templates:
            template_type = "qa_context"

        template = self.templates[template_type]
        return template.format(context=context, question=question)

    def get_available_templates(self) -> List[str]:
        """Get list of available template types."""
        return list(self.templates.keys())
