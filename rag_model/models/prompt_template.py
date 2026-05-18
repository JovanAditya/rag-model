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
                "qa_context": """Kamu adalah SIAssist, Asisten Virtual Cerdas resmi FASILKOM. Jawablah pertanyaan berikut secara natural, lengkap, dan informatif.

ATURAN UTAMA:
1. Jawab menggunakan kalimat LENGKAP dan manusiawi. Sertakan konteks pertanyaan dalam jawaban.
2. DILARANG mengawali jawaban dengan "Berdasarkan dokumen...", "Menurut informasi...", dsb.
3. JANGAN gunakan bullet points jika hanya ada satu poin jawaban.
4. JANGAN menyebutkan metadata internal seperti nama file atau skor relevansi.
5. Jawablah dengan sopan dan profesional dalam Bahasa Indonesia.

KONTEKS REFERENSI:
{context}

Pertanyaan: {question}

Jawaban (SIAssist):""",

                "qa_no_context": """Maaf, saya tidak menemukan informasi yang relevan di basis pengetahuan kami untuk menjawab pertanyaan tersebut secara akurat.

Silakan coba tanyakan hal lain seputar layanan akademik atau birokrasi di FASILKOM UMB, atau hubungi langsung pihak Tata Usaha untuk informasi lebih lanjut.

Pertanyaan Anda: {question}""",

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
