"""Query optimization for better RAG retrieval performance."""

from typing import List, Dict, Any
import re

class QueryOptimizer:
    """Optimize queries for better retrieval performance."""

    def __init__(self):
        self.stop_words = {
            'apa', 'bagaimana', 'berapa', 'siapa', 'dimana', 'kapan', 'mengapa', 'adakah',
            'tentang', 'mengenai', 'berkaitan', 'terkait', 'hal', 'ini'
        }

        self.academic_terms = {
            'universitas mercu buana': ['umb', 'universitas mercu buana'],
            'visi': ['tujuan', 'cita-cita', 'maksud'],
            'misi': ['tujuan', 'fungsi', 'peran'],
            'fakultas teknik': ['teknik', 'ft'],
            'program studi': ['prodi', 'jurusan'],
            'krs': ['kartu rekam studi', 'registrasi'],
            'skripsi': ['tugas akhir', 'thesis'],
            'ujian': ['test', 'assessment'],
            'mahasiswa': ['student']
        }

    def optimize_query(self, original_query: str) -> List[str]:
        """
        Generate multiple query variations for better retrieval.

        Returns:
            List of query variations
        """
        variations = []
        query_lower = original_query.lower()

        # 1. Original query
        variations.append(original_query)

        # 2. Remove stop words
        words = [word for word in original_query.split() if word.lower() not in self.stop_words]
        if words:
            variations.append(' '.join(words))

        # 3. Academic term expansion
        expanded_query = query_lower
        for term, expansions in self.academic_terms.items():
            if term in query_lower:
                for expansion in expansions:
                    if expansion not in query_lower:
                        expanded_query = expanded_query.replace(term, f'{term} {expansion}')

        if expanded_query != query_lower:
            variations.append(expanded_query.title())  # Maintain proper capitalization

        # 4. Key phrase extraction
        key_phrases = self._extract_key_phrases(original_query)
        variations.extend(key_phrases)

        # 5. Add contextual terms
        if 'universitas' in query_lower and 'mercubuana' not in query_lower:
            variations.append(original_query + ' Universitas Mercu Buana')

        return list(set(variations))  # Remove duplicates

    def _extract_key_phrases(self, query: str) -> List[str]:
        """Extract key academic phrases from query."""
        phrases = []

        # Common academic patterns
        patterns = [
            r'visi.*misi',
            r'program.*studi',
            r'fakultas.*teknik',
            r'jurusan.*[a-z]+',
            r'syarat.*[a-z]+',
            r'prosedur.*[a-z]+',
            r'jadwal.*[a-z]+',
            r'biaya.*[a-z]+'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            phrases.extend(matches)

        return phrases

    def generate_contextual_query(self, original_query: str, retrieved_context: str) -> str:
        """
        Generate contextual query based on retrieved documents.

        Args:
            original_query: Original user query
            retrieved_context: Context from retrieved documents

        Returns:
            Enhanced contextual query
        """
        # Extract keywords from retrieved context
        context_words = set()
        if retrieved_context:
            context_words.update(re.findall(r'\b\w+\b', retrieved_context.lower()))

        # Combine with original query keywords
        query_words = set(re.findall(r'\b\w+\b', original_query.lower()))
        all_words = query_words.union(context_words)

        # Filter and prioritize
        filtered_words = [
            word for word in all_words
            if word not in self.stop_words and len(word) > 2
        ]

        # Prioritize words that appeared in both query and context
        prioritized_words = [
            word for word in filtered_words
            if word in query_words and word in context_words
        ]

        # Add remaining filtered words
        remaining_words = [
            word for word in filtered_words
            if word not in prioritized_words
        ]

        # Combine prioritized words first
        enhanced_query = ' '.join(prioritized_words[:3] + remaining_words[:2])

        return enhanced_query if enhanced_query else original_query