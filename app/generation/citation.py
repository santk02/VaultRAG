import re
from typing import List, Tuple

from app.models import Chunk


class CitationChecker:
    """
    Verifies that citations in generated answers are valid and grounded in retrieved chunks.

    This is a critical differentiator - most RAG systems trust the prompt, but we verify
    that every citation actually points to a chunk that was retrieved.
    """

    CITATION_PATTERN = r"\[Source:\s*([^,]+),\s*page\s*(\d+)\]"

    def extract_citations(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract all citations from text.

        Args:
            text: Text containing citations

        Returns:
            List of (filename, page_number) tuples
        """
        citations = re.findall(self.CITATION_PATTERN, text)
        return [(filename.strip(), int(page)) for filename, page in citations]

    def validate_citations(
        self, answer: str, chunks: List[Chunk]
    ) -> Tuple[bool, List[str], List[Tuple[str, int]]]:
        """
        Validate that all citations in the answer reference actual retrieved chunks.

        Args:
            answer: Generated answer text
            chunks: Retrieved chunks that were used as context

        Returns:
            Tuple of (is_valid, invalid_sentences, valid_citations)
        """
        citations = self.extract_citations(answer)
        valid_citations = []
        invalid_citations = []

        # Build set of valid (filename, page) combinations from chunks
        valid_sources = {(chunk.filename, chunk.page_number) for chunk in chunks}

        for filename, page_num in citations:
            if (filename, page_num) in valid_sources:
                valid_citations.append((filename, page_num))
            else:
                invalid_citations.append((filename, page_num))

        # Check for sentences without citations
        sentences = self._split_sentences(answer)
        sentences_without_citations = []

        for sentence in sentences:
            if self._is_factual_sentence(sentence):
                sentence_citations = self.extract_citations(sentence)
                if not sentence_citations:
                    sentences_without_citations.append(sentence)

        is_valid = len(invalid_citations) == 0 and len(sentences_without_citations) == 0

        return is_valid, sentences_without_citations, valid_citations

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLP
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\[])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_factual_sentence(self, sentence: str) -> bool:
        """
        Determine if a sentence requires a citation.

        Factual sentences (statements of fact, data, specific claims) need citations.
        Transitional sentences (questions, opinions, boilerplate) may not.
        """
        sentence = sentence.strip().lower()

        # Skip empty sentences
        if not sentence:
            return False

        # Skip opinion phrases
        opinion_phrases = [
            "i believe",
            "i think",
            "in my opinion",
            "it seems",
            "it appears",
            "possibly",
        ]
        if any(phrase in sentence for phrase in opinion_phrases):
            return False

        # Skip refusal statements
        if "don't have enough information" in sentence:
            return False

        # Skip questions
        if sentence.endswith("?"):
            return False

        # Sentences with numbers, dates, or specific facts need citations
        factual_indicators = [
            r"\d+",  # numbers
            r"\b\d{4}\b",  # years
            r"percent",  # percentages
            r"dollar",  # currency
            r"according to",  # references
        ]

        if any(re.search(pattern, sentence) for pattern in factual_indicators):
            return True

        # Default: factual sentences should have citations
        return len(sentence) > 20  # Longer sentences are more likely to be factual

    def strip_invalid_citations(self, answer: str, chunks: List[Chunk]) -> str:
        """
        Remove citations that don't reference actual chunks.

        Args:
            answer: Generated answer text
            chunks: Retrieved chunks

        Returns:
            Answer text with invalid citations removed
        """
        is_valid, _, valid_citations = self.validate_citations(answer, chunks)

        if is_valid:
            return answer

        # Remove a whole sentence when its only citation is invalid. Keeping
        # the prose would turn a rejected citation into an uncited claim.
        valid_citation_strings = {
            f"[Source: {filename}, page {page}]" for filename, page in valid_citations
        }

        sentences = self._split_sentences(answer)
        cleaned_sentences = []
        for sentence in sentences:
            sentence_citations = self.extract_citations(sentence)
            if sentence_citations and any(
                f"[Source: {filename}, page {page}]" not in valid_citation_strings
                for filename, page in sentence_citations
            ):
                continue
            cleaned_sentences.append(sentence)

        return " ".join(cleaned_sentences).strip()


# Global citation checker instance
_checker: CitationChecker = None


def get_citation_checker() -> CitationChecker:
    """Get or create the global citation checker instance."""
    global _checker
    if _checker is None:
        _checker = CitationChecker()
    return _checker
