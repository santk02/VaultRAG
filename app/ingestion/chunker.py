from typing import List
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.ingestion.parser import Page
from app.config import settings
import tiktoken


@dataclass
class Chunk:
    """A text chunk with metadata for citation tracking."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    page_number: int
    text: str
    token_count: int


class Chunker:
    """Text chunker that preserves page numbers for citation accuracy."""

    def __init__(self):
        """Initialize chunker with settings from config."""
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.get_encoding(
                "cl100k_base"
            )  # used only for length measurement, not real tokenization for any specific LLM
        except Exception:
            # Fallback to character count if tiktoken fails
            self.encoding = None
            self._count_tokens = self._count_characters

        # Separator priority (paragraph > line > sentence > word > char) means the
        # splitter respects paragraph/sentence boundaries whenever it can fit within
        # chunk_size, only falling back to a hard character break as a last resort.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return self._count_characters(text)

    def _count_characters(self, text: str) -> int:
        """Fallback character counting."""
        return len(text)

    def chunk_pages(self, pages: List[Page], doc_id: str) -> List[Chunk]:
        """
        Chunk document pages while preserving page numbers.

        This is critical for citation accuracy - every chunk must carry
        its original page number through the entire pipeline.

        Args:
            pages: List of Page objects from parser
            doc_id: Document ID for chunk metadata

        Returns:
            List of Chunk objects with page numbers preserved
        """
        chunks = []
        chunk_index = 0

        for page in pages:
            # Chunk each page independently (not the whole document at once) so every
            # resulting chunk can be labeled with a single, unambiguous page_number.
            page_chunks = self.splitter.split_text(page.text)

            for chunk_text in page_chunks:
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                token_count = self._count_tokens(chunk_text)

                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    page_number=page.page_number,  # Critical: preserve page number
                    text=chunk_text,
                    token_count=token_count,
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks


def create_chunks(pages: List[Page], doc_id: str) -> List[Chunk]:
    """
    Convenience function to create chunks from pages.

    Args:
        pages: List of Page objects
        doc_id: Document ID

    Returns:
        List of Chunk objects
    """
    chunker = Chunker()
    return chunker.chunk_pages(pages, doc_id)
