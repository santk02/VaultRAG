import pytest
from app.ingestion.parser import parse_document, ParserError, compute_content_hash
from app.ingestion.chunker import create_chunks


def test_compute_content_hash():
    """Test content hash computation — this hash is what indexer.py uses to detect
    and skip re-uploads of the same document."""
    text1 = "This is a test document."
    text2 = "This is a test document."
    text3 = "This is different."

    hash1 = compute_content_hash(text1)
    hash2 = compute_content_hash(text2)
    hash3 = compute_content_hash(text3)

    assert hash1 == hash2  # Same content should produce same hash
    assert hash1 != hash3  # Different content should produce different hash
    assert len(hash1) == 64  # SHA-256 produces 64-character hex string


def test_chunker_basic():
    """Test basic chunking functionality."""
    from app.ingestion.parser import Page

    pages = [
        Page(page_number=1, text="This is the first page with some content."),
        Page(page_number=2, text="This is the second page with more content."),
    ]

    doc_id = "test_doc_123"
    chunks = create_chunks(pages, doc_id)

    assert len(chunks) > 0
    assert all(chunk.doc_id == doc_id for chunk in chunks)
    assert all(chunk.page_number in [1, 2] for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_chunker_page_preservation():
    """Test that page numbers are preserved through chunking — critical for citation
    accuracy per VAULTRAG_BLUEPRINT.md section 7."""
    from app.ingestion.parser import Page

    pages = [
        Page(page_number=5, text="Content from page 5."),
        Page(page_number=10, text="Content from page 10."),
    ]

    chunks = create_chunks(pages, "test_doc")

    # All chunks should preserve their original page numbers
    page_5_chunks = [c for c in chunks if c.page_number == 5]
    page_10_chunks = [c for c in chunks if c.page_number == 10]

    assert len(page_5_chunks) > 0
    assert len(page_10_chunks) > 0
    assert all(c.page_number == 5 for c in page_5_chunks)
    assert all(c.page_number == 10 for c in page_10_chunks)


def test_parser_error_handling():
    """Test that parser raises appropriate errors."""
    # Test with non-existent file
    with pytest.raises(ParserError):
        parse_document("nonexistent.pdf", "pdf")

    # Test with unsupported file type
    with pytest.raises(ParserError):
        parse_document("test.xyz", "xyz")


def test_chunk_token_counting():
    """Test that token counting works."""
    from app.ingestion.parser import Page

    pages = [Page(page_number=1, text="This is a test sentence with some words.")]

    chunks = create_chunks(pages, "test_doc")

    # At least one chunk should be created
    assert len(chunks) > 0

    # Token count should be positive
    for chunk in chunks:
        assert chunk.token_count > 0
        # Rough estimate - token count should be reasonable
        assert chunk.token_count <= len(chunk.text)  # At most one token per char
