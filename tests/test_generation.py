from app.generation.prompts import build_context_text, build_user_prompt, SYSTEM_PROMPT
from app.generation.citation import CitationChecker
from app.models import Chunk


def test_build_context_text():
    """Test context text building from chunks — this exact "[i] (Source: ..., page N)" format
    is what the LLM is instructed to echo back in citations, and citation.py parses it.
    """
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            chunk_index=0,
            page_number=1,
            text="This is content from page 1.",
            token_count=10,
            filename="document.pdf",
        ),
        Chunk(
            chunk_id="chunk_2",
            doc_id="doc_1",
            chunk_index=1,
            page_number=2,
            text="This is content from page 2.",
            token_count=10,
            filename="document.pdf",
        ),
    ]

    context = build_context_text(chunks)

    assert "[1]" in context
    assert "[2]" in context
    assert "Source: document.pdf, page 1" in context
    assert "Source: document.pdf, page 2" in context
    assert "This is content from page 1." in context
    assert "This is content from page 2." in context


def test_build_user_prompt():
    """Test user prompt building."""
    question = "What is the content?"
    context = "[1] (Source: doc.pdf, page 1)\nSome content"

    prompt = build_user_prompt(question, context)

    assert "Context:" in prompt
    assert question in prompt
    assert context in prompt


def test_system_prompt():
    """Test that system prompt contains required rules."""
    assert "citation" in SYSTEM_PROMPT.lower()
    assert "source:" in SYSTEM_PROMPT
    assert "page" in SYSTEM_PROMPT.lower()
    assert "enough information" in SYSTEM_PROMPT.lower()


def test_citation_extraction():
    """Test citation extraction from text."""
    checker = CitationChecker()

    text = "According to the document [Source: report.pdf, page 5], the value is 42."
    citations = checker.extract_citations(text)

    assert len(citations) == 1
    assert citations[0] == ("report.pdf", 5)


def test_citation_extraction_multiple():
    """Test extraction of multiple citations."""
    checker = CitationChecker()

    text = """
    The first claim [Source: doc1.pdf, page 3] states X.
    The second claim [Source: doc2.pdf, page 7] states Y.
    """
    citations = checker.extract_citations(text)

    assert len(citations) == 2
    assert ("doc1.pdf", 3) in citations
    assert ("doc2.pdf", 7) in citations


def test_citation_validation_valid():
    """Test validation of valid citations."""
    checker = CitationChecker()

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            chunk_index=0,
            page_number=5,
            text="Content",
            token_count=10,
            filename="report.pdf",
        )
    ]

    answer = "The answer is [Source: report.pdf, page 5]."
    is_valid, _, valid_citations = checker.validate_citations(answer, chunks)

    assert is_valid
    assert len(valid_citations) == 1
    assert valid_citations[0] == ("report.pdf", 5)


def test_citation_validation_invalid():
    """Test validation of invalid citations — a page number the chunk set never retrieved
    is treated as a hallucinated citation, per the blueprint's citation-verification requirement.
    """
    checker = CitationChecker()

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            chunk_index=0,
            page_number=5,
            text="Content",
            token_count=10,
            filename="report.pdf",
        )
    ]

    answer = "The answer is [Source: report.pdf, page 10]."  # Page 10 not in chunks
    is_valid, _, valid_citations = checker.validate_citations(answer, chunks)

    assert not is_valid
    assert len(valid_citations) == 0


def test_citation_strip_invalid():
    """Test stripping invalid citations."""
    checker = CitationChecker()

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            chunk_index=0,
            page_number=5,
            text="Content",
            token_count=10,
            filename="report.pdf",
        )
    ]

    answer = (
        "Valid [Source: report.pdf, page 5] and invalid [Source: report.pdf, page 10]."
    )
    cleaned = checker.strip_invalid_citations(answer, chunks)

    assert "[Source: report.pdf, page 5]" in cleaned
    assert "[Source: report.pdf, page 10]" not in cleaned


def test_sentence_splitting():
    """Test sentence splitting."""
    checker = CitationChecker()

    text = "First sentence. Second sentence! Third question?"
    sentences = checker._split_sentences(text)

    assert len(sentences) == 3
    assert "First sentence" in sentences[0]
    assert "Second sentence" in sentences[1]
    assert "Third question" in sentences[2]


def test_factual_sentence_detection():
    """Test detection of factual sentences."""
    checker = CitationChecker()

    # Factual sentence with number
    assert checker._is_factual_sentence("The value is 42.")

    # Factual sentence with year
    assert checker._is_factual_sentence("This happened in 2023.")

    # Opinion (should not require citation)
    assert not checker._is_factual_sentence("I believe this is true.")

    # Question (should not require citation)
    assert not checker._is_factual_sentence("What is the value?")

    # Short sentence (might not be factual)
    assert not checker._is_factual_sentence("Yes.")
