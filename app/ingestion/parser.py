from typing import List
from dataclasses import dataclass
import fitz  # PyMuPDF
from docx import Document
import hashlib


@dataclass
class Page:
    """A page from a document with text and page number."""

    page_number: int
    text: str


class ParserError(Exception):
    """Custom exception for parsing errors."""

    pass


def parse_pdf(file_path: str) -> List[Page]:
    """
    Parse a PDF file and extract text with page numbers.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of Page objects with page_number and text

    Raises:
        ParserError: If PDF has no text layer or cannot be parsed
    """
    try:
        doc = fitz.open(file_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if not text.strip():
                continue  # Skip empty pages (common on cover/divider pages)

            pages.append(
                Page(page_number=page_num + 1, text=text)
            )  # PyMuPDF is 0-indexed; citations are 1-indexed

        doc.close()

        if not pages:
            # Every page came back empty — almost always a scanned PDF with no OCR text
            # layer. Fail loudly per the blueprint rather than silently indexing nothing.
            raise ParserError(
                "PDF contains no extractable text. "
                "This may be a scanned document without an OCR text layer."
            )

        return pages

    except Exception as e:
        if isinstance(e, ParserError):
            raise
        raise ParserError(f"Failed to parse PDF: {str(e)}")


def parse_docx(file_path: str) -> List[Page]:
    """
    Parse a DOCX file and extract text with section numbers as page proxies.

    Note: DOCX files don't have real page numbers, so we use section/paragraph
    groupings as a proxy. This is documented in the README as a limitation.

    Args:
        file_path: Path to the DOCX file

    Returns:
        List of Page objects with section_number as page_number proxy
    """
    try:
        doc = Document(file_path)
        pages = []
        current_page = 1
        current_text = []
        chars_per_page = (
            3000  # Approximate characters per page — DOCX has no page metadata,
        )
        # so "page_number" here is a proxy, not a real page (documented limitation)

        for para in doc.paragraphs:
            if not para.text.strip():
                continue

            current_text.append(para.text)

            # Check if we've accumulated enough for a "page"
            if sum(len(p) for p in current_text) >= chars_per_page:
                pages.append(
                    Page(page_number=current_page, text="\n".join(current_text))
                )
                current_page += 1
                current_text = []

        # Add remaining text as last page
        if current_text:
            pages.append(Page(page_number=current_page, text="\n".join(current_text)))

        if not pages:
            raise ParserError("DOCX file contains no extractable text")

        return pages

    except Exception as e:
        raise ParserError(f"Failed to parse DOCX: {str(e)}")


def compute_content_hash(text: str) -> str:
    """
    Compute SHA-256 hash of text content for deduplication.

    Args:
        text: Text content to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()


def parse_document(file_path: str, file_type: str) -> tuple[List[Page], str]:
    """
    Parse a document based on its file type.

    Args:
        file_path: Path to the document
        file_type: File extension ('pdf', 'docx', etc.)

    Returns:
        Tuple of (pages list, full text for hashing)
    """
    file_type = file_type.lower().lstrip(".")

    if file_type == "pdf":
        pages = parse_pdf(file_path)
    elif file_type == "docx":
        pages = parse_docx(file_path)
    else:
        raise ParserError(f"Unsupported file type: {file_type}")

    full_text = "\n".join(page.text for page in pages)
    content_hash = compute_content_hash(
        full_text
    )  # used by indexer.py to dedupe re-uploads

    return pages, content_hash
