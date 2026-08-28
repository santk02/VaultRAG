# System prompt for citation-enforced generation
SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided below.

Rules:
1. Every factual sentence must end with a citation in the format: [Source: filename, page N]
2. Use only the filenames and page numbers given in the context passages.
3. If the context does not contain the answer to the question, reply exactly: "I don't have enough information in the provided documents to answer that."
4. Never use outside knowledge or information not present in the context.
5. Never guess or invent page numbers - only use page numbers from the provided context.
6. When multiple sources support the same fact, cite all relevant sources.
7. Keep answers concise and directly responsive to the question.
8. If the information is spread across multiple pages, cite each page separately.

Context passages are provided below. Each passage includes a source identifier with filename and page number.
"""


def build_context_text(chunks: list) -> str:
    """
    Build context text from chunks for the LLM prompt.
    
    Args:
        chunks: List of Chunk objects with filename and page_number
        
    Returns:
        Formatted context string with numbered passages
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_part = (
            f"[{i}] (Source: {chunk.filename}, page {chunk.page_number})\n"
            f"{chunk.text}"
        )
        context_parts.append(context_part)
    
    return "\n\n".join(context_parts)


def build_user_prompt(question: str, context_text: str) -> str:
    """
    Build the user prompt with question and context.
    
    Args:
        question: User's question
        context_text: Formatted context from chunks
        
    Returns:
        Complete user prompt string
    """
    return f"""Context:
{context_text}

Question: {question}"""
