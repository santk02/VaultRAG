import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.extraction.schemas import EXTRACTION_SCHEMAS
from app.generation.llm import get_generator
from app.models import ExtractionRequest, ExtractionResponse

router = APIRouter(prefix="/v1/extract", tags=["extraction"])


@router.post("", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest) -> ExtractionResponse:
    """Extract and validate a supported structured schema from document text."""
    schema = EXTRACTION_SCHEMAS.get(request.schema_name)
    if schema is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported schema: {request.schema_name}",
        )

    started = time.time()
    # Reuses generate() (built for Q&A) as a generic JSON-extraction call by passing
    # an empty chunk list — the schema's own JSON Schema is embedded in the prompt so
    # the model knows the exact field names/types to return.
    prompt = (
        "Extract the requested fields from the document. Return JSON only, "
        f"matching this schema: {schema.model_json_schema()}\n\n"
        f"Document:\n{request.text}"
    )
    generator = get_generator()
    model_name = generator.get_model_name(mode=settings.mode)
    try:
        # complete(), not generate() — generate() would apply the citation-enforcing
        # Q&A system prompt and corrupt the JSON output.
        response = await generator.complete(prompt=prompt, mode=settings.mode)
        payload: Any = json.loads(response[0])
        validated = schema.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Invalid extraction output: {exc}"
        ) from exc

    return ExtractionResponse(
        extracted_data=validated.model_dump(mode="json"),
        model_used=model_name,
        latency_ms=(time.time() - started) * 1000,
    )
