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
    prompt = (
        "Extract the requested fields from the document. Return JSON only, "
        f"matching this schema: {schema.model_json_schema()}\n\n"
        f"Document:\n{request.text}"
    )
    generator = get_generator()
    model_name = generator.get_model_name(mode=settings.mode)
    try:
        response = await generator.generate(
            question=prompt,
            chunks=[],
            mode=settings.mode,
        )
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
