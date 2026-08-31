from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel


class LoanAgreementExtraction(BaseModel):
    """Validated fields extracted from a loan agreement.

    This is also the target schema for the QLoRA fine-tuning task in fine_tuning/ —
    the blueprint's Phase 5 narrow-task example (extract these 4 fields from raw text).
    All fields are Optional since the model may not find every field in a given document.
    """

    borrower_name: Optional[str] = None
    loan_amount: Optional[Decimal] = None
    interest_rate: Optional[Decimal] = None
    maturity_date: Optional[date] = None


# Registry of supported extraction schemas by name — POST /v1/extract looks up
# schema_name here; add new document types by adding a BaseModel + registry entry.
EXTRACTION_SCHEMAS = {
    "loan_agreement": LoanAgreementExtraction,
}
