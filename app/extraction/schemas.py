from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel


class LoanAgreementExtraction(BaseModel):
    """Validated fields extracted from a loan agreement."""

    borrower_name: Optional[str] = None
    loan_amount: Optional[Decimal] = None
    interest_rate: Optional[Decimal] = None
    maturity_date: Optional[date] = None


EXTRACTION_SCHEMAS = {
    "loan_agreement": LoanAgreementExtraction,
}
