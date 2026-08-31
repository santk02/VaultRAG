from typing import Optional
from guardrails import Guard
from guardrails.hub import ToxicLanguage, PIIFilter
from app.config import settings


class OutputGuard:
    """
    Output validation using Guardrails AI.

    Validates generated answers for:
    - Toxic content
    - PII leakage
    - Empty responses
    - Expected format
    """

    def __init__(self):
        """Initialize guards."""
        self.toxic_guard = None
        self.pii_guard = None

        # Initialize guards only if an API key is provided — same guarded-init pattern
        # as Langfuse in tracing.py, so validate_all() is a safe no-op in dev/CI.
        if settings.guardrails_api_key:
            try:
                self.toxic_guard = Guard().use_many(ToxicLanguage, threshold=0.5)
                self.pii_guard = Guard().use_many(
                    PIIFilter, pii_entities=["EMAIL", "PHONE", "SSN", "CREDIT_CARD"]
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Guardrails AI: {e}")

    def validate_toxicity(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Check if text contains toxic content.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        if not self.toxic_guard:
            return True, None  # Skip validation if guard not initialized

        try:
            result = self.toxic_guard.parse(text)
            if result.validation_passed:
                return True, None
            else:
                return False, "Response contains potentially toxic content"
        except Exception as e:
            print(f"Toxicity check failed: {e}")
            return (
                True,
                None,
            )  # Fail open — an unrelated guard error must not block a valid answer

    def validate_pii(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Check if text contains PII that should be redacted.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        if not self.pii_guard:
            return True, None  # Skip validation if guard not initialized

        try:
            result = self.pii_guard.parse(text)
            if result.validation_passed:
                return True, None
            else:
                return False, "Response contains PII that should be redacted"
        except Exception as e:
            print(f"PII check failed: {e}")
            return True, None  # Fail open if validation fails

    def validate_format(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate that response format is expected.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for empty response
        if not text or not text.strip():
            return False, "Response is empty"

        # Check for refusal pattern (this is expected and valid)
        if "don't have enough information" in text.lower():
            return True, None

        # Check for minimum length
        if len(text.strip()) < 10:
            return False, "Response is too short"

        return True, None

    def validate_all(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Run all validations on the text.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Format validation
        is_valid, error = self.validate_format(text)
        if not is_valid:
            return False, error

        # Toxicity validation
        is_valid, error = self.validate_toxicity(text)
        if not is_valid:
            return False, error

        # PII validation
        is_valid, error = self.validate_pii(text)
        if not is_valid:
            return False, error

        return True, None


# Global guard instance
_guard: OutputGuard = None


def get_output_guard() -> OutputGuard:
    """Get or create the global output guard instance."""
    global _guard
    if _guard is None:
        _guard = OutputGuard()
    return _guard
