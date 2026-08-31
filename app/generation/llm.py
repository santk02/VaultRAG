from typing import List

import litellm

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, build_context_text, build_user_prompt
from app.models import Chunk


class LLMGenerator:
    """
    LLM generator using LiteLLM to interface with Ollama models.

    Supports both offline (Ollama) and cloud (Claude API) modes via configuration.
    """

    def __init__(self):
        """Initialize generator with settings."""
        self.mode = settings.mode
        self.ollama_base_url = settings.ollama_base_url
        self.offline_model = settings.offline_model
        self.alternative_model = settings.alternative_model
        self.cloud_model = settings.cloud_model

        # Configure LiteLLM for Ollama; cloud calls carry their own base URL via the
        # "anthropic/" model prefix and don't use this setting.
        litellm.api_base = self.ollama_base_url

    def get_model_name(self, model_override: str = None, mode: str = None) -> str:
        """
        Get the appropriate model name based on mode and configuration.

        Args:
            model_override: Optional model name override

        Returns:
            Model identifier for LiteLLM
        """
        if model_override:
            return model_override

        selected_mode = mode or self.mode
        if selected_mode == "cloud":
            if not settings.anthropic_api_key:
                raise ValueError("Cloud mode requires ANTHROPIC_API_KEY")
            return f"anthropic/{self.cloud_model}"  # LiteLLM provider-prefixed model id
        else:
            return f"ollama/{self.offline_model}"  # zero data egress — routes to local Ollama

    async def generate(
        self,
        question: str,
        chunks: List[Chunk],
        model_override: str = None,
        mode: str = None,
    ) -> tuple[str, str, float]:
        """
        Generate an answer to the question using the retrieved chunks.

        Args:
            question: User's question
            chunks: Retrieved chunks with context
            model_override: Optional model name override

        Returns:
            Tuple of (answer, model_used, latency_ms)
        """
        import time

        start_time = time.time()
        model_name = self.get_model_name(model_override, mode)

        # Build context and prompts
        context_text = build_context_text(chunks)
        user_prompt = build_user_prompt(question, context_text)

        try:
            # Call LLM via LiteLLM — num_retries handles transient failures,
            # same call shape works for both cloud and offline model_name values
            response = await litellm.acompletion(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                num_retries=2,
                timeout=60,
            )

            answer = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000

            return answer, model_name, latency_ms

        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")

    async def complete(
        self, prompt: str, model_override: str = None, mode: str = None
    ) -> tuple[str, str, float]:
        """
        Raw single-turn completion without the citation-enforcing SYSTEM_PROMPT.

        Used by extraction (app/extraction/router.py), which needs bare JSON output —
        the Q&A system prompt would make the model wrap its answer in
        "[Source: ..., page N]" citations and break json.loads() downstream.
        """
        import time

        start_time = time.time()
        model_name = self.get_model_name(model_override, mode)

        try:
            response = await litellm.acompletion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                num_retries=2,
                timeout=60,
            )
            answer = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000
            return answer, model_name, latency_ms
        except Exception as e:
            raise RuntimeError(f"LLM completion failed: {str(e)}")


# Global generator instance
_generator: LLMGenerator = None


def get_generator() -> LLMGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator


async def generate_answer(
    question: str, chunks: List[Chunk], model_override: str = None, mode: str = None
) -> tuple[str, str, float]:
    """
    Convenience function to generate an answer.

    Args:
        question: User's question
        chunks: Retrieved chunks
        model_override: Optional model name override

    Returns:
        Tuple of (answer, model_used, latency_ms)
    """
    generator = get_generator()
    return await generator.generate(question, chunks, model_override, mode)
