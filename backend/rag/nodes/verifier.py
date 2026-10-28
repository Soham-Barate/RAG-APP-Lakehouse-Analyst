"""
Answer Verification Node.

After the generator produces an answer, this node asks the LLM to verify
every claim against the source documents.  Claims that cannot be traced
to a source get flagged, and the confidence score is adjusted downward.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from backend.rag.prompts.templates import VERIFICATION_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """
    Post-generation answer verification.

    For each claim in the answer, checks whether:
    - SUPPORTED:  The claim is directly backed by a source document.
    - PARTIAL:    The claim is partially supported (some details added).
    - UNSUPPORTED: The claim has no basis in the provided sources.

    The confidence score is adjusted based on the verification ratio.
    """

    def __init__(
        self,
        llm_service: Any,
        min_support_ratio: float = 0.6,
    ) -> None:
        self.llm_service = llm_service
        self.min_support_ratio = min_support_ratio

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Verify the generated answer against source context."""
        start = time.perf_counter()

        answer = state.get("answer", "")
        context = state.get("compressed_context", "")
        confidence = state.get("confidence_score", 0.5)

        if not answer or not context:
            return {}

        try:
            prompt = VERIFICATION_PROMPT.format(
                answer=answer,
                context=context,
            )

            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
                json_mode=True,
            )

            result = json.loads(response)

            supported = result.get("supported_claims", 0)
            partial = result.get("partial_claims", 0)
            unsupported = result.get("unsupported_claims", 0)
            total = supported + partial + unsupported

            if total > 0:
                support_ratio = (supported + 0.5 * partial) / total
            else:
                support_ratio = 1.0

            # Adjust confidence based on verification
            verified_confidence = confidence * (0.5 + 0.5 * support_ratio)

            # Add disclaimer if support ratio is low
            disclaimer = state.get("disclaimer")
            if support_ratio < self.min_support_ratio and not disclaimer:
                disclaimer = (
                    "⚠️ Some claims in this answer could not be fully verified "
                    "against the source documents. Please cross-check important "
                    "facts independently."
                )

            latency_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Verification: %d/%d supported (%.0f%%), confidence %.2f→%.2f (%.1fms)",
                supported,
                total,
                support_ratio * 100,
                confidence,
                verified_confidence,
                latency_ms,
            )

            return {
                "confidence_score": round(verified_confidence, 4),
                "disclaimer": disclaimer,
                "guardrail_flags": {
                    **state.get("guardrail_flags", {}),
                    "verification": {
                        "supported": supported,
                        "partial": partial,
                        "unsupported": unsupported,
                        "support_ratio": round(support_ratio, 4),
                        "issues": result.get("issues", []),
                    },
                },
            }

        except Exception as exc:
            logger.warning("Answer verification failed, skipping: %s", exc)
            return {}
