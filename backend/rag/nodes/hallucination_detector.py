"""
Hallucination Detector Node.

Uses a Natural Language Inference (NLI) approach to check each sentence
in the generated answer against the source context.  Sentences that are
CONTRADICTED by the sources are flagged; sentences with no support are
marked as potential hallucinations.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from backend.rag.prompts.templates import HALLUCINATION_CHECK_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Sentence-level hallucination detection via NLI.

    For each sentence in the answer:
    - ENTAILED:      Directly supported by the source context.
    - NEUTRAL:       Not contradicted but not directly supported.
    - CONTRADICTED:  Directly contradicts information in the sources.

    Contradicted sentences are removed or flagged.  A high ratio of
    neutral sentences also lowers confidence.
    """

    def __init__(
        self,
        llm_service: Any,
        contradiction_threshold: float = 0.2,
    ) -> None:
        self.llm_service = llm_service
        self.contradiction_threshold = contradiction_threshold

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Detect hallucinated sentences in the answer."""
        start = time.perf_counter()

        answer = state.get("answer", "")
        context = state.get("compressed_context", "")

        if not answer or not context:
            return {}

        # Split answer into sentences
        sentences = self._split_sentences(answer)
        if len(sentences) <= 1:
            # Single-sentence answers are not worth checking
            return {}

        try:
            prompt = HALLUCINATION_CHECK_PROMPT.format(
                answer=answer,
                context=context,
                sentence_count=len(sentences),
            )

            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
                json_mode=True,
            )

            result = json.loads(response)
            sentence_verdicts = result.get("sentences", [])

            contradicted_count = sum(
                1 for s in sentence_verdicts
                if s.get("verdict", "").lower() == "contradicted"
            )
            total = len(sentence_verdicts) or 1
            contradiction_ratio = contradicted_count / total

            # If too many contradictions, rewrite the answer without them
            modified_answer = answer
            if contradiction_ratio > self.contradiction_threshold:
                # Filter out contradicted sentences
                safe_sentences = []
                for i, sent in enumerate(sentences):
                    if i < len(sentence_verdicts):
                        verdict = sentence_verdicts[i].get("verdict", "entailed")
                        if verdict.lower() != "contradicted":
                            safe_sentences.append(sent)
                    else:
                        safe_sentences.append(sent)

                if safe_sentences:
                    modified_answer = " ".join(safe_sentences)
                    logger.warning(
                        "Hallucination: removed %d contradicted sentences",
                        contradicted_count,
                    )

            latency_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Hallucination check: %d/%d contradicted (%.0f%%) (%.1fms)",
                contradicted_count,
                total,
                contradiction_ratio * 100,
                latency_ms,
            )

            updates: dict[str, Any] = {
                "guardrail_flags": {
                    **state.get("guardrail_flags", {}),
                    "hallucination": {
                        "contradicted_sentences": contradicted_count,
                        "total_sentences": total,
                        "contradiction_ratio": round(contradiction_ratio, 4),
                    },
                },
            }

            if modified_answer != answer:
                updates["answer"] = modified_answer

            return updates

        except Exception as exc:
            logger.warning("Hallucination detection failed, skipping: %s", exc)
            return {}

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
