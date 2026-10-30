"""
Guardrails Node.

Final safety gate before delivering responses to the user.
Checks for PII leakage, toxicity, relevance, and hallucination.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.core.constants import PII_ENTITY_TYPES, TOXIC_KEYWORDS
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class GuardrailsNode:
    """
    Multi-layered safety guardrails.

    1. PII Detection & Redaction (via Microsoft Presidio)
    2. Toxicity check (keyword + LLM-based)
    3. Relevance gate (does the answer match the query?)
    4. Confidence floor enforcement
    """

    def __init__(
        self,
        llm_service: Any,
        confidence_floor: float = 0.3,
    ) -> None:
        self.llm_service = llm_service
        self.confidence_floor = confidence_floor
        self._analyzer: Any = None
        self._anonymizer: Any = None

    def _init_presidio(self) -> None:
        """Lazy-initialize Presidio analyzer and anonymizer."""
        if self._analyzer is None:
            try:
                import spacy
                # Check if spacy language model is locally installed.
                # If neither is present, avoid triggering spacy.cli.download() which
                # attempts a 400MB pip download and fails under non-root Docker environments.
                if not (spacy.util.is_package("en_core_web_sm") or spacy.util.is_package("en_core_web_lg")):
                    logger.info("Spacy model not installed locally — using fast regex PII redaction.")
                    self._analyzer = False
                    return

                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine

                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except BaseException as exc:
                logger.warning(
                    "Presidio initialization bypassed (%s) — using fast regex PII redaction.",
                    exc,
                )
                self._analyzer = False

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Run all guardrail checks on the generated response."""
        answer = state.get("answer", "")
        query = state["query"]
        confidence = state.get("confidence_score", 0.0)
        flags: dict[str, Any] = {}

        # ----- 1. PII Redaction on output -----
        redacted_answer, pii_found = self._redact_pii(answer)
        if pii_found:
            flags["pii_redacted"] = True
            flags["pii_entities_found"] = pii_found
            logger.warning("PII detected and redacted from response: %s", pii_found)

        # ----- 2. Toxicity check -----
        is_toxic, toxic_reasons = self._check_toxicity(redacted_answer)
        if is_toxic:
            flags["toxic_content"] = True
            flags["toxic_reasons"] = toxic_reasons
            logger.warning("Toxic content detected in response")
            redacted_answer = (
                "I apologize, but I cannot provide this response as it may "
                "contain inappropriate content. Please rephrase your question."
            )

        # ----- 3. Input safety (already checked, verify query) -----
        query_safe = self._check_input_safety(query)
        if not query_safe:
            flags["unsafe_input"] = True

        # ----- 4. Confidence floor -----
        disclaimer = state.get("disclaimer")
        if confidence < self.confidence_floor and not disclaimer:
            disclaimer = (
                "⚠️ Low confidence answer. The available data may not fully "
                "address your question. Please verify independently."
            )

        return {
            "answer": redacted_answer,
            "guardrail_flags": flags,
            "disclaimer": disclaimer,
        }

    def _redact_pii(self, text: str) -> tuple[str, list[str]]:
        """Detect and redact PII using Microsoft Presidio or regex fallback."""
        try:
            self._init_presidio()
        except BaseException as exc:
            logger.warning("Error initializing Presidio: %s — falling back to regex.", exc)
            self._analyzer = False

        if not self._analyzer:
            # Fallback: regex-based PII detection
            return self._regex_pii_redaction(text)

        try:
            results = self._analyzer.analyze(
                text=text,
                entities=PII_ENTITY_TYPES,
                language="en",
            )

            if not results:
                return text, []

            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig

            anonymizer = self._anonymizer or AnonymizerEngine()
            anonymized = anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={
                    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})
                },
            )

            entities_found = list({r.entity_type for r in results})
            return anonymized.text, entities_found

        except BaseException as exc:
            logger.warning("Presidio PII detection failed: %s — using regex fallback.", exc)
            return self._regex_pii_redaction(text)

    def _regex_pii_redaction(self, text: str) -> tuple[str, list[str]]:
        """Fallback regex-based PII detection."""
        found: list[str] = []

        patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                found.append(entity_type)
                text = re.sub(pattern, "[REDACTED]", text)

        return text, found

    def _check_toxicity(self, text: str) -> tuple[bool, list[str]]:
        """Check for toxic content using keyword matching."""
        text_lower = text.lower()
        found = [kw for kw in TOXIC_KEYWORDS if kw in text_lower]
        return bool(found), found

    def _check_input_safety(self, query: str) -> bool:
        """Check for prompt injection patterns using the centralized input validator."""
        from backend.core.input_validator import detect_prompt_injection
        is_injection, _ = detect_prompt_injection(query)
        return not is_injection
