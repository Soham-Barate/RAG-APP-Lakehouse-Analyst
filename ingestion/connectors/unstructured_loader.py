"""
Unstructured data loader for reviews, support tickets, and knowledge base articles.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class UnstructuredLoader:
    """
    Loads unstructured documents from various file formats.

    Supports:
    - JSON / JSONL: Product reviews, support tickets
    - CSV: Tabular data with text columns
    - Markdown: Knowledge base articles
    - Plain text: General documents
    """

    async def load_directory(
        self,
        directory: str | Path,
        document_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load all supported files from a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning("Directory not found: %s", directory)
            return []

        documents: list[dict[str, Any]] = []

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file():
                file_docs = await self.load_file(file_path, document_type)
                documents.extend(file_docs)

        logger.info(
            "Loaded %d documents from %s", len(documents), directory
        )
        return documents

    async def load_file(
        self,
        file_path: str | Path,
        document_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load documents from a single file."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            if suffix == ".json":
                return self._load_json(path, document_type)
            elif suffix == ".jsonl":
                return self._load_jsonl(path, document_type)
            elif suffix == ".csv":
                return self._load_csv(path, document_type)
            elif suffix in (".md", ".markdown"):
                return self._load_markdown(path, document_type)
            elif suffix == ".txt":
                return self._load_text(path, document_type)
            elif suffix == ".parquet":
                return self._load_parquet(path, document_type)
            else:
                logger.debug("Skipping unsupported file: %s", path)
                return []
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return []

    def _load_json(
        self, path: Path, document_type: str | None
    ) -> list[dict[str, Any]]:
        """Load a JSON file (array of objects)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [self._to_document(item, path, document_type) for item in data]
        elif isinstance(data, dict):
            return [self._to_document(data, path, document_type)]
        return []

    def _load_jsonl(
        self, path: Path, document_type: str | None
    ) -> list[dict[str, Any]]:
        """Load a JSONL file (one JSON object per line)."""
        documents: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        documents.append(self._to_document(item, path, document_type))
                    except json.JSONDecodeError:
                        continue
        return documents

    def _load_csv(
        self, path: Path, document_type: str | None
    ) -> list[dict[str, Any]]:
        """Load a CSV file, treating each row as a document."""
        documents: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                content_parts = [f"{k}: {v}" for k, v in row.items() if v]
                content = "\n".join(content_parts)
                documents.append({
                    "content": content,
                    "metadata": {
                        "document_type": document_type or "knowledge_base",
                        "source_file": path.name,
                        **{k: v for k, v in row.items() if v},
                    },
                })
        return documents

    def _load_markdown(
        self, path: Path, document_type: str | None
    ) -> list[dict[str, Any]]:
        """Load a markdown file as a single document."""
        content = path.read_text(encoding="utf-8")
        return [{
            "content": content,
            "metadata": {
                "document_type": document_type or "knowledge_base",
                "source_file": path.name,
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
            },
        }]

    def _load_text(
        self, path: Path, document_type: str | None
    ) -> list[dict[str, Any]]:
        """Load a plain text file."""
        content = path.read_text(encoding="utf-8")
        return [{
            "content": content,
            "metadata": {
                "document_type": document_type or "knowledge_base",
                "source_file": path.name,
            },
        }]

    def _load_parquet(
        self, path: Path, document_type: str | None
    ) -> list[dict[str, Any]]:
        """Load a Parquet file."""
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            documents: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Use text-heavy columns as content
                content_keys = [
                    k for k in row_dict
                    if isinstance(row_dict[k], str) and len(str(row_dict[k])) > 20
                ]
                if content_keys:
                    content = "\n".join(f"{k}: {row_dict[k]}" for k in content_keys)
                else:
                    content = "\n".join(f"{k}: {v}" for k, v in row_dict.items())

                documents.append({
                    "content": content,
                    "metadata": {
                        "document_type": document_type or self._infer_type(path.stem),
                        "source_file": path.name,
                        **{k: str(v) for k, v in row_dict.items()
                           if k not in content_keys},
                    },
                })
            return documents
        except ImportError:
            logger.warning("pandas not installed, cannot load Parquet")
            return []

    def _to_document(
        self,
        item: dict[str, Any],
        path: Path,
        document_type: str | None,
    ) -> dict[str, Any]:
        """Convert a raw dict to a standardized document."""
        # Try common content field names
        content = ""
        for key in ["content", "text", "body", "description", "review_text",
                     "message", "comment", "summary"]:
            if key in item and item[key]:
                content = str(item[key])
                break

        if not content:
            content = "\n".join(f"{k}: {v}" for k, v in item.items())

        return {
            "content": content,
            "metadata": {
                "document_type": (
                    document_type
                    or item.get("type")
                    or self._infer_type(path.stem)
                ),
                "source_file": path.name,
                **{k: str(v) for k, v in item.items() if k != "content"},
            },
        }

    @staticmethod
    def _infer_type(filename: str) -> str:
        """Infer document type from filename."""
        name = filename.lower()
        if "product" in name:
            return "product"
        elif "order" in name:
            return "order"
        elif "review" in name:
            return "review"
        elif "ticket" in name or "support" in name:
            return "ticket"
        elif "customer" in name:
            return "customer"
        return "knowledge_base"
