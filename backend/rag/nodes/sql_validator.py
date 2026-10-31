"""
AST-based SQL Parser and Security Validator.

Uses SQLGlot to parse queries into an Abstract Syntax Tree (AST)
and verify strict read-only execution, table/column allowlisting,
preventing SQL injection, multi-statement attacks, and external filesystem reads.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


class SQLValidationError(Exception):
    """Raised when generated SQL violates AST security constraints."""
    pass


FORBIDDEN_AST_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Pragma,
    exp.Set,
    exp.Copy,
)

FORBIDDEN_FUNCTIONS = {
    # DuckDB file-reading & external resource access
    "read_parquet",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_ndjson_auto",
    "read_text",
    "read_blob",
    "scan_parquet",
    "scan_csv",
    "parquet_scan",
    "csv_scan",
    "glob",
    "httpfs",
    "system",
    "shell",
    "checkpoint",
    "vacuum",
    "attach",
    "detach",
    "duckdb_settings",
}


class SQLASTValidator:
    """Validates SQL queries against strict read-only AST safety policies."""

    @classmethod
    def validate(
        cls,
        sql: str,
        allowed_tables: set[str],
        allowed_columns: set[str] | None = None,
        max_limit: int = 50,
    ) -> str:
        """
        Parse and validate SQL using AST.
        Returns the sanitized, limit-enforced SQL query string.
        """
        if not sql or not sql.strip():
            raise SQLValidationError("Empty SQL statement.")

        # 1. Clean comment blocks and trailing semicolons
        cleaned = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*[\s\S]*?\*/", " ", cleaned).strip().rstrip(";")

        # 2. Parse all statements into AST
        try:
            statements = sqlglot.parse(cleaned, read="duckdb")
        except Exception as exc:
            raise SQLValidationError(f"SQL syntax error: {exc}") from exc

        if not statements:
            raise SQLValidationError("No valid SQL statement found.")
        if len(statements) > 1:
            raise SQLValidationError("Multiple statements are strictly forbidden.")

        stmt = statements[0]
        if stmt is None:
            raise SQLValidationError("Failed to parse SQL AST.")

        # 3. Reject any DDL/DML/Command statement types
        if isinstance(stmt, FORBIDDEN_AST_TYPES):
            raise SQLValidationError(f"Statement type '{type(stmt).__name__}' is forbidden. Only SELECT is allowed.")

        # Must be a Select or Union
        if not isinstance(stmt, (exp.Select, exp.Union)):
            raise SQLValidationError(f"Non-query statement type '{type(stmt).__name__}' is forbidden.")

        # Check for nested mutations inside WITH/CTE
        for forbidden_type in FORBIDDEN_AST_TYPES:
            if stmt.find(forbidden_type):
                raise SQLValidationError(f"Forbidden nested operation '{forbidden_type.__name__}' detected.")

        # 4. Check for forbidden file reading / system functions
        for func in stmt.find_all(exp.Anonymous):
            func_name = func.this.lower()
            if func_name in FORBIDDEN_FUNCTIONS:
                raise SQLValidationError(f"Disallowed external function call: '{func_name}()'.")

        for func in stmt.find_all(exp.Func):
            func_name = func.sql_name().lower()
            if func_name in FORBIDDEN_FUNCTIONS:
                raise SQLValidationError(f"Disallowed function call: '{func_name}'.")

        # 5. Extract CTE aliases
        cte_names: set[str] = set()
        for with_clause in stmt.find_all(exp.With):
            for cte in with_clause.expressions:
                if isinstance(cte, exp.CTE) and cte.alias:
                    cte_names.add(cte.alias.lower())

        # 6. Validate Table Access
        normalized_allowed_tables = {t.lower() for t in allowed_tables}
        found_tables: set[str] = set()

        for table_node in stmt.find_all(exp.Table):
            table_name = table_node.name.lower()
            # If it's a CTE reference, it's allowed
            if table_name in cte_names:
                continue
            if table_name not in normalized_allowed_tables:
                raise SQLValidationError(
                    f"Table '{table_name}' is not permitted."
                )
            found_tables.add(table_name)

        if not found_tables and not cte_names:
            raise SQLValidationError("Query must reference an allowlisted table.")

        # 7. Validate Column Access if configured
        if allowed_columns:
            normalized_allowed_cols = {c.lower() for c in allowed_columns}
            # Allow column aliases defined in query
            query_aliases: set[str] = set()
            for alias_node in stmt.find_all(exp.Alias):
                if alias_node.alias:
                    query_aliases.add(alias_node.alias.lower())

            for col_node in stmt.find_all(exp.Column):
                col_name = col_node.name.lower()
                # Skip star / wildcard
                if col_name in ("*", ""):
                    continue
                if (
                    col_name not in normalized_allowed_cols
                    and col_name not in query_aliases
                    and col_name not in cte_names
                    and col_name not in normalized_allowed_tables
                ):
                    raise SQLValidationError(
                        f"Column '{col_name}' is not permitted in query. Allowed columns: {sorted(normalized_allowed_cols)}"
                    )

        # 8. Enforce Limit Policy
        limit_node = stmt.find(exp.Limit)
        if limit_node is None:
            # Append LIMIT max_limit
            stmt = stmt.limit(max_limit)
        else:
            try:
                curr_limit = int(limit_node.expression.name)
                if curr_limit > max_limit:
                    limit_node.set("expression", exp.Literal.number(max_limit))
            except Exception:
                limit_node.set("expression", exp.Literal.number(max_limit))

        # Re-generate clean DuckDB SQL
        return stmt.sql(dialect="duckdb")
