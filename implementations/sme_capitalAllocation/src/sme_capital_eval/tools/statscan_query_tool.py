"""StatsCan benchmark query tool — Adapter pattern implementation.

Port (abstract interface) + two adapters:
- SQLiteBenchmarkAdapter: queries ``data/processed/benchmarks.db`` (default)
- ParquetBenchmarkAdapter: (stub) placeholder for Pandas/Parquet swap

The Benchmark Agent calls StatCanQueryTool — the tool delegates to whichever
adapter is injected, keeping LLM-layer code stable across storage changes.
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# parents[3] = implementations/sme_capitalAllocation/
DB_PATH = Path(__file__).parents[3] / "data" / "processed" / "benchmarks.db"


# ---------------------------------------------------------------------------
# Port
# ---------------------------------------------------------------------------


class BenchmarkQueryPort(ABC):
    """Abstract interface for deterministic benchmark data retrieval."""

    @abstractmethod
    def query_capex_norm(self, naics: str, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return CapEx benchmarks for a given NAICS code.

        Parameters
        ----------
        naics : str
            2–6 digit NAICS code.
        asset_type : str, optional
            Asset type filter (e.g. ``'Machinery and equipment'``).

        Returns
        -------
        list of dict
            Rows with at least: ``naics``, ``asset_type``, ``median_capex_cad``, ``ref_year``.
        """

    @abstractmethod
    def query_financial_ratio(self, naics: str, measure: str) -> List[Dict[str, Any]]:
        """Return financial ratio benchmarks for a given NAICS code.

        Parameters
        ----------
        naics : str
            2–6 digit NAICS code.
        measure : str
            Ratio name (e.g. ``'debt_to_equity'``, ``'current_ratio'``).

        Returns
        -------
        list of dict
            Rows with at least: ``naics``, ``measure``, ``median_value``, ``ref_year``.
        """

    @abstractmethod
    def query_sfgsme_lending(self, industry: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return SFGSME lending benchmarks (approval rates, typical loan sizes).

        Parameters
        ----------
        industry : str, optional
            Industry filter string.

        Returns
        -------
        list of dict
            Rows with: ``industry``, ``measure``, ``value``, ``ref_year``.
        """


# ---------------------------------------------------------------------------
# Adapter 1: SQLite
# ---------------------------------------------------------------------------


class SQLiteBenchmarkAdapter(BenchmarkQueryPort):
    """Queries the processed StatsCan SQLite database.

    Falls back gracefully if the DB has not been built yet (returns empty list).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db = db_path or DB_PATH

    def _execute(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        if not self._db.exists():
            return []
        try:
            with sqlite3.connect(self._db) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    def query_capex_norm(self, naics: str, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query capex_benchmarks table by NAICS prefix match."""
        if asset_type:
            return self._execute(
                "SELECT * FROM capex_benchmarks WHERE naics LIKE ? AND asset_type LIKE ? "
                "ORDER BY ref_year DESC LIMIT 10",
                (f"{naics}%", f"%{asset_type}%"),
            )
        return self._execute(
            "SELECT * FROM capex_benchmarks WHERE naics LIKE ? ORDER BY ref_year DESC LIMIT 10",
            (f"{naics}%",),
        )

    def query_financial_ratio(self, naics: str, measure: str) -> List[Dict[str, Any]]:
        """Query financial_ratios table."""
        return self._execute(
            "SELECT * FROM financial_ratios WHERE naics LIKE ? AND measure LIKE ? "
            "ORDER BY ref_year DESC LIMIT 5",
            (f"{naics}%", f"%{measure}%"),
        )

    def query_sfgsme_lending(self, industry: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query sfgsme_lending table."""
        if industry:
            return self._execute(
                "SELECT * FROM sfgsme_lending WHERE industry LIKE ? ORDER BY ref_year DESC LIMIT 10",
                (f"%{industry}%",),
            )
        return self._execute("SELECT * FROM sfgsme_lending ORDER BY ref_year DESC LIMIT 10")


# ---------------------------------------------------------------------------
# Adapter 2: Parquet (stub)
# ---------------------------------------------------------------------------


class ParquetBenchmarkAdapter(BenchmarkQueryPort):
    """Stub adapter for Pandas/Parquet-backed benchmark retrieval."""

    def query_capex_norm(self, naics: str, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Not yet implemented."""
        raise NotImplementedError("ParquetBenchmarkAdapter is not implemented yet.")

    def query_financial_ratio(self, naics: str, measure: str) -> List[Dict[str, Any]]:
        """Not yet implemented."""
        raise NotImplementedError

    def query_sfgsme_lending(self, industry: Optional[str] = None) -> List[Dict[str, Any]]:
        """Not yet implemented."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CrewAI Tool
# ---------------------------------------------------------------------------


class StatCanQueryInput(BaseModel):
    """Input schema for StatCanQueryTool."""

    query_type: str = Field(
        description=(
            "Type of benchmark to look up. One of: "
            "'capex_norm', 'financial_ratio', 'sfgsme_lending'."
        )
    )
    naics: str = Field(default="", description="NAICS code to filter by (e.g. '311', '484').")
    measure: str = Field(
        default="",
        description="For financial_ratio queries: ratio name (e.g. 'debt_to_equity').",
    )
    asset_type: str = Field(default="", description="Asset type filter for capex queries.")
    industry: str = Field(default="", description="Industry string for SFGSME queries.")


class StatCanQueryTool(BaseTool):
    """Deterministic lookup into processed StatsCan and SFGSME datasets.

    The Benchmark Agent calls this tool to retrieve real numbers —
    it never has the LLM guess or hallucinate industry benchmarks.
    """

    name: str = "statscan_query_tool"
    description: str = (
        "Query StatsCan benchmark databases. "
        "query_type options: 'capex_norm', 'financial_ratio', 'sfgsme_lending'. "
        "Provide 'naics' for industry filtering."
    )
    args_schema: Type[BaseModel] = StatCanQueryInput
    _adapter: BenchmarkQueryPort

    def __init__(self, adapter: Optional[BenchmarkQueryPort] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_adapter", adapter or SQLiteBenchmarkAdapter())

    def _run(  # noqa: PLR0911
        self,
        query_type: str,
        naics: str = "",
        measure: str = "",
        asset_type: str = "",
        industry: str = "",
    ) -> str:
        """Execute the benchmark lookup and return results as a JSON string."""
        try:
            if query_type == "capex_norm":
                rows = self._adapter.query_capex_norm(naics, asset_type or None)
            elif query_type == "financial_ratio":
                rows = self._adapter.query_financial_ratio(naics, measure)
            elif query_type == "sfgsme_lending":
                rows = self._adapter.query_sfgsme_lending(industry or None)
            else:
                return json.dumps({"error": f"Unknown query_type: {query_type!r}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        if not rows:
            return json.dumps({
                "result": [],
                "note": "No data found. Database may not be built yet — run data engineering pipeline.",
            })
        return json.dumps({"query_type": query_type, "naics": naics, "result": rows})
