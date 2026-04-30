"""Policy retrieval tool — Adapter pattern implementation.

Port (abstract interface) + two adapters:
- InMemoryPolicyAdapter: loads text files from disk (current default)
- VectorStorePolicyAdapter: (stub) placeholder for future ChromaDB/FAISS swap

Swapping the backend requires only instantiating a different adapter;
zero changes to GovernanceAgent or AggregatorAgent code.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Port (Interface)
# ---------------------------------------------------------------------------

# parents[3] = implementations/sme_capitalAllocation/
POLICY_DIR = Path(__file__).parents[3] / "data" / "processed" / "policy_docs"


class PolicyRetrievalPort(ABC):
    """Abstract interface for policy document retrieval.

    Implementations must be swappable without touching agent code.
    """

    @abstractmethod
    def get_policy(self, doc_name: str) -> str:
        """Return the policy document content as a string.

        Parameters
        ----------
        doc_name : str
            Logical name of the document, e.g. ``'samp'`` or ``'treasury_policy'``.

        Returns
        -------
        str
            Full text of the policy document, or an empty string if not found.
        """

    @abstractmethod
    def list_available(self) -> list:
        """Return names of all available policy documents."""


# ---------------------------------------------------------------------------
# Adapter 1: InMemory (plain-text files)
# ---------------------------------------------------------------------------


class InMemoryPolicyAdapter(PolicyRetrievalPort):
    """Loads policy documents from plain-text files on disk.

    Caches files in memory after first read to avoid repeated I/O.
    """

    def __init__(self, policy_dir: Optional[Path] = None) -> None:
        self._dir = policy_dir or POLICY_DIR
        self._cache: Dict[str, str] = {}

    def _load(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]
        candidates = [
            self._dir / f"{name}.txt",
            self._dir / f"{name}.md",
        ]
        for path in candidates:
            if path.exists():
                text = path.read_text()
                self._cache[name] = text
                return text
        return ""

    def get_policy(self, doc_name: str) -> str:
        """Return policy text by logical name."""
        return self._load(doc_name.lower().replace(" ", "_"))

    def list_available(self) -> list:
        """Return stem names of all .txt/.md files in the policy directory."""
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.iterdir() if p.suffix in (".txt", ".md")]


# ---------------------------------------------------------------------------
# Adapter 2: VectorStore (stub — swap in without touching agent code)
# ---------------------------------------------------------------------------


class VectorStorePolicyAdapter(PolicyRetrievalPort):
    """Stub adapter for future vector-store-backed semantic retrieval.

    Implement ``get_policy`` to do a nearest-neighbour search over
    chunked policy documents (e.g. ChromaDB, FAISS).
    """

    def get_policy(self, doc_name: str) -> str:
        """Not yet implemented."""
        raise NotImplementedError("VectorStorePolicyAdapter is not implemented yet.")

    def list_available(self) -> list:
        """Not yet implemented."""
        raise NotImplementedError("VectorStorePolicyAdapter is not implemented yet.")


# ---------------------------------------------------------------------------
# CrewAI Tool wrapper
# ---------------------------------------------------------------------------


class PolicyRetrievalInput(BaseModel):
    """Input schema for PolicyRetrievalTool."""

    doc_name: str = Field(
        description=(
            "Name of the policy document to retrieve. "
            "Options: 'samp' (Strategic Asset Management Plan), "
            "'treasury_policy' (Treasury & Capital Allocation Policy)."
        )
    )


class PolicyRetrievalTool(BaseTool):
    """Retrieves NXTFrontier fund-level policy documents for governance evaluation.

    Backed by a :class:`PolicyRetrievalPort` adapter — storage layer is
    configurable without changing agent code.
    """

    name: str = "policy_retrieval_tool"
    description: str = (
        "Retrieve a NXTFrontier policy document by name. "
        "Available: 'samp', 'treasury_policy'."
    )
    args_schema: Type[BaseModel] = PolicyRetrievalInput
    _adapter: PolicyRetrievalPort

    def __init__(self, adapter: Optional[PolicyRetrievalPort] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_adapter", adapter or InMemoryPolicyAdapter())

    def _run(self, doc_name: str) -> str:
        """Return the policy document text or an error JSON.

        Parameters
        ----------
        doc_name : str
            Logical policy document name.

        Returns
        -------
        str
            Policy document text, or JSON error string.
        """
        text = self._adapter.get_policy(doc_name)
        if not text:
            available = self._adapter.list_available()
            return json.dumps({
                "error": f"Document '{doc_name}' not found.",
                "available": available,
            })
        return text
