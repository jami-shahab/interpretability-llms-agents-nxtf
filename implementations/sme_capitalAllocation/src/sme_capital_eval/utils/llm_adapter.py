"""LLM provider adapter — port + concrete adapters.

Follows the Adapter (Port/Adapter) design pattern.  
All agents depend only on ``LLMAdapterPort``; swapping the LLM backend
requires zero changes to agent code.

Supported adapters:
  - GeminiLLMAdapter : Google Gemini via CrewAI (requires GEMINI_API_KEY)
  - OllamaLLMAdapter : Local Ollama server (no API key required)
  - OpenAILLMAdapter : OpenAI API (requires OPENAI_API_KEY)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from crewai import LLM


# ---------------------------------------------------------------------------
# Port (Interface)
# ---------------------------------------------------------------------------


class LLMAdapterPort(ABC):
    """Abstract port that all LLM adapters must implement."""

    @abstractmethod
    def build(self) -> LLM:
        """Return a configured CrewAI LLM instance.

        Returns
        -------
        LLM
            A CrewAI LLM ready to be passed to ``Agent(llm=...)``.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the model (e.g. 'gemini-2.0-flash-lite')."""


# ---------------------------------------------------------------------------
# Gemini Adapter
# ---------------------------------------------------------------------------


class GeminiLLMAdapter(LLMAdapterPort):
    """Adapter for Google Gemini models via the CrewAI/LiteLLM integration.

    Parameters
    ----------
    model : str
        Short model name, e.g. ``"gemini-2.0-flash-lite"``.
    api_key : str, optional
        Gemini API key. Defaults to ``GEMINI_API_KEY`` environment variable.
    temperature : float
        Sampling temperature. Default ``0`` (deterministic).
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite",
        api_key: Optional[str] = None,
        temperature: float = 0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def build(self) -> LLM:
        """Build a Gemini LLM instance.

        CrewAI uses LiteLLM internally; prefix ``gemini/`` routes to Google AI.
        """
        return LLM(
            model=f"gemini/{self._model}",
            api_key=self._api_key,
            temperature=self._temperature,
        )


# ---------------------------------------------------------------------------
# Ollama Adapter
# ---------------------------------------------------------------------------


class OllamaLLMAdapter(LLMAdapterPort):
    """Adapter for local Ollama models.

    No API key required.  Ollama must be running locally (or on the
    specified ``base_url``).

    Parameters
    ----------
    model : str
        Ollama model tag, e.g. ``"gemma4:e2b"``, ``"llama3.1:8b"``.
    base_url : str
        Ollama server URL. Default ``http://localhost:11434``.
    temperature : float
        Sampling temperature. Default ``0`` (deterministic).
    """

    def __init__(
        self,
        model: str = "gemma4:e2b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def build(self) -> LLM:
        """Build an Ollama LLM instance.

        CrewAI uses LiteLLM; prefix ``ollama/`` routes to the local Ollama server.
        """
        return LLM(
            model=f"ollama/{self._model}",
            base_url=self._base_url,
            temperature=self._temperature,
        )


# ---------------------------------------------------------------------------
# OpenAI Adapter
# ---------------------------------------------------------------------------


class OpenAILLMAdapter(LLMAdapterPort):
    """Adapter for OpenAI models.

    Requires ``OPENAI_API_KEY`` in the environment (or passed explicitly).
    Recommended small model: ``gpt-4o-mini`` (cheap, strong instruction-following).

    Parameters
    ----------
    model : str
        OpenAI model name, e.g. ``"gpt-4o-mini"``, ``"gpt-3.5-turbo"``.
    api_key : str, optional
        OpenAI API key. Defaults to ``OPENAI_API_KEY`` environment variable.
    temperature : float
        Sampling temperature. Default ``0`` (deterministic).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def build(self) -> LLM:
        """Build an OpenAI LLM instance.

        LiteLLM treats OpenAI as the default provider — no prefix required.
        """
        return LLM(
            model=self._model,
            api_key=self._api_key,
            temperature=self._temperature,
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_adapter(
    provider: str,
    model: str,
    ollama_url: str = "http://localhost:11434",
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> LLMAdapterPort:
    """Factory: create the correct adapter from CLI/config arguments.

    Parameters
    ----------
    provider : str
        ``"gemini"``, ``"ollama"``, or ``"openai"``.
    model : str
        Model identifier appropriate for the provider.
    ollama_url : str
        Ollama server URL (only used when ``provider == "ollama"``).
    gemini_api_key : str, optional
        API key override (only used when ``provider == "gemini"``).
    openai_api_key : str, optional
        API key override (only used when ``provider == "openai"``).

    Returns
    -------
    LLMAdapterPort
    """
    if provider == "gemini":
        return GeminiLLMAdapter(model=model, api_key=gemini_api_key)
    if provider == "ollama":
        return OllamaLLMAdapter(model=model, base_url=ollama_url)
    if provider == "openai":
        return OpenAILLMAdapter(model=model, api_key=openai_api_key)
    raise ValueError(
        f"Unknown provider '{provider}'. Choose 'gemini', 'ollama', or 'openai'."
    )

