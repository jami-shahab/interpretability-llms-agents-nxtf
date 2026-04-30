"""Langfuse client singleton — gracefully disabled if keys are not configured."""

import os
from typing import Optional


def get_client() -> Optional[object]:
    """Return a Langfuse client if credentials are available, else None.

    Returns
    -------
    Langfuse | None
        Configured client, or ``None`` if ``LANGFUSE_PUBLIC_KEY`` /
        ``LANGFUSE_SECRET_KEY`` are not set.
    """
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pub or not sec:
        return None
    try:
        from langfuse import Langfuse  # type: ignore

        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        return Langfuse(public_key=pub, secret_key=sec, host=host)
    except Exception:
        return None
