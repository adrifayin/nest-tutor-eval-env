"""Server package — exposes ASGI ``app`` for ``uvicorn server:app``."""

from .app import app

__all__ = ["app"]

