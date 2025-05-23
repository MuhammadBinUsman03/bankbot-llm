"""
API subpackage for AI Core.
"""

from .app import app
from .routes import router

__all__ = ["app", "router"]