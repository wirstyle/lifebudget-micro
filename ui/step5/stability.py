"""Compatibility helper for inactive Step 5 stability diagnostics.

The current Step 5 UI renders reliability/stability information through the
main post-run diagnostics flow. This module keeps the historical
``render_stability`` import available for older call sites without rendering
duplicate UI.
"""

from __future__ import annotations

from typing import Any


def render_stability(*args: Any, **kwargs: Any) -> None:
    """No-op renderer kept for compatibility with older Step 5 imports."""
    return None