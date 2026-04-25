# services/__init__.py

"""
Service layer for LifeBudget Micro.

This package contains lightweight orchestration / adapter logic that sits
between:
- src/      -> core business / quantitative logic
- ui/       -> rendering and interaction
- app.py    -> top-level composition

Typical responsibilities:
- adapt UI payloads into engine-ready configs
- prepare reporting payloads for display
- build recommendation payloads
- keep glue logic out of app.py and out of the UI modules
"""

__all__ = [
    "ui_adapters",
    "recommendation_service",
    "reporting_service",
    "projection_service",
]
