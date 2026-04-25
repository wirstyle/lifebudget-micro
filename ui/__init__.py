# ui/__init__.py

"""
UI layer for LifeBudget Micro

This package contains:
- steps: main user flow (step1 → step7)
- step5: investment workspace submodules
- common: reusable UI components (metrics, tables, cards, messages)
- state: session_state management helpers

Design principles:
- Keep rendering logic separate from business logic (src/)
- Keep app.py as a thin entrypoint/router
- Ensure each step is independently testable
"""

__all__ = [
    "steps",
    "step5",
    "common",
    "state",
]
