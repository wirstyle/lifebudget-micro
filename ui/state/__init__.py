# ui/state/__init__.py

"""
State management utilities for LifeBudget Micro.

This package centralises everything related to Streamlit session_state:
- bootstrap: initialise default values
- keys: define canonical session keys
- updates: safe update patterns to avoid Streamlit mutation errors

Goal:
- Avoid scattered session_state logic across app.py
- Provide a single source of truth for state handling
"""

__all__ = [
    "bootstrap",
    "keys",
    "updates",
]
