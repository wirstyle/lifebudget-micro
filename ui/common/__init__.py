# ui/common/__init__.py

"""
Common reusable UI components for LifeBudget Micro.

This module groups shared visual elements used across steps:
- metrics: KPI displays (st.metric wrappers, formatted outputs)
- tables: dataframe rendering helpers
- messages: info / warning / success UI blocks
- cards: structured UI containers for recommendations, summaries, etc.

Goal:
Avoid duplicating UI patterns across steps and keep consistency.
"""

__all__ = [
    "metrics",
    "tables",
    "messages",
    "cards",
]
