# ui/step5/__init__.py

"""
Step 5 submodules (Engine Workspace) for LifeBudget Micro.

This package contains modular UI components for the investment workspace:
- simple_mode: presets and basic sliders
- advanced_params: advanced configuration controls
- run_panel: engine execution controls
- post_run: results, tables, diagnostics
- preset_recommendations: suggested preset improvements + apply
- universe_recommendations: candidate baskets, acceptance gate, apply
- governance: coherence / philosophy panel
- stability: temporal robustness / candidate stability views

These modules are intentionally conservative, but together they now cover much
more of the legacy Step 5 surface from the old monolithic app.py.
"""

__all__ = [
    "simple_mode",
    "advanced_params",
    "run_panel",
    "post_run",
    "preset_recommendations",
    "universe_recommendations",
    "governance",
    "stability",
]
