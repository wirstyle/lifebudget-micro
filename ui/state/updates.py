from __future__ import annotations

"""Deferred Streamlit session-state update helpers.

Streamlit reruns the script from top to bottom after most user interactions.
This is useful, but it can be risky to mutate widget-owned keys directly after
their widgets have already been created.

This module provides a small deferred-update layer:

- queue updates during a button callback or action;
- rerun the app;
- apply the queued values before widgets are recreated;
- optionally carry success messages across reruns.

This is used by presets, recommendation buttons and step-level widget patches.
It keeps user-facing actions such as "Apply recommended preset" or "Keep current
setup" stable without scattering direct session-state mutation across the UI.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping

import streamlit as st

try:
    from .keys import PENDING_UPDATES as PENDING_UPDATES_KEY
except Exception:
    # Fallback keeps this helper importable in small tests or isolated contexts
    # where ui.state.keys may not be available.
    PENDING_UPDATES_KEY = "_pending_session_updates"


def _clone(value: Any) -> Any:
    """Return a defensive copy before storing a value in session state.

    Many state payloads are dictionaries, lists, DataFrames or nested objects.
    Copying avoids accidental mutation between queued payloads and later app
    state.
    """
    try:
        return deepcopy(value)
    except Exception:
        return value


def _coerce_updates(updates: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Normalise an update mapping into a plain dictionary with string keys."""
    if not isinstance(updates, Mapping):
        return {}
    return {str(key): value for key, value in dict(updates).items()}


def queue_session_updates(updates: Mapping[str, Any], *, merge_nested: bool = False) -> None:
    """Queue global session-state updates for the next app rerun.

    The values are not applied immediately. They are stored under
    ``PENDING_UPDATES_KEY`` and later applied by ``apply_queued_session_updates``
    near the start of ``app.py``.

    Parameters
    ----------
    updates:
        Mapping of session-state keys to values that should be applied on the
        next run.
    merge_nested:
        If True, and both the existing pending value and the new value are
        dictionaries, merge them instead of replacing the whole nested payload.
    """
    new_updates = _coerce_updates(updates)
    if not new_updates:
        return

    pending = st.session_state.get(PENDING_UPDATES_KEY, {})
    if not isinstance(pending, dict):
        pending = {}

    for key, value in new_updates.items():
        if merge_nested and isinstance(pending.get(key), dict) and isinstance(value, Mapping):
            merged = dict(pending.get(key, {}))
            merged.update(dict(value))
            pending[key] = merged
        else:
            pending[key] = _clone(value)

    st.session_state[PENDING_UPDATES_KEY] = pending


def apply_queued_session_updates(*, clear_queue: bool = True) -> Dict[str, Any]:
    """Apply queued global session-state updates.

    This should run before screen widgets are created. In the main app, it is
    called immediately after bootstrapping session defaults.

    Returns
    -------
    dict
        The updates that were applied. This is useful for debugging or tests.
    """
    pending = st.session_state.get(PENDING_UPDATES_KEY, {})
    if not isinstance(pending, dict) or not pending:
        return {}

    applied = dict(pending)
    for key, value in applied.items():
        st.session_state[key] = _clone(value)

    if clear_queue:
        st.session_state[PENDING_UPDATES_KEY] = {}

    return applied


def apply_all_pending_updates(*, clear_queue: bool = True) -> Dict[str, Any]:
    """Convenience alias for applying the global queued updates at app start."""
    return apply_queued_session_updates(clear_queue=clear_queue)


def clear_queued_session_updates() -> None:
    """Clear the global update queue without applying it."""
    st.session_state[PENDING_UPDATES_KEY] = {}


def queue_and_rerun(updates: Mapping[str, Any], *, merge_nested: bool = False) -> None:
    """Queue updates and immediately trigger a Streamlit rerun.

    Use this when a user action should update state and refresh the UI in one
    step, for example applying a recommendation or switching a configured
    preset.
    """
    queue_session_updates(updates, merge_nested=merge_nested)
    st.rerun()


def apply_pending_step_patch(patch_key: str, success_key: str | None = None) -> tuple[dict[str, Any], str | None]:
    """Apply a per-step deferred widget patch.

    Some screens keep a step-local patch key rather than using the global queue.
    This is useful when a widget preset or action belongs only to that screen.
    The patch is popped after being applied so it is not repeated on later runs.

    Parameters
    ----------
    patch_key:
        Session-state key containing a dictionary of updates to apply.
    success_key:
        Optional session-state key containing a success message to show after
        the patch is applied.

    Returns
    -------
    tuple
        ``(applied_updates, success_message)``.
    """
    pending = st.session_state.pop(patch_key, None)
    applied = {}

    if isinstance(pending, dict):
        for key, value in pending.items():
            st.session_state[key] = _clone(value)
        applied = dict(pending)

    success_message = st.session_state.pop(success_key, None) if success_key else None
    return applied, success_message if isinstance(success_message, str) and success_message else None


def queue_updates_with_feedback(
    updates: Mapping[str, Any],
    *,
    patch_key: str,
    success_key: str | None = None,
    success_message: str | None = None,
    merge_nested: bool = False,
) -> None:
    """Queue a step-local patch and optional success message, then rerun.

    This pattern is useful for preset buttons. The button can request widget
    changes and a user-facing confirmation without mutating widget-owned keys
    directly in the same render pass.
    """
    payload = {patch_key: dict(_coerce_updates(updates))}
    if success_key and success_message:
        payload[success_key] = str(success_message)
    queue_and_rerun(payload, merge_nested=merge_nested)


def queue_step_patch(
    patch_key: str,
    updates: Mapping[str, Any],
    *,
    success_key: str | None = None,
    success_message: str | None = None,
    merge_nested: bool = False,
) -> None:
    """Semantic alias for per-step deferred widget patches."""
    queue_updates_with_feedback(
        updates,
        patch_key=patch_key,
        success_key=success_key,
        success_message=success_message,
        merge_nested=merge_nested,
    )


def safe_get_state(key: str, default: Any = None) -> Any:
    """Read a session-state value with a fallback default."""
    return st.session_state.get(key, default)


def clear_state_keys(*keys: str) -> None:
    """Remove selected keys from session state if present."""
    for key in keys:
        if key in st.session_state:
            st.session_state.pop(key, None)


def reset_state_keys(keys: Iterable[str], default: Any = None) -> None:
    """Reset selected keys to the same cloned default value."""
    for key in list(keys or []):
        st.session_state[str(key)] = _clone(default)