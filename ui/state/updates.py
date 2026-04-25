from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping

import streamlit as st

try:
    from .keys import PENDING_UPDATES as PENDING_UPDATES_KEY
except Exception:
    PENDING_UPDATES_KEY = "_pending_session_updates"


def _clone(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def _coerce_updates(updates: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(updates, Mapping):
        return {}
    return {str(key): value for key, value in dict(updates).items()}


def queue_session_updates(updates: Mapping[str, Any], *, merge_nested: bool = False) -> None:
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
    st.session_state[PENDING_UPDATES_KEY] = {}


def queue_and_rerun(updates: Mapping[str, Any], *, merge_nested: bool = False) -> None:
    queue_session_updates(updates, merge_nested=merge_nested)
    st.rerun()


def apply_pending_step_patch(patch_key: str, success_key: str | None = None) -> tuple[dict[str, Any], str | None]:
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
    return st.session_state.get(key, default)


def clear_state_keys(*keys: str) -> None:
    for key in keys:
        if key in st.session_state:
            st.session_state.pop(key, None)


def reset_state_keys(keys: Iterable[str], default: Any = None) -> None:
    for key in list(keys or []):
        st.session_state[str(key)] = _clone(default)
