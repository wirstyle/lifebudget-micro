# ui/common/tables.py

"""
Reusable table rendering helpers for Streamlit.

Goal:
- Standardise dataframe display
- Add sorting / formatting helpers
- Reduce duplication in app.py
- Centralise common comparison / download patterns
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import pandas as pd
import streamlit as st


def _is_valid_df(df: pd.DataFrame | None) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def _safe_title(title: Optional[str]) -> None:
    if title:
        st.markdown(f"### {title}")


def _normalize_editor_df(df) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df.copy().reset_index(drop=True)


def _df_fingerprint(df: pd.DataFrame | None) -> str:
    if not isinstance(df, pd.DataFrame):
        return "__none__"
    try:
        work = _normalize_editor_df(df)
        return work.to_json(orient="split", date_format="iso", default_handler=str)
    except Exception:
        try:
            return repr(df.to_dict("records"))
        except Exception:
            return "__unserializable__"


def init_dataframe_state(
    state_key: str,
    default_df: pd.DataFrame | Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    current = st.session_state.get(state_key)
    if isinstance(current, pd.DataFrame):
        current = _normalize_editor_df(current)
        st.session_state[state_key] = current
        return current

    df = default_df() if callable(default_df) else default_df
    df = _normalize_editor_df(df)
    st.session_state[state_key] = df
    return df


def stateful_data_editor(
    *,
    state_key: str,
    editor_key: str,
    default_df: pd.DataFrame | Callable[[], pd.DataFrame],
    **editor_kwargs,
) -> pd.DataFrame:
    """
    Robust pattern for st.data_editor.

    Key behaviour:
    - state_key stores the canonical dataframe used by app logic
    - a separate editor_source_key stores the dataframe fed to the widget
    - when canonical data changes outside the widget, we bump a versioned
      widget key so Streamlit remounts the editor instead of reusing stale
      internal widget state
    """
    editor_source_key = f"{state_key}__editor_source"
    last_value_key = f"{state_key}__editor_last_value"
    source_fp_key = f"{state_key}__editor_source_fingerprint"
    version_key = f"{state_key}__editor_version"

    canonical_df = init_dataframe_state(state_key, default_df)
    canonical_df = _normalize_editor_df(canonical_df)
    canonical_fp = _df_fingerprint(canonical_df)

    current_source = st.session_state.get(editor_source_key)
    current_source = _normalize_editor_df(current_source) if isinstance(current_source, pd.DataFrame) else pd.DataFrame()
    stored_fp = str(st.session_state.get(source_fp_key, "__missing__"))

    needs_resync = (
        not isinstance(st.session_state.get(editor_source_key), pd.DataFrame)
        or stored_fp != canonical_fp
        or not current_source.equals(canonical_df)
    )

    if needs_resync:
        st.session_state[editor_source_key] = canonical_df.copy()
        st.session_state[source_fp_key] = canonical_fp
        st.session_state[version_key] = int(st.session_state.get(version_key, 0) or 0) + 1

    effective_editor_key = f"{editor_key}__v{int(st.session_state.get(version_key, 0) or 0)}"
    editor_df = _normalize_editor_df(st.session_state[editor_source_key])

    edited_df = st.data_editor(
        editor_df,
        key=effective_editor_key,
        **editor_kwargs,
    )
    edited_df = _normalize_editor_df(edited_df)

    previous_df = st.session_state.get(last_value_key)
    changed_vs_previous = not isinstance(previous_df, pd.DataFrame) or not edited_df.equals(previous_df)

    if changed_vs_previous:
        st.session_state[last_value_key] = edited_df.copy()
        st.session_state[state_key] = edited_df.copy()
        st.session_state[editor_source_key] = edited_df.copy()
        st.session_state[source_fp_key] = _df_fingerprint(edited_df)

    return _normalize_editor_df(st.session_state.get(state_key))


def show_table(
    df: pd.DataFrame,
    title: Optional[str] = None,
    use_container_width: bool = True,
    hide_index: bool = True,
) -> None:
    """Basic standardized table display."""
    if not _is_valid_df(df):
        st.info("No data available.")
        return

    _safe_title(title)
    st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)


def show_table_if_not_empty(
    df: pd.DataFrame,
    title: Optional[str] = None,
    empty_message: str = "No data available.",
    use_container_width: bool = True,
    hide_index: bool = True,
) -> bool:
    """Render a table only when it contains rows. Returns True if rendered."""
    if not _is_valid_df(df):
        st.info(empty_message)
        return False

    _safe_title(title)
    st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)
    return True


def sortable_table(
    df: pd.DataFrame,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    title: Optional[str] = None,
) -> None:
    """Display table with optional sorting."""
    if not _is_valid_df(df):
        st.info("No data available.")
        return

    work = df.copy()
    if sort_by and sort_by in work.columns:
        try:
            work = work.sort_values(by=sort_by, ascending=ascending)
        except Exception:
            pass

    show_table(work, title=title)


def format_percentage_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not _is_valid_df(df):
        return df

    work = df.copy()
    for col in columns:
        if col in work.columns:
            try:
                numeric = pd.to_numeric(work[col], errors="coerce")
                work[col] = numeric.map(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "—")
            except Exception:
                pass
    return work


def format_decimal_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    decimals: int = 3,
) -> pd.DataFrame:
    if not _is_valid_df(df):
        return df

    work = df.copy()
    for col in columns:
        if col in work.columns:
            try:
                numeric = pd.to_numeric(work[col], errors="coerce")
                work[col] = numeric.map(lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else "—")
            except Exception:
                pass
    return work


def format_currency_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    currency_symbol: str = "£",
    decimals: int = 2,
) -> pd.DataFrame:
    if not _is_valid_df(df):
        return df

    work = df.copy()
    for col in columns:
        if col in work.columns:
            try:
                numeric = pd.to_numeric(work[col], errors="coerce")
                work[col] = numeric.map(
                    lambda x: f"{currency_symbol}{x:,.{decimals}f}" if pd.notnull(x) else "—"
                )
            except Exception:
                pass
    return work


def highlight_max(df: pd.DataFrame, columns: Sequence[str]):
    if not _is_valid_df(df):
        return df

    def _highlight(series: pd.Series):
        try:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.dropna().empty:
                return ["" for _ in series]
            max_value = numeric.max()
            is_max = numeric == max_value
            return ["background-color: #d4edda" if bool(v) else "" for v in is_max]
        except Exception:
            return ["" for _ in series]

    try:
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            return df
        return df.style.apply(_highlight, subset=valid_columns)
    except Exception:
        return df


def comparison_table(
    df: pd.DataFrame,
    title: Optional[str] = None,
    delta_columns: Sequence[str] | None = None,
    use_container_width: bool = True,
    hide_index: bool = True,
    identical_message: str = "Current and comparison configurations are materially identical.",
) -> None:
    if not _is_valid_df(df):
        st.info("No comparison data available.")
        return

    work = df.copy()
    delta_columns = [str(col) for col in (delta_columns or []) if str(col) in work.columns]

    if delta_columns:
        numeric_present = False
        total_delta = 0.0

        for col in delta_columns:
            try:
                numeric = pd.to_numeric(work[col], errors="coerce")
                valid = numeric.dropna()
                if not valid.empty:
                    numeric_present = True
                    total_delta += float(valid.abs().sum())
            except Exception:
                pass

        if numeric_present and total_delta < 1e-10:
            st.success(identical_message)
            return

    show_table(
        work,
        title=title or "Comparison",
        use_container_width=use_container_width,
        hide_index=hide_index,
    )


def leaderboard_table(
    df: pd.DataFrame,
    title: str = "Candidate leaderboard",
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> None:
    if not _is_valid_df(df):
        st.info("No candidates to display.")
        return

    work = df.copy()
    if sort_by and sort_by in work.columns:
        try:
            work = work.sort_values(sort_by, ascending=ascending)
        except Exception:
            pass

    _safe_title(title)
    try:
        st.dataframe(work, use_container_width=True, hide_index=True)
    except Exception:
        st.write(work)


def focus_metrics_table(metrics: dict, title: str = "Focus metrics") -> None:
    if not isinstance(metrics, dict) or not metrics:
        st.info("No focus metrics available.")
        return

    rows = [{"metric": str(k), "value": v} for k, v in metrics.items()]
    df = pd.DataFrame(rows)
    show_table(df, title=title, hide_index=True)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if not isinstance(df, pd.DataFrame):
        return b""
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_json_bytes(df: pd.DataFrame) -> bytes:
    if not isinstance(df, pd.DataFrame):
        return b""
    return df.to_json(orient="records", indent=2).encode("utf-8")


def download_dataframe_button(
    label: str,
    df: pd.DataFrame,
    filename: str,
    *,
    key: Optional[str] = None,
    filetype: str = "csv",
) -> None:
    if not _is_valid_df(df):
        return

    filetype = str(filetype or "csv").lower().strip()
    if filetype == "json":
        data = dataframe_to_json_bytes(df)
        mime = "application/json"
    else:
        data = dataframe_to_csv_bytes(df)
        mime = "text/csv"

    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        key=key,
    )
