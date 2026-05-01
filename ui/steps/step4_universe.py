from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from ui.common.messages import section_header
from ui.common.metrics import currency_metric
from ui.common.tables import show_table_if_not_empty
from ui.services.step4_universe_service import (
    INVESTMENT_PHILOSOPHY_OPTIONS,
    UNIVERSE_SIZES,
    allowed_style_presets_for_philosophy,
    allowed_strategy_templates_for_philosophy,
    allowed_universe_strategies_for_size,
    apply_investment_philosophy_bundle,
    build_bridge_explanation,
    build_generated_universe,
    build_step4_universe_payload_from_state,
    build_strategy_candidate_pool,
    build_universe_mix,
    build_universe_mix_detail,
    build_universe_preview_text,
    build_universe_size_status,
    asset_display_label,
    normalize_asset_ticker,
    coerce_snapshot,
    get_canonical_investment_philosophy,
    maybe_apply_initial_universe_size_default,
    panel_df_to_csv_bytes,
    panel_download_filename,
    parse_custom_assets,
    preferred_universe_strategies_for_philosophy,
    recommended_strategy_combo_for_philosophy,
    recommended_universe_sizes_for_philosophy,
    resolve_recommended_strategy_for_size,
    resolve_step4_asset_panel,
    resolve_universe_selection,
    store_investment_context,
    strategy_combo_status,
    sync_step4_state,
    update_asset_panel_state,
    build_step4_panel_input_signature,
    format_step4_timing_summary,
    STEP4_PANEL_SIGNATURE_KEY,
    STEP4_PANEL_TIMINGS_KEY,
    STEP4_RAW_DAILY_PANEL_KEY,
    STEP4_RAW_WEEKLY_PANEL_KEY,
    STEP4_RAW_PANEL_SOURCE_KEY,
)
from ui.state.keys import (
    ASSET_AUTO_ADJUST,
    ASSET_END_DATE,
    ASSET_PANEL_DF,
    ASSET_PANEL_READY,
    ASSET_PANEL_SOURCE_LABEL,
    ASSET_RETURN_FREQUENCY,
    ASSET_SOURCE_MODE,
    ASSET_START_DATE,
    ASSET_UPLOADED_FILE,
    CURRENT_STEP,
    CUSTOM_UNIVERSE_TEXT,
    INVESTMENT_START_DATE_STABILITY_DATES,
    INVESTMENT_START_DATE_STABILITY_ENABLED,
    UNIVERSE_CUSTOM_ENABLED,
)


# Cached deployment panels contain 106 assets. Keep the public/demo size menu
# honest by exposing only universe sizes that can be materially supported by
# the cached data bundle. Larger sizes need a larger external/live data source.
DEPLOYMENT_MAX_VISIBLE_UNIVERSE_SIZE = 100

# If the user enters the Investment Strategy Lab directly from Home, there may
# be no saved Step 1-3 savings target yet. Use a small educational default so
# the demo bridge is meaningful, but always let an inherited target from the
# Personal Finance Setup override it.
STEP4_DEFAULT_WEEKLY_CONTRIBUTION = 48.0
STEP4_DEFAULT_MONTHLY_CONTRIBUTION = STEP4_DEFAULT_WEEKLY_CONTRIBUTION * 52.0 / 12.0

# UI-only toggle controlled from the Step 4 sidebar. It reveals audit/download
# tools without keeping technical diagnostics in the normal setup flow.
STEP4_SHOW_MARKET_DATA_TOOLS = "step4_show_market_data_tools"
STEP4_PANEL_SIDEBAR_REFRESH_SIGNATURE = "step4_panel_sidebar_refresh_signature_v1"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _first_positive_value(*values) -> float:
    for value in values:
        candidate = _safe_float(value, 0.0)
        if candidate > 0.0:
            return float(candidate)
    return 0.0


def _resolve_contribution_bridge_context(investment_context: dict, plan_snapshot: dict) -> dict:
    """Return a Step 4 contribution context with a safe demo fallback.

    Priority:
    1. Keep a positive weekly/monthly contribution inherited from Step 1-3.
    2. If no prior contribution exists, use the educational demo default
       (£48/week ≈ £208/month).

    This avoids the hosted demo showing a confusing £0 bridge when a user jumps
    directly into the Investment Strategy Lab from Home.
    """
    ctx = dict(investment_context or {})
    snapshot = dict(plan_snapshot or {}) if isinstance(plan_snapshot, dict) else {}

    existing_weekly = _first_positive_value(
        ctx.get("weekly_equivalent"),
        ctx.get("weekly_contribution"),
        ctx.get("target_weekly"),
        snapshot.get("target_a_weekly"),
        snapshot.get("target_weekly"),
        snapshot.get("weekly_savings_target"),
        st.session_state.get("step1_target_weekly_savings"),
        st.session_state.get("target_a_weekly"),
        st.session_state.get("weekly_savings_target"),
    )

    existing_monthly = _first_positive_value(
        ctx.get("monthly_contribution"),
        ctx.get("target_a_monthly"),
        snapshot.get("target_a_monthly"),
        snapshot.get("target_monthly"),
        snapshot.get("monthly_savings_target"),
    )

    if existing_weekly > 0.0:
        weekly = float(existing_weekly)
        monthly = float(existing_monthly) if existing_monthly > 0.0 else weekly * 52.0 / 12.0
        source = "Inherited from Personal Finance Setup"
    elif existing_monthly > 0.0:
        monthly = float(existing_monthly)
        weekly = monthly * 12.0 / 52.0
        source = "Inherited from Personal Finance Setup"
    else:
        weekly = float(STEP4_DEFAULT_WEEKLY_CONTRIBUTION)
        monthly = float(STEP4_DEFAULT_MONTHLY_CONTRIBUTION)
        source = "Demo default"

    ctx["weekly_equivalent"] = float(weekly)
    ctx["monthly_contribution"] = float(monthly)
    ctx["contribution_bridge_source"] = source

    # Keep common aliases in sync for downstream Step 5/6 code that reads the
    # shared investment_context directly from session_state.
    existing_session_ctx = st.session_state.get("investment_context", {})
    if isinstance(existing_session_ctx, dict):
        merged_session_ctx = dict(existing_session_ctx)
        merged_session_ctx.update(ctx)
        st.session_state["investment_context"] = merged_session_ctx
    else:
        st.session_state["investment_context"] = dict(ctx)

    return ctx


def _deployment_supported_universe_sizes() -> list[int]:
    return [int(x) for x in UNIVERSE_SIZES if int(x) <= DEPLOYMENT_MAX_VISIBLE_UNIVERSE_SIZE]


PHILOSOPHY_FEEL_CARDS = {
    "Growth": {
        "one_liner": "Higher growth potential, but a rougher ride.",
        "expected_return": "8%–12%",
        "volatility": "High (≈15%–25%)",
        "typical_size": "50–100 assets",
        "example_path": "+25% → -15% → +12%",
        "summary": "Higher upside potential, with a bumpier ride.",
        "detail": "In strong years this approach may capture significant upside. In weaker periods, it can experience deeper drawdowns and sharper swings.",
        "over_time": "3y → outcomes vary widely · 5y → growth potential becomes clearer · 10y → requires discipline through volatility",
        "real_world": "Closest to equity-heavy / aggressive growth approaches.",
    },
    "Balanced": {
        "one_liner": "Growth with controlled swings.",
        "expected_return": "6%–9%",
        "volatility": "Medium (≈10%–15%)",
        "typical_size": "25–50 assets",
        "example_path": "+15% → -8% → +10%",
        "summary": "A middle ground between growth and control.",
        "detail": "Participates in upside while aiming to limit extreme behaviour and keep the path more investable over time.",
        "over_time": "3y → more stable experience · 5y → balanced growth vs risk · 10y → often the most sustainable",
        "real_world": "Closest to diversified multi-asset portfolios (e.g. 60/40 style).",
    },
    "Defensive": {
        "one_liner": "Slower, but easier to stick with.",
        "expected_return": "4%–7%",
        "volatility": "Low to medium-low (≈5%–10%)",
        "typical_size": "12–25 assets",
        "example_path": "+10% → -5% → +6%",
        "summary": "More stability, lower upside, fewer shocks.",
        "detail": "Prioritises smoother progression and drawdown control, even if that means giving up part of the upside in stronger markets.",
        "over_time": "3y → smoother trajectory · 5y → may lag in strong markets · 10y → more stable, less aggressive",
        "real_world": "Closest to conservative / income-oriented portfolio styles.",
    },
}


def _default_asset_start_date() -> dt.date:
    raw = st.session_state.get(ASSET_START_DATE)
    if isinstance(raw, dt.date):
        return raw
    return dt.date(2005, 1, 1)


def _default_asset_end_date() -> dt.date:
    raw = st.session_state.get(ASSET_END_DATE)
    if isinstance(raw, dt.date):
        return raw
    return dt.date.today()

def _render_philosophy_feel_block(philosophy: str) -> None:
    card = dict(PHILOSOPHY_FEEL_CARDS.get(str(philosophy), PHILOSOPHY_FEEL_CARDS["Balanced"]))

    st.markdown("### How this investment philosophy may feel over time")

    c1, c2, c3 = st.columns([1.25, 1.0, 1.0])
    with c1:
        st.markdown(f"**{philosophy}**")
        st.caption(card.get("summary", ""))
        st.write(card.get("detail", ""))
    with c2:
        st.markdown("**Expected return (long-run)**")
        st.write(card.get("expected_return", "—"))
        st.caption("Illustrative range, not a guaranteed outcome.")
        st.markdown("**Volatility**")
        st.write(card.get("volatility", "—"))
    with c3:
        st.markdown("**Typical universe size**")
        st.write(card.get("typical_size", "—"))
        st.markdown("**Example yearly path**")
        st.write(card.get("example_path", "—"))

    st.caption("Over time · " + str(card.get("over_time", "")))
    st.caption(
        "CAGR is the estimated long-term average yearly growth. "
        "Volatility describes how bumpy the journey may feel."
    )

    with st.expander("What do CAGR and volatility mean?", expanded=False):
        st.write(
            "**CAGR** is a long-term average yearly growth rate. It does not mean the portfolio "
            "grows by that amount every year."
        )
        st.write(
            "**Volatility** describes how much the portfolio may move up and down along the way. "
            "Higher expected returns usually come with wider swings."
        )
        st.caption("These figures are educational style ranges, not guarantees or direct benchmarks.")

    st.info("In plain English: " + str(card.get("one_liner", "")))

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Why this matters**")
        st.write(
            "This philosophy shapes the type of assets selected, the usual size of the universe, "
            "and the amount of volatility the user should expect to tolerate."
        )
        st.caption(card.get("real_world", ""))

    with d2:
        st.markdown("**Long-term mindset**")
        st.write(
            "This setup is designed for regular monthly investing, not short-term trading. "
            "Short-term swings are expected, but the system evaluates assets with a long-term portfolio mindset."
        )
        st.caption("Step 6 will show how this strategy may develop over time.")

def _sync_widget_defaults_from_philosophy(philosophy: str) -> None:
    bundle = apply_investment_philosophy_bundle(philosophy)
    default_size = int(bundle.get("default_universe_size", 25) or 25)
    preferred_strategy = resolve_recommended_strategy_for_size(philosophy, default_size)
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)
    st.session_state["universe_size_input"] = int(default_size)
    st.session_state["universe_strategy_input"] = str(preferred_strategy)
    st.session_state["show_all_sizes_input"] = False
    st.session_state["show_all_strategies_input"] = False
    st.session_state["step5_template"] = str(rec_template)
    st.session_state["step5_style"] = str(rec_style)


def _build_visible_size_options(philosophy: str, show_all_sizes: bool, current_size: int) -> list[int]:
    supported_sizes = _deployment_supported_universe_sizes()
    if show_all_sizes:
        options = list(supported_sizes)
    else:
        options = [
            int(x)
            for x in recommended_universe_sizes_for_philosophy(philosophy)
            if int(x) in set(supported_sizes)
        ]

    if not options:
        options = [25] if 25 in supported_sizes else list(supported_sizes[:1])

    try:
        current_value = int(current_size)
    except Exception:
        current_value = int(options[0])

    # Preserve current selections only when they are supported by the cached
    # deployment bundle. This intentionally prevents stale 150/250 values from
    # remaining visible after the project moved to a 106-asset cached panel.
    if current_value in supported_sizes and current_value not in options:
        options.append(current_value)

    return sorted(set(int(x) for x in options))


def _build_visible_strategy_options(philosophy: str, selected_size: int, show_all_strategies: bool, current_strategy: str) -> list[str]:
    allowed = list(allowed_universe_strategies_for_size(selected_size))
    if show_all_strategies:
        options = list(allowed)
    else:
        prefs = [x for x in preferred_universe_strategies_for_philosophy(philosophy) if x in allowed]
        options = prefs or list(allowed)
    if current_strategy not in options and current_strategy in allowed:
        options.append(str(current_strategy))
    return list(dict.fromkeys(options))


def _render_size_strategy_guidance(philosophy: str, selected_size: int, selected_strategy: str) -> None:
    size_status = build_universe_size_status(philosophy, selected_size)
    recommended_sizes = list(size_status.get("recommended_sizes", []))
    recommended_strategy = resolve_recommended_strategy_for_size(philosophy, selected_size)

    st.caption(
        f"Size guidance: ideal **{int(size_status.get('ideal_size', selected_size))} assets** · "
        f"preferred subset **{', '.join(str(x) for x in recommended_sizes)}** · "
        f"current status **{size_status.get('status', 'Unknown')}**."
    )
    if selected_strategy == recommended_strategy:
        st.caption(f"Strategy guidance: **{selected_strategy}** matches the current {philosophy} bundle.")
    else:
        st.caption(f"Strategy guidance: preferred for {philosophy} is **{recommended_strategy}**; current selection is **{selected_strategy}**.")


def _render_strategy_setup_guidance(philosophy: str) -> None:
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)
    template_options = allowed_strategy_templates_for_philosophy(philosophy)
    style_options = allowed_style_presets_for_philosophy(philosophy, rec_template)
    st.caption(f"Preferred strategy setup for {philosophy}: template={rec_template} · style={rec_style}.")
    st.caption("Allowed templates in normal mode: " + ", ".join(template_options) + ".")
    st.caption(f"Allowed styles for {rec_template}: " + ", ".join(style_options) + ".")


def _seed_custom_assets_if_needed(custom_enabled: bool, preset_assets: list[str]) -> None:
    prev_enabled = bool(st.session_state.get("_step4_prev_custom_enabled", False))
    if custom_enabled and not prev_enabled:
        existing = parse_custom_assets(st.session_state.get(CUSTOM_UNIVERSE_TEXT, ""))
        seed_assets = existing or list(preset_assets)
        st.session_state["custom_asset_list_input"] = list(seed_assets)
        st.session_state[CUSTOM_UNIVERSE_TEXT] = ", ".join(seed_assets)
    st.session_state["_step4_prev_custom_enabled"] = bool(custom_enabled)


def _render_custom_asset_editor(selected_size: int, selected_strategy: str, custom_enabled: bool, preset_assets: list[str]) -> str:
    custom_allowed = int(selected_size) in {12, 25}
    if not custom_allowed:
        st.caption("Custom editing is only enabled for 12-asset and 25-asset universes in this modular version.")
        return ""

    if not custom_enabled:
        last_custom = parse_custom_assets(st.session_state.get(CUSTOM_UNIVERSE_TEXT, ""))
        if last_custom:
            st.caption("Custom editing is off, so the preset universe is active. Your last custom selection is being kept in memory.")
        return ""

    _seed_custom_assets_if_needed(custom_enabled, preset_assets)

    candidate_pool = build_strategy_candidate_pool(selected_size, selected_strategy)
    current_custom_assets = parse_custom_assets(st.session_state.get(CUSTOM_UNIVERSE_TEXT, ""))
    raw_widget_assets = list(st.session_state.get("custom_asset_list_input", [])) or current_custom_assets or list(preset_assets)
    widget_assets = [normalize_asset_ticker(x) for x in raw_widget_assets if normalize_asset_ticker(x)]
    pool_assets = [normalize_asset_ticker(x) for x in candidate_pool if normalize_asset_ticker(x)]
    all_options = list(dict.fromkeys([*widget_assets, *pool_assets]))
    st.session_state["custom_asset_list_input"] = [x for x in widget_assets if x in all_options]

    selected_custom_assets = st.multiselect(
        "Primary custom asset list",
        options=all_options,
        key="custom_asset_list_input",
        format_func=asset_display_label,
        help="Select a custom override basket from the strategy-compatible asset preparation pool. Labels show the ticker plus a plain-English asset description, but the engine still uses the ticker only.",
    )
    custom_assets = [normalize_asset_ticker(x) for x in list(selected_custom_assets or []) if normalize_asset_ticker(x)]
    custom_text = ", ".join(custom_assets)
    st.caption(f"Preset size: {int(selected_size)} · Custom selected size: {len(custom_assets)}")
    st.caption("Turning this off returns to the preset universe, but keeps your last custom selection in case you turn it back on.")
    if custom_assets:
        st.success("Custom universe override active: this universe is currently using an override basket instead of the generated preset.")
        c1, c2 = st.columns([1.2, 2.8])
        with c1:
            if st.button("Return to preset universe", key="step4_return_to_preset_universe"):
                st.session_state["universe_custom_enabled_input"] = False
                st.session_state[UNIVERSE_CUSTOM_ENABLED] = False
                st.rerun()
        with c2:
            st.caption("This universe is currently running from a custom override basket. Clear it to go back to the generated preset universe.")
    return custom_text


def _render_combined_yahoo_notice(selected_assets: list[str], candidate_assets: list[str]) -> None:
    universe_union = sorted(set(selected_assets) | set(candidate_assets))
    if not universe_union:
        return
    st.session_state["_step4_yahoo_tickers_to_download"] = list(universe_union)
    if len(universe_union) >= 50 and len(universe_union) <= 100:
        st.info(f"Data preparation basket: {len(universe_union)} assets loaded for a formal scale test.")
    elif len(universe_union) > 100:
        st.info(f"Data preparation basket: {len(universe_union)} assets loaded for extended research coverage.")
    else:
        st.info(f"Data preparation basket: {len(universe_union)} assets.")


def _asset_label_list(assets: list[str]) -> str:
    cleaned = [normalize_asset_ticker(x) for x in list(assets or []) if normalize_asset_ticker(x)]
    return ", ".join(asset_display_label(x) for x in cleaned)


def _panel_preview_for_display(panel_df: pd.DataFrame, rows: int = 50) -> pd.DataFrame:
    preview = panel_df.head(int(rows)).copy()
    if "asset" in preview.columns:
        preview["asset"] = preview["asset"].apply(asset_display_label)
    return preview


def _render_data_panel_preview(panel_df: pd.DataFrame, *, height: int = 320) -> None:
    st.dataframe(_panel_preview_for_display(panel_df), use_container_width=True, hide_index=True, height=height)


def _render_asset_diagnostics(*, requested_assets: list[str], ready_panel: pd.DataFrame | None) -> None:
    requested = [normalize_asset_ticker(x) for x in list(requested_assets or []) if normalize_asset_ticker(x)]
    if requested:
        st.caption(f"Tickers requested ({len(requested)}): " + _asset_label_list(requested))

    loaded: list[str] = []
    if isinstance(ready_panel, pd.DataFrame) and not ready_panel.empty and "asset" in ready_panel.columns:
        loaded = sorted(set(normalize_asset_ticker(x) for x in ready_panel["asset"].dropna().tolist() if normalize_asset_ticker(x)))

    if requested and loaded and set(loaded) != set(requested):
        st.caption(f"Loaded subset ({len(loaded)}): " + _asset_label_list(loaded))
        missing = [x for x in requested if x not in set(loaded)]
        if missing:
            st.caption("Missing from Yahoo: " + _asset_label_list(missing))
    elif (not requested) and loaded:
        st.caption(f"Assets loaded ({len(loaded)}): " + _asset_label_list(loaded))



# -----------------------------------------------------------------------------
# Step 4 dashboard-style renderer
# -----------------------------------------------------------------------------

def _render_step4_intro_banners() -> None:
    """Render a compact orientation note.

    Step 4 is a setup screen, not a lesson page. Keep the main surface focused
    on the two real decisions: strategy posture and asset universe.
    """
    st.info(
        "**Purpose:** choose an investment philosophy, a matching asset universe, "
        "and the cached market-data panel used by the Strategy Engine. No optimisation happens here."
    )

def _render_contribution_bridge_card(investment_context: dict) -> None:
    """Render a low-height contribution strip instead of a large card."""
    source = str(
        investment_context.get("contribution_bridge_source", "Inherited from Personal Finance Setup")
        or "Inherited from Personal Finance Setup"
    )
    monthly = _safe_float(investment_context.get("monthly_contribution", 0.0), 0.0)
    weekly = _safe_float(investment_context.get("weekly_equivalent", 0.0), 0.0)
    source_text = "Demo default" if source == "Demo default" else "Personal Finance Setup"
    source_note = "fallback" if source == "Demo default" else "inherited"

    st.markdown(
        f"""
        <div style="border:1px solid rgba(49,51,63,0.14); border-radius:16px; padding:0.95rem 1.05rem; background:#ffffff; margin:0.7rem 0 0.85rem 0;">
            <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:1.25rem; align-items:center;">
                <div>
                    <div style="font-size:0.78rem; color:#64748b; font-weight:500;">Monthly contribution</div>
                    <div style="font-weight:800; font-size:1.08rem; margin-top:0.25rem;">£{monthly:,.0f}</div>
                </div>
                <div>
                    <div style="font-size:0.78rem; color:#64748b; font-weight:500;">Weekly equivalent</div>
                    <div style="font-weight:800; font-size:1.08rem; margin-top:0.25rem;">£{weekly:,.0f}</div>
                </div>
                <div>
                    <div style="font-size:0.78rem; color:#64748b; font-weight:500;">Source</div>
                    <div style="font-weight:800; font-size:1.02rem; margin-top:0.25rem;">{source_text}</div>
                    <div style="font-size:0.76rem; color:#64748b; margin-top:0.16rem;">{source_note}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_strategy_badge(label: str, value: str) -> None:
    st.markdown(
        f"<div style='display:flex; align-items:center; justify-content:space-between; gap:0.75rem; margin:0.35rem 0;'>"
        f"<span style='font-size:0.85rem; color:#64748b;'>{label}</span>"
        f"<span style='background:#dcfce7; color:#166534; padding:0.22rem 0.55rem; border-radius:999px; font-weight:650; font-size:0.78rem;'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_philosophy_strategy_card(current_philosophy: str, bundle: dict, rec_template: str, rec_style: str) -> None:
    """Render a lighter strategy-choice block.

    Avoid large metrics inside narrow columns; they truncate in centered layout.
    Use compact badges and keep explanations behind an expander.
    """
    card = dict(PHILOSOPHY_FEEL_CARDS.get(str(current_philosophy), PHILOSOPHY_FEEL_CARDS["Balanced"]))
    with st.container(border=True):
        st.markdown("### 1. Strategy choice")
        st.caption("Choose the overall risk posture. The default engine bundle follows this choice.")

        left, right = st.columns([1.05, 1.0])
        with left:
            selected_philosophy = st.selectbox(
                "Investment philosophy",
                options=INVESTMENT_PHILOSOPHY_OPTIONS,
                index=INVESTMENT_PHILOSOPHY_OPTIONS.index(current_philosophy),
                key="investment_philosophy_step4",
                help="The main posture that guides universe size, strategy defaults and projection tone.",
            )
            if selected_philosophy != current_philosophy:
                _sync_widget_defaults_from_philosophy(selected_philosophy)
                st.rerun()

            st.markdown(f"**{current_philosophy}:** {card.get('one_liner', '')}")
            st.caption(card.get("detail", ""))
            st.markdown(
                f"""
                <div style="display:flex; flex-wrap:wrap; gap:0.45rem; margin-top:0.55rem;">
                    <span style="background:#f1f5f9; padding:0.28rem 0.55rem; border-radius:999px; font-size:0.78rem;"><b>Return</b> {card.get('expected_return', '—')}</span>
                    <span style="background:#f1f5f9; padding:0.28rem 0.55rem; border-radius:999px; font-size:0.78rem;"><b>Volatility</b> {card.get('volatility', '—')}</span>
                    <span style="background:#f1f5f9; padding:0.28rem 0.55rem; border-radius:999px; font-size:0.78rem;"><b>Typical size</b> {card.get('typical_size', '—')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.markdown("**Default strategy bundle**")
            _render_strategy_badge("Template", str(rec_template))
            _render_strategy_badge("Style", str(rec_style))
            _render_strategy_badge("Projection", str(bundle.get("projection_profile", current_philosophy)))
            st.caption("Coherent with the selected philosophy.")

        with st.expander("Why this fits / terms explained", expanded=False):
            st.write(
                f"The {current_philosophy} setup keeps the philosophy, universe size, strategy template "
                "and projection tone aligned before the Strategy Engine run."
            )
            st.write("**CAGR** is a long-term average annual growth rate; it is not a yearly guarantee.")
            st.write("**Volatility** describes how bumpy the journey may feel along the way.")
            st.caption(card.get("over_time", ""))
            st.caption(card.get("real_world", ""))
            st.divider()
            _render_strategy_setup_guidance(current_philosophy)

def _render_alignment_message(current_combo_status: str, current_philosophy: str, rec_template: str) -> None:
    if current_combo_status == "recommended":
        return
    if current_combo_status == "allowed":
        st.info("Strategy setup is allowed for this risk profile, but it is not the primary recommended combo.")
    else:
        st.warning("Current strategy setup is outside the allowed normal-mode space for this risk profile. It will be corrected before the Strategy Engine runs.")


def _render_universe_mix_compact(selected_assets: list[str]) -> None:
    """Render universe mix details without creating a nested expander.

    This helper is called from inside the Step 4 advanced setup expander.
    Streamlit does not allow expanders inside expanders, so the table content
    must be rendered directly here rather than wrapped in another st.expander.
    """
    st.markdown("**Universe mix and full composition**")
    mix_df, mix_summary = build_universe_mix(selected_assets)
    detail_df = build_universe_mix_detail(selected_assets)
    rendered = show_table_if_not_empty(
        mix_df,
        empty_message="No universe mix available yet.",
        use_container_width=True,
        hide_index=True,
    )
    if rendered:
        n_assets = int(mix_summary.get("n_assets", len(selected_assets)) or len(selected_assets))
        group_count = int(mix_summary.get("group_count", 0) or 0)
        if group_count <= 0 and isinstance(mix_df, pd.DataFrame) and "group" in mix_df.columns:
            group_count = int(mix_df["group"].nunique())
        classified_share = float(mix_summary.get("classified_share", 0.0) or 0.0)
        st.caption(f"{n_assets} assets across {group_count} groups. Classified share: {classified_share:.0%}.")
    if isinstance(detail_df, pd.DataFrame) and not detail_df.empty:
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
    else:
        st.info("No detailed universe taxonomy is available yet.")





def _render_asset_universe_details(selected_assets: list[str]) -> None:
    """Render the selected universe composition in a compact diagnostics block."""
    mix_df, mix_summary = build_universe_mix(selected_assets)
    detail_df = build_universe_mix_detail(selected_assets)

    n_assets = int(mix_summary.get("n_assets", len(selected_assets)) or len(selected_assets)) if isinstance(mix_summary, dict) else len(selected_assets)
    group_count = int(mix_summary.get("group_count", 0) or 0) if isinstance(mix_summary, dict) else 0
    if group_count <= 0 and isinstance(mix_df, pd.DataFrame) and not mix_df.empty and "group" in mix_df.columns:
        group_count = int(mix_df["group"].nunique())
    classified_share = float(mix_summary.get("classified_share", 0.0) or 0.0) if isinstance(mix_summary, dict) else 0.0

    st.markdown("**Universe mix**")
    if selected_assets:
        st.caption(f"{n_assets} assets across {group_count or 'multiple'} groups. Classified share: {classified_share:.0%}.")
    show_table_if_not_empty(
        mix_df,
        empty_message="No universe mix available yet.",
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Full asset list**")
    if isinstance(detail_df, pd.DataFrame) and not detail_df.empty:
        st.dataframe(detail_df, use_container_width=True, hide_index=True, height=260)
    else:
        st.info("No detailed universe taxonomy is available yet.")

def _render_investment_setup_card(current_philosophy: str, bundle: dict, rec_template: str, rec_style: str) -> tuple[int, str, list[str], list[str]]:
    """Render the main Step 4 decision as one compact card.

    The visual hierarchy is intentionally vertical:
    1. choose the risk posture;
    2. choose the asset universe that should feed the Strategy Engine;
    3. show one compact current-setup summary.

    Keep this as UI/layout only. The asset selection, state sync and panel
    construction logic below stays unchanged.
    """
    card = dict(PHILOSOPHY_FEEL_CARDS.get(str(current_philosophy), PHILOSOPHY_FEEL_CARDS["Balanced"]))

    current_size = int(
        st.session_state.get(
            "universe_size_input",
            st.session_state.get("universe_size", bundle["default_universe_size"]),
        )
        or bundle["default_universe_size"]
    )

    with st.container(border=True):
        # ------------------------------------------------------------------
        # 1) Risk profile comes first. It drives the recommended universe.
        # ------------------------------------------------------------------
        st.markdown("### Risk profile")
        profile_left, profile_right = st.columns([0.47, 0.53], gap="large")

        with profile_left:
            selected_philosophy = st.selectbox(
                "Investment philosophy",
                options=INVESTMENT_PHILOSOPHY_OPTIONS,
                index=INVESTMENT_PHILOSOPHY_OPTIONS.index(current_philosophy),
                key="investment_philosophy_step4",
                help="The main posture that guides universe size, strategy defaults and projection tone.",
            )
            if selected_philosophy != current_philosophy:
                _sync_widget_defaults_from_philosophy(selected_philosophy)
                st.rerun()

        profile_summary = " ".join(
            part.strip()
            for part in [str(card.get("one_liner", "") or ""), str(card.get("detail", "") or "")]
            if part.strip()
        )

        with profile_right:
            if profile_summary:
                st.markdown(
                    f"""
                    <div style="margin-top:1.62rem; color:#475569; font-size:0.95rem; line-height:1.55;">
                        {profile_summary}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown(
            f"""
            <div style="margin-top:0.95rem; color:#334155; font-size:0.91rem; line-height:1.55; width:100%; display:flex; flex-wrap:wrap; gap:0.35rem 0.55rem;">
                <span><strong>Return:</strong> {card.get('expected_return', '—')}</span>
                <span style="color:#94a3b8;">·</span>
                <span><strong>Volatility:</strong> {card.get('volatility', '—')}</span>
                <span style="color:#94a3b8;">·</span>
                <span><strong>Typical universe:</strong> {card.get('typical_size', '—')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ------------------------------------------------------------------
        # 2) Universe basket follows the risk profile. Checkboxes sit above
        #    the selects they affect, so the relationship is visually clear.
        # ------------------------------------------------------------------
        st.markdown("### Universe basket")
        u1, u2 = st.columns(2)

        with u1:
            show_all_sizes = bool(
                st.checkbox(
                    "Show all sizes",
                    value=bool(st.session_state.get("show_all_sizes_input", False)),
                    key="show_all_sizes_input",
                    help="Show all universe sizes supported by the cached deployment panel.",
                )
            )
            size_options = _build_visible_size_options(current_philosophy, show_all_sizes, current_size)
            if current_size not in size_options:
                current_size = int(bundle["default_universe_size"])
                st.session_state["universe_size_input"] = current_size

            selected_size = st.selectbox(
                "Universe size",
                size_options,
                index=size_options.index(int(st.session_state.get("universe_size_input", current_size))),
                key="universe_size_input",
                help="The public deployment demo supports sizes up to 100 assets from the cached bundle.",
            )

        current_strategy = str(
            st.session_state.get(
                "universe_strategy_input",
                st.session_state.get(
                    "universe_strategy",
                    resolve_recommended_strategy_for_size(current_philosophy, selected_size),
                ),
            )
            or resolve_recommended_strategy_for_size(current_philosophy, selected_size)
        )

        with u2:
            show_all_strategies = bool(
                st.checkbox(
                    "Show all strategies",
                    value=bool(st.session_state.get("show_all_strategies_input", False)),
                    key="show_all_strategies_input",
                    help="Show all strategies available for the selected size instead of only the philosophy-preferred subset.",
                )
            )
            strategy_options = _build_visible_strategy_options(
                current_philosophy,
                int(selected_size),
                show_all_strategies,
                current_strategy,
            )
            if current_strategy not in strategy_options:
                current_strategy = resolve_recommended_strategy_for_size(current_philosophy, int(selected_size))
                st.session_state["universe_strategy_input"] = current_strategy

            selected_strategy = st.selectbox(
                "Universe strategy",
                strategy_options,
                index=strategy_options.index(str(st.session_state.get("universe_strategy_input", current_strategy))),
                key="universe_strategy_input",
            )

        preset_assets = build_generated_universe(selected_size, selected_strategy)
        custom_allowed = int(selected_size) in {12, 25}
        default_custom_enabled = (
            bool(st.session_state.get("universe_custom_enabled_input", st.session_state.get(UNIVERSE_CUSTOM_ENABLED, False)))
            if custom_allowed
            else False
        )
        custom_enabled = st.checkbox(
            "Use custom asset list",
            value=default_custom_enabled,
            disabled=not custom_allowed,
            key="universe_custom_enabled_input",
            help="Advanced: replace the recommended basket with a manual asset list. Available for 12- and 25-asset universes.",
        )
        if not custom_allowed:
            st.caption("Custom asset editing is available for 12- and 25-asset universes only.")

        custom_assets_text = ""
        if bool(custom_enabled):
            with st.expander("Edit custom asset list", expanded=True):
                custom_assets_text = _render_custom_asset_editor(
                    int(selected_size),
                    str(selected_strategy),
                    bool(custom_enabled),
                    list(preset_assets),
                )

        selected_assets, selection_source = resolve_universe_selection(
            selected_size,
            selected_strategy,
            custom_enabled,
            custom_assets_text,
        )
        sync_step4_state(
            philosophy=current_philosophy,
            universe_size=int(selected_size),
            strategy_name=str(selected_strategy),
            custom_enabled=bool(custom_enabled),
            custom_text=str(custom_assets_text),
            selected_assets=selected_assets,
            selection_source=selection_source,
        )

        current_combo_status = strategy_combo_status(
            current_philosophy,
            st.session_state.get("step5_template"),
            st.session_state.get("step5_style"),
        )
        _render_alignment_message(current_combo_status, current_philosophy, rec_template)

        mix_df, mix_summary = build_universe_mix(selected_assets)
        group_count = int(mix_summary.get("group_count", 0) or 0) if isinstance(mix_summary, dict) else 0
        if group_count <= 0 and isinstance(mix_df, pd.DataFrame) and not mix_df.empty and "group" in mix_df.columns:
            group_count = int(mix_df["group"].nunique())


    if selected_assets:
        source_label = "custom basket" if bool(custom_enabled) else "recommended basket"
        st.info(
            f"**Current setup:** {current_philosophy} risk profile · {len(selected_assets)} assets · "
            f"{selected_strategy} · {source_label}."
        )
    else:
        st.warning("No valid assets found yet for the primary universe.")

    candidate_assets = build_strategy_candidate_pool(selected_size, selected_strategy)
    return int(selected_size), str(selected_strategy), list(selected_assets), list(candidate_assets)

def _render_universe_selection_card(current_philosophy: str, bundle: dict, rec_template: str) -> tuple[int, str, list[str], list[str]]:
    current_size = int(st.session_state.get("universe_size_input", st.session_state.get("universe_size", bundle["default_universe_size"])) or bundle["default_universe_size"])
    show_all_sizes = bool(st.session_state.get("show_all_sizes_input", False))
    show_all_strategies = bool(st.session_state.get("show_all_strategies_input", False))
    size_options = _build_visible_size_options(current_philosophy, show_all_sizes, current_size)
    if current_size not in size_options:
        current_size = int(bundle["default_universe_size"])
        st.session_state["universe_size_input"] = current_size

    with st.container(border=True):
        st.markdown("### 2. Universe selection")
        st.caption("Choose the asset-universe size and construction style. Defaults are aligned with the philosophy above.")
        u1, u2 = st.columns(2)
        with u1:
            selected_size = st.selectbox(
                "Universe size",
                size_options,
                index=size_options.index(int(st.session_state.get("universe_size_input", current_size))),
                key="universe_size_input",
                help="The public deployment demo supports sizes up to 100 assets from the cached bundle.",
            )
            show_all_sizes = bool(
                st.checkbox(
                    "Show all supported sizes",
                    value=bool(st.session_state.get("show_all_sizes_input", False)),
                    key="show_all_sizes_input",
                    help="Show all universe sizes supported by the cached deployment panel.",
                )
            )
        current_strategy = str(
            st.session_state.get(
                "universe_strategy_input",
                st.session_state.get("universe_strategy", resolve_recommended_strategy_for_size(current_philosophy, selected_size)),
            )
            or resolve_recommended_strategy_for_size(current_philosophy, selected_size)
        )
        strategy_options = _build_visible_strategy_options(current_philosophy, int(selected_size), show_all_strategies, current_strategy)
        if current_strategy not in strategy_options:
            current_strategy = resolve_recommended_strategy_for_size(current_philosophy, int(selected_size))
            st.session_state["universe_strategy_input"] = current_strategy
        with u2:
            selected_strategy = st.selectbox(
                "Universe strategy",
                strategy_options,
                index=strategy_options.index(str(st.session_state.get("universe_strategy_input", current_strategy))),
                key="universe_strategy_input",
            )
            show_all_strategies = bool(
                st.checkbox(
                    "Show all strategies",
                    value=bool(st.session_state.get("show_all_strategies_input", False)),
                    key="show_all_strategies_input",
                    help="Show all strategies available for the selected size instead of only the philosophy-preferred subset.",
                )
            )

        preset_assets = build_generated_universe(selected_size, selected_strategy)
        custom_allowed = int(selected_size) in {12, 25}
        default_custom_enabled = bool(
            st.session_state.get("universe_custom_enabled_input", st.session_state.get(UNIVERSE_CUSTOM_ENABLED, False))
        ) if custom_allowed else False

        with st.expander("Custom asset list (advanced)", expanded=False):
            custom_enabled = st.checkbox(
                "Use custom asset list instead of the recommended universe",
                value=default_custom_enabled,
                disabled=not custom_allowed,
                key="universe_custom_enabled_input",
            )
            if custom_allowed:
                st.caption("Use only if you have specific assets to include or exclude.")
            else:
                st.caption("Custom editing is only enabled for 12- and 25-asset universes.")
            custom_assets_text = _render_custom_asset_editor(int(selected_size), str(selected_strategy), bool(custom_enabled), list(preset_assets))

        selected_assets, selection_source = resolve_universe_selection(selected_size, selected_strategy, custom_enabled, custom_assets_text)
        sync_step4_state(
            philosophy=current_philosophy,
            universe_size=int(selected_size),
            strategy_name=str(selected_strategy),
            custom_enabled=bool(custom_enabled),
            custom_text=str(custom_assets_text),
            selected_assets=selected_assets,
            selection_source=selection_source,
        )

        current_combo_status = strategy_combo_status(current_philosophy, st.session_state.get("step5_template"), st.session_state.get("step5_style"))
        _render_alignment_message(current_combo_status, current_philosophy, rec_template)

        mix_df, mix_summary = build_universe_mix(selected_assets)
        group_count = int(mix_summary.get("group_count", 0) or 0) if isinstance(mix_summary, dict) else 0
        if group_count <= 0 and isinstance(mix_df, pd.DataFrame) and not mix_df.empty and "group" in mix_df.columns:
            group_count = int(mix_df["group"].nunique())
        actual_size = len(selected_assets)
        if selected_assets:
            st.info(
                f"**Universe preview:** {actual_size} assets · {selected_strategy} · "
                f"{group_count or 'multiple'} asset groups. Full tickers and composition are available below."
            )
        else:
            st.warning("No valid assets found yet for the primary universe.")

        with st.expander("Universe composition and engine defaults", expanded=False):
            _render_size_strategy_guidance(current_philosophy, int(selected_size), str(selected_strategy))
            st.divider()
            _render_strategy_setup_guidance(current_philosophy)
            st.divider()
            st.caption(build_universe_preview_text(selected_assets))
            _render_universe_mix_compact(selected_assets)

    candidate_assets = build_strategy_candidate_pool(selected_size, selected_strategy)
    return int(selected_size), str(selected_strategy), list(selected_assets), list(candidate_assets)
def _render_market_data_setup_card(selected_assets: list[str], candidate_assets: list[str]) -> None:
    """Keep market-data setup available, but out of the main Step 4 flow.

    The hosted demo normally uses cached Yahoo-generated panels, so this is
    infrastructure rather than a primary user decision. The expander still
    executes its widgets on every rerun, but keeps the screen visually focused
    on the investment setup card.
    """
    st.session_state[ASSET_SOURCE_MODE] = "yahoo"
    st.session_state[ASSET_UPLOADED_FILE] = None
    freq_options = ["monthly", "weekly", "daily"]
    current_freq = str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly")
    if current_freq not in freq_options:
        current_freq = "monthly"
    if str(st.session_state.get(ASSET_RETURN_FREQUENCY) or "") not in freq_options:
        st.session_state[ASSET_RETURN_FREQUENCY] = "monthly"
    widget_freq = str(st.session_state.get("asset_return_frequency_input") or "")
    if widget_freq not in freq_options:
        st.session_state["asset_return_frequency_input"] = current_freq if current_freq in freq_options else "monthly"

    freq = str(st.session_state.get("asset_return_frequency_input", current_freq) or current_freq)
    if freq not in freq_options:
        freq = "monthly"
        st.session_state["asset_return_frequency_input"] = "monthly"
    auto_adjust = bool(st.session_state.get(ASSET_AUTO_ADJUST, True))

    if selected_assets:
        union_assets = sorted(set(selected_assets) | set(candidate_assets))
        st.session_state["_step4_yahoo_tickers_to_download"] = list(union_assets)
        market_caption = f"Cached deployment panel · data-preparation basket: {len(union_assets)} assets."
    else:
        market_caption = "No valid tickers have been selected yet for market-data preparation."

    with st.expander("Universe and data details", expanded=False):
        _render_asset_universe_details(selected_assets)
        st.divider()
        st.markdown("**Market-data setup**")
        if selected_assets:
            st.info(market_caption)
        else:
            st.warning(market_caption)

        st.caption(
            "Hosted demo mode uses cached Yahoo-generated panels by default, avoiding live Yahoo/rate-limit issues. "
            "Monthly is the recommended deployment frequency."
        )

        d1, d2 = st.columns(2)
        with d1:
            start_date = st.date_input("Panel start date", value=_default_asset_start_date(), key="asset_start_date_input")
        with d2:
            end_date = st.date_input("Panel end date", value=_default_asset_end_date(), key="asset_end_date_input")

        c1, c2 = st.columns([1.0, 1.15])
        with c1:
            freq = st.selectbox(
                "Return frequency",
                options=freq_options,
                index=freq_options.index(freq),
                key="asset_return_frequency_input",
            )
        with c2:
            auto_adjust = st.checkbox(
                "Use Yahoo auto-adjusted prices (live/local download only)",
                value=bool(st.session_state.get(ASSET_AUTO_ADJUST, True)),
                key="asset_auto_adjust_input",
            )
        st.caption("Auto-adjust only affects live/local Yahoo downloads; cached deployment CSVs are already prepared.")

    st.session_state[ASSET_START_DATE] = start_date
    st.session_state[ASSET_END_DATE] = end_date
    st.session_state[ASSET_RETURN_FREQUENCY] = freq
    st.session_state[ASSET_AUTO_ADJUST] = bool(auto_adjust)


def _render_panel_status_and_diagnostics(panel_error: str, selected_strategy: str) -> None:
    """Render optional technical market-data tools when explicitly requested."""
    if not bool(st.session_state.get(STEP4_SHOW_MARKET_DATA_TOOLS, False)):
        return

    ready_panel = st.session_state.get(ASSET_PANEL_DF)
    ready_asset_count = 0
    panel_rows = 0
    start_txt = "—"
    end_txt = "—"
    if isinstance(ready_panel, pd.DataFrame) and not ready_panel.empty:
        panel_rows = int(len(ready_panel))
        if "asset" in ready_panel.columns:
            ready_asset_count = int(ready_panel["asset"].nunique())
        if "date" in ready_panel.columns:
            try:
                start_txt = str(ready_panel["date"].min().date())
                end_txt = str(ready_panel["date"].max().date())
            except Exception:
                start_txt = str(ready_panel["date"].min())
                end_txt = str(ready_panel["date"].max())

    timings = dict(st.session_state.get(STEP4_PANEL_TIMINGS_KEY, {}) or {})
    total_seconds = _first_positive_value(
        timings.get("deployment_panel_total_seconds"),
        timings.get("resolve_total_seconds"),
        timings.get("total_seconds"),
    )

    with st.expander("Technical market-data tools", expanded=True):
        st.caption(
            "Technical preview, audit details, and CSV downloads. "
            "This section is hidden during the normal setup flow."
        )
        if st.button(
            "Hide diagnostics & downloads",
            key="step4_hide_market_data_tools_button",
            use_container_width=True,
        ):
            st.session_state[STEP4_SHOW_MARKET_DATA_TOOLS] = False
            st.rerun()

        if not bool(st.session_state.get(ASSET_PANEL_READY, False)):
            st.error("The Strategy Engine is unavailable until the market-data panel is ready.")
            if panel_error:
                st.caption(f"Latest panel error: {panel_error}")

        s1, s2, s3 = st.columns(3)
        with s1:
            st.caption("Coverage")
            st.write(f"**{start_txt} → {end_txt}**" if start_txt != "—" else "**—**")
        with s2:
            st.caption("Load time")
            st.write(f"**{total_seconds:.2f}s**" if total_seconds > 0 else "**—**")
        with s3:
            st.caption("Source")
            st.write("**Cached panel**")

        if isinstance(ready_panel, pd.DataFrame) and not ready_panel.empty:
            st.markdown("**Prepared panel preview**")
            _render_data_panel_preview(ready_panel, height=300)
        else:
            st.info("No panel preview available yet.")

        timing_rows = [
            ("Prepared panel read", timings.get("deployment_panel_read_seconds")),
            ("Raw daily read", timings.get("deployment_raw_daily_read_seconds")),
            ("Raw weekly read", timings.get("deployment_raw_weekly_read_seconds")),
            ("Yahoo download", timings.get("download_yahoo_seconds")),
            ("Return panel build", timings.get("return_panel_seconds")),
            ("Total resolve", timings.get("resolve_total_seconds") or timings.get("deployment_panel_total_seconds")),
        ]
        timing_display = []
        for label, raw_value in timing_rows:
            try:
                value = float(raw_value)
            except Exception:
                continue
            timing_display.append({"metric": label, "value": f"{value:.2f}s"})

        if "reuse_existing_panel" in timings:
            timing_display.append({"metric": "Cache reused", "value": str(bool(timings.get("reuse_existing_panel")))})
        if timings.get("deployment_cache_mode"):
            timing_display.append({"metric": "Cache mode", "value": str(timings.get("deployment_cache_mode"))})

        if timing_display:
            st.markdown("**Market-data timing breakdown**")
            st.dataframe(pd.DataFrame(timing_display), use_container_width=True, hide_index=True, height=280)
        else:
            st.caption("Timing details appear after the panel has been prepared or reused.")

        ready_panel_for_diag = st.session_state.get(ASSET_PANEL_DF)
        requested = [
            normalize_asset_ticker(x)
            for x in list(st.session_state.get("_step4_yahoo_tickers_to_download", []) or [])
            if normalize_asset_ticker(x)
        ]
        loaded: list[str] = []
        if isinstance(ready_panel_for_diag, pd.DataFrame) and not ready_panel_for_diag.empty and "asset" in ready_panel_for_diag.columns:
            loaded = sorted(
                set(
                    normalize_asset_ticker(x)
                    for x in ready_panel_for_diag["asset"].dropna().tolist()
                    if normalize_asset_ticker(x)
                )
            )
        missing = sorted(set(requested) - set(loaded)) if requested and loaded else []

        if requested or loaded:
            st.markdown("**Ticker audit**")
            t1, t2, t3 = st.columns(3)
            with t1:
                st.caption("Requested")
                st.write(f"**{len(requested) if requested else '—'}**")
            with t2:
                st.caption("Loaded in panel")
                st.write(f"**{len(loaded) if loaded else '—'}**")
            with t3:
                st.caption("Missing")
                st.write(f"**{len(missing)}**" if requested and loaded else "**—**")
            st.caption("Full selected-universe composition is shown in Universe and data details.")

        current_signature = st.session_state.get(STEP4_PANEL_SIGNATURE_KEY, "")
        if current_signature:
            st.caption("Panel input signature is stored for cache/rebuild checks.")

        st.markdown("**Downloads**")
        raw_source = str(st.session_state.get(STEP4_RAW_PANEL_SOURCE_KEY, "") or "")
        raw_daily_panel = st.session_state.get(STEP4_RAW_DAILY_PANEL_KEY)
        raw_weekly_panel = st.session_state.get(STEP4_RAW_WEEKLY_PANEL_KEY)
        if raw_source:
            st.caption(f"Raw return panel source: {raw_source}")

        d1, d2, d3 = st.columns(3)
        with d1:
            if isinstance(ready_panel, pd.DataFrame) and not ready_panel.empty:
                st.download_button(
                    "Prepared panel",
                    data=panel_df_to_csv_bytes(ready_panel),
                    file_name=panel_download_filename(
                        st.session_state.get(ASSET_PANEL_SOURCE_LABEL, "panel"),
                        selected_strategy,
                        str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly"),
                    ),
                    mime="text/csv",
                    key="step4_download_panel_csv",
                    use_container_width=True,
                )
            else:
                st.caption("Prepared panel unavailable")
        with d2:
            if isinstance(raw_daily_panel, pd.DataFrame) and not raw_daily_panel.empty:
                st.download_button(
                    "Raw daily panel",
                    data=panel_df_to_csv_bytes(raw_daily_panel),
                    file_name="yahoo_raw_daily_returns.csv",
                    mime="text/csv",
                    key="step4_download_raw_daily_panel_csv",
                    use_container_width=True,
                )
            else:
                st.caption("Raw daily panel unavailable")
        with d3:
            if isinstance(raw_weekly_panel, pd.DataFrame) and not raw_weekly_panel.empty:
                st.download_button(
                    "Raw weekly panel",
                    data=panel_df_to_csv_bytes(raw_weekly_panel),
                    file_name="yahoo_raw_weekly_returns.csv",
                    mime="text/csv",
                    key="step4_download_raw_weekly_panel_csv",
                    use_container_width=True,
                )
            else:
                st.caption("Raw weekly panel unavailable")

def render_step_4() -> None:
    section_header("Risk Profile and Asset Universe")
    _render_step4_intro_banners()
    plan_snapshot = coerce_snapshot()
    raw_investment_context = store_investment_context(plan_snapshot)
    investment_context = _resolve_contribution_bridge_context(raw_investment_context, plan_snapshot)
    maybe_apply_initial_universe_size_default()
    current_philosophy = get_canonical_investment_philosophy()
    bundle = apply_investment_philosophy_bundle(current_philosophy)
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(current_philosophy)
    selected_size, selected_strategy, selected_assets, candidate_assets = _render_investment_setup_card(
        current_philosophy,
        bundle,
        rec_template,
        rec_style,
    )
    _render_market_data_setup_card(selected_assets, candidate_assets)
    st.session_state[INVESTMENT_START_DATE_STABILITY_ENABLED] = False
    st.session_state[INVESTMENT_START_DATE_STABILITY_DATES] = []
    panel_df = None
    panel_source_label = ""
    panel_error = ""
    panel_timings = {}
    build_trigger = "auto"
    current_panel_signature = build_step4_panel_input_signature(selected_assets=selected_assets, candidate_assets=candidate_assets, source_mode="yahoo", start_date=st.session_state.get(ASSET_START_DATE), end_date=st.session_state.get(ASSET_END_DATE), frequency=str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly"), auto_adjust=bool(st.session_state.get(ASSET_AUTO_ADJUST, True)), uploaded_file=None)
    previous_panel_signature = str(st.session_state.get(STEP4_PANEL_SIGNATURE_KEY, "") or "")
    should_rebuild_panel = previous_panel_signature != current_panel_signature or (not bool(st.session_state.get(ASSET_PANEL_READY, False)))
    if should_rebuild_panel and selected_assets:
        try:
            panel_df, panel_source_label, union_caption = resolve_step4_asset_panel(selected_assets=selected_assets, candidate_assets=candidate_assets, source_mode="yahoo", start_date=st.session_state.get(ASSET_START_DATE), end_date=st.session_state.get(ASSET_END_DATE), frequency=str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly"), auto_adjust=bool(st.session_state.get(ASSET_AUTO_ADJUST, True)), uploaded_file=None)
            panel_timings = dict(st.session_state.get(STEP4_PANEL_TIMINGS_KEY, {}) or {})
            if union_caption:
                st.session_state["_step4_latest_union_caption"] = str(union_caption)
        except Exception as exc:
            panel_error = str(exc)
    else:
        ready_panel = st.session_state.get(ASSET_PANEL_DF)
        if isinstance(ready_panel, pd.DataFrame) and not ready_panel.empty:
            panel_df = ready_panel
            panel_source_label = str(st.session_state.get(ASSET_PANEL_SOURCE_LABEL, "") or "")
            panel_timings = dict(st.session_state.get(STEP4_PANEL_TIMINGS_KEY, {}) or {})
            panel_timings["reuse_existing_panel"] = True
    update_asset_panel_state(panel_df, panel_source_label, panel_error, panel_signature=current_panel_signature, timings=panel_timings, build_trigger=build_trigger if should_rebuild_panel else "reuse_existing_panel")

    # app.py renders the sidebar before the active step. When this screen prepares
    # or reuses a valid market-data panel, the sidebar has already rendered with
    # the previous readiness state. Trigger exactly one refresh per panel input
    # signature so the sidebar can show the ready state and diagnostics/downloads
    # button immediately, without creating a rerun loop.
    if bool(st.session_state.get(ASSET_PANEL_READY, False)) and current_panel_signature:
        last_sidebar_refresh_signature = str(
            st.session_state.get(STEP4_PANEL_SIDEBAR_REFRESH_SIGNATURE, "") or ""
        )
        if last_sidebar_refresh_signature != str(current_panel_signature):
            st.session_state[STEP4_PANEL_SIDEBAR_REFRESH_SIGNATURE] = str(current_panel_signature)
            st.rerun()

    _render_panel_status_and_diagnostics(panel_error, selected_strategy)
    payload = build_step4_universe_payload_from_state()
    st.markdown("---")
    left, right = st.columns([1.0, 1.8])
    with left:
        if st.button("← Back to Personal Finance Setup", key="step4_back_to_personal_finance", use_container_width=True):
            st.session_state[CURRENT_STEP] = 1
            st.session_state["current_step"] = 1
            st.rerun()
    with right:
        continue_disabled = not bool(st.session_state.get(ASSET_PANEL_READY, False))
        if st.button("Continue to Strategy Engine →", key="step4_continue", disabled=continue_disabled, use_container_width=True):
            st.session_state["step4_universe_payload"] = payload
            st.session_state[CURRENT_STEP] = 5
            st.session_state["current_step"] = 5
            st.rerun()
