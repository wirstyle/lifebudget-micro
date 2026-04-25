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
    if show_all_sizes:
        options = list(UNIVERSE_SIZES)
    else:
        options = list(recommended_universe_sizes_for_philosophy(philosophy))
    if current_size not in options:
        options.append(int(current_size))
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


def _render_data_panel_preview(panel_df: pd.DataFrame) -> None:
    st.dataframe(_panel_preview_for_display(panel_df), use_container_width=True, hide_index=True)


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


def render_step_4() -> None:
    section_header("Step 4 — Investment universe setup (foundation)")
    st.caption(
        "This step prepares the universe-selection layer for the investment module: preset universes, "
        "asset lists, data-source controls, and the validated return panel. "
        "This is now a data-preparation step, not a search or tuning step."
    )

    plan_snapshot = coerce_snapshot()
    investment_context = store_investment_context(plan_snapshot)

    st.markdown("### Monthly contribution bridge")
    b1, b2, b3 = st.columns(3)
    with b1:
        currency_metric("Monthly investment contribution", float(investment_context["monthly_contribution"]), decimals=0)
    with b2:
        currency_metric("Weekly equivalent", float(investment_context["weekly_equivalent"]), decimals=0)
    with b3:
        currency_metric("Available monthly surplus", float(investment_context["baseline_monthly"]), decimals=0)
    st.caption(
        f"Step 4 carries **£{float(investment_context['monthly_contribution']):,.0f}/month** into the investment setup "
        f"(about **£{float(investment_context['weekly_equivalent']):,.0f}/week** from the earlier budgeting steps)."
    )

    maybe_apply_initial_universe_size_default()
    current_philosophy = get_canonical_investment_philosophy()
    bundle = apply_investment_philosophy_bundle(current_philosophy)

    st.markdown("### Mother philosophy")
    selected_philosophy = st.selectbox(
        "Investment philosophy",
        options=INVESTMENT_PHILOSOPHY_OPTIONS,
        index=INVESTMENT_PHILOSOPHY_OPTIONS.index(current_philosophy),
        key="investment_philosophy_step4",
        help="Mother philosophy: the main posture that guides the preferred universe style, strategy setup preset space and projection tone.",
    )
    if selected_philosophy != current_philosophy:
        _sync_widget_defaults_from_philosophy(selected_philosophy)
        st.rerun()

    current_philosophy = get_canonical_investment_philosophy()
    bundle = apply_investment_philosophy_bundle(current_philosophy)
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(current_philosophy)
    st.caption(
        f"Current bundle: **{bundle['universe_strategy_preferences'][0]}** universe · "
        f"**{rec_template}** template · **{rec_style}** style · **{bundle['projection_profile']}** projection."
    )

    _render_philosophy_feel_block(current_philosophy)

    current_size = int(st.session_state.get("universe_size_input", st.session_state.get("universe_size", bundle["default_universe_size"])) or bundle["default_universe_size"])
    show_all_sizes = bool(st.session_state.get("show_all_sizes_input", False))
    show_all_strategies = bool(st.session_state.get("show_all_strategies_input", False))

    size_options = _build_visible_size_options(current_philosophy, show_all_sizes, current_size)
    if current_size not in size_options:
        current_size = int(bundle["default_universe_size"])
        st.session_state["universe_size_input"] = current_size

    st.markdown("### Primary universe")
    u1, u2 = st.columns(2)
    with u1:
        selected_size = st.selectbox("Primary universe size", size_options, index=size_options.index(int(st.session_state.get("universe_size_input", current_size))), key="universe_size_input")
        show_all_sizes = bool(
            st.checkbox(
                "Show all sizes",
                value=bool(st.session_state.get("show_all_sizes_input", False)),
                key="show_all_sizes_input",
                help="Show the full research universe-size menu instead of only the sizes preferred for this philosophy.",
            )
        )

    current_strategy = str(st.session_state.get("universe_strategy_input", st.session_state.get("universe_strategy", resolve_recommended_strategy_for_size(current_philosophy, selected_size))) or resolve_recommended_strategy_for_size(current_philosophy, selected_size))
    strategy_options = _build_visible_strategy_options(current_philosophy, int(selected_size), show_all_strategies, current_strategy)
    if current_strategy not in strategy_options:
        current_strategy = resolve_recommended_strategy_for_size(current_philosophy, int(selected_size))
        st.session_state["universe_strategy_input"] = current_strategy
    with u2:
        selected_strategy = st.selectbox(
            "Primary universe strategy",
            strategy_options,
            index=strategy_options.index(str(st.session_state.get("universe_strategy_input", current_strategy))),
            key="universe_strategy_input",
        )
        show_all_strategies = bool(
            st.checkbox(
                "Show all strategies",
                value=bool(st.session_state.get("show_all_strategies_input", False)),
                key="show_all_strategies_input",
                help="Show all strategies available for the selected universe size instead of only the philosophy-preferred subset.",
            )
        )

    with st.expander("Universe guidance (advanced)", expanded=False):
        _render_size_strategy_guidance(current_philosophy, int(selected_size), str(selected_strategy))
        st.divider()
        _render_strategy_setup_guidance(current_philosophy)

    current_combo_status = strategy_combo_status(
        current_philosophy,
        st.session_state.get("step5_template"),
        st.session_state.get("step5_style"),
    )
    if current_combo_status == "recommended":
        st.success("Strategy setup bundle is aligned with the current mother philosophy.")
    elif current_combo_status == "allowed":
        st.info("Strategy setup bundle is allowed for this philosophy, but it is not the primary preferred combo.")
    else:
        st.warning("Current strategy setup bundle is outside the allowed normal-mode space for this philosophy. It will be corrected in Step 5.")

    preset_assets = build_generated_universe(selected_size, selected_strategy)
    custom_allowed = int(selected_size) in {12, 25}
    default_custom_enabled = bool(st.session_state.get("universe_custom_enabled_input", st.session_state.get(UNIVERSE_CUSTOM_ENABLED, False))) if custom_allowed else False
    custom_enabled = st.checkbox("Custom asset list", value=default_custom_enabled, disabled=not custom_allowed, key="universe_custom_enabled_input")
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

    requested_size = int(selected_size)
    actual_size = len(selected_assets)
    if selected_assets:
        if actual_size == requested_size:
            st.caption(f"Primary universe ({actual_size} assets | {selected_strategy}): {build_universe_preview_text(selected_assets)}")
        else:
            st.caption(f"Primary universe ({actual_size}/{requested_size} assets | {selected_strategy}): {build_universe_preview_text(selected_assets)}")
    else:
        st.warning("No valid assets found yet for the primary universe.")

    mix_df, mix_summary = build_universe_mix(selected_assets)
    st.markdown("### Universe mix")
    rendered = show_table_if_not_empty(mix_df, empty_message="No universe mix available yet.", use_container_width=True, hide_index=True)
    if rendered:
        n_assets = int(mix_summary.get("n_assets", len(selected_assets)) or len(selected_assets))
        group_count = int(mix_summary.get("group_count", 0) or 0)
        if group_count <= 0 and isinstance(mix_df, pd.DataFrame) and "group" in mix_df.columns:
            group_count = int(mix_df["group"].nunique())
        classified_share = float(mix_summary.get("classified_share", 0.0) or 0.0)
        st.caption(f"{n_assets} assets across {group_count} groups. Classified share: {classified_share:.0%}.")
    detail_df = build_universe_mix_detail(selected_assets)
    with st.expander("Inspect asset composition", expanded=False):
        if isinstance(detail_df, pd.DataFrame) and not detail_df.empty:
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("No detailed universe taxonomy is available yet.")

    st.markdown("### Asset panel source")
    st.session_state[ASSET_SOURCE_MODE] = "yahoo"
    st.session_state[ASSET_UPLOADED_FILE] = None
    is_yahoo = True
    uploaded_file = None

    st.caption(
        "This prototype uses Yahoo Finance as the validated market-data source "
        "for the selected investment universe."
    )

    # Data-preparation support pool: used only to widen download/validation coverage.
    # It does not trigger optimisation, suggestions, or candidate reruns.
    candidate_assets = build_strategy_candidate_pool(selected_size, selected_strategy)
    if selected_assets:
        _render_combined_yahoo_notice(list(selected_assets), list(candidate_assets))
    else:
        st.warning("No valid tickers have been selected yet for Yahoo download.")

    freq_options = ["monthly", "weekly", "daily"]
    current_freq = str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly")
    if current_freq not in freq_options:
        current_freq = "monthly"

    # Keep the canonical key aligned with the new default.
    if str(st.session_state.get(ASSET_RETURN_FREQUENCY) or "") not in freq_options:
        st.session_state[ASSET_RETURN_FREQUENCY] = "monthly"

    # Widget state: default to monthly on first load, but do not overwrite
    # a valid explicit user choice on later reruns.
    widget_freq = str(st.session_state.get("asset_return_frequency_input") or "")
    if widget_freq not in freq_options:
        st.session_state["asset_return_frequency_input"] = current_freq if current_freq in freq_options else "monthly"

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Yahoo start date", value=_default_asset_start_date(), key="asset_start_date_input")
    with c2:
        end_date = st.date_input("Yahoo end date", value=_default_asset_end_date(), key="asset_end_date_input")

    freq = "monthly"
    st.session_state["asset_return_frequency_input"] = "monthly"
    auto_adjust = bool(st.session_state.get(ASSET_AUTO_ADJUST, True))

    with st.expander("Advanced data controls", expanded=False):
        freq = st.selectbox("Return frequency", options=freq_options, key="asset_return_frequency_input")
        auto_adjust = st.checkbox(
            "Use Yahoo auto-adjusted prices",
            value=bool(st.session_state.get(ASSET_AUTO_ADJUST, True)),
            key="asset_auto_adjust_input",
        )
        st.caption("Monthly is the recommended default for this prototype. Weekly/daily modes are available for diagnostics.")
    st.session_state[ASSET_START_DATE] = start_date
    st.session_state[ASSET_END_DATE] = end_date
    st.session_state[ASSET_RETURN_FREQUENCY] = freq
    st.session_state[ASSET_AUTO_ADJUST] = bool(auto_adjust)

    # Retired experimental diagnostic. Step 4 now keeps this disabled to avoid showing
    # a legacy start-date stress-test control that is no longer executed here.
    st.session_state[INVESTMENT_START_DATE_STABILITY_ENABLED] = False
    st.session_state[INVESTMENT_START_DATE_STABILITY_DATES] = []

    panel_df = None
    panel_source_label = ""
    panel_error = ""
    panel_timings = {}
    build_trigger = "auto"
    uploaded_signature_obj = uploaded_file if not is_yahoo else None
    current_panel_signature = build_step4_panel_input_signature(
        selected_assets=selected_assets,
        candidate_assets=candidate_assets,
        source_mode="yahoo" if is_yahoo else "upload",
        start_date=st.session_state.get(ASSET_START_DATE) if is_yahoo else None,
        end_date=st.session_state.get(ASSET_END_DATE) if is_yahoo else None,
        frequency=str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly"),
        auto_adjust=bool(st.session_state.get(ASSET_AUTO_ADJUST, True)),
        uploaded_file=uploaded_signature_obj,
    )
    previous_panel_signature = str(st.session_state.get(STEP4_PANEL_SIGNATURE_KEY, "") or "")
    force_refresh = False

    should_rebuild_panel = force_refresh or (previous_panel_signature != current_panel_signature) or (not bool(st.session_state.get(ASSET_PANEL_READY, False)))

    if should_rebuild_panel and selected_assets:
        try:
            panel_df, panel_source_label, union_caption = resolve_step4_asset_panel(
                selected_assets=selected_assets,
                candidate_assets=candidate_assets,
                source_mode="yahoo",
                start_date=st.session_state.get(ASSET_START_DATE),
                end_date=st.session_state.get(ASSET_END_DATE),
                frequency=str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly"),
                auto_adjust=bool(st.session_state.get(ASSET_AUTO_ADJUST, True)),
                uploaded_file=None,
            )
            panel_timings = dict(st.session_state.get(STEP4_PANEL_TIMINGS_KEY, {}) or {})
            if union_caption:
                st.session_state["_step4_latest_union_caption"] = str(union_caption)
            st.success(
                f"Market data loaded: {panel_df['asset'].nunique()} assets · "
                f"{panel_df['date'].min().date()} → {panel_df['date'].max().date()} · "
                f"{str(st.session_state.get(ASSET_RETURN_FREQUENCY, 'monthly'))} returns."
            )
            with st.expander("Data panel preview", expanded=False):
                _render_data_panel_preview(panel_df)
        except Exception as exc:
            panel_error = str(exc)
            st.error(f"Could not download data from Yahoo Finance: {panel_error}")
    else:
        ready_panel = st.session_state.get(ASSET_PANEL_DF)
        if isinstance(ready_panel, pd.DataFrame) and not ready_panel.empty:
            panel_df = ready_panel
            panel_source_label = str(st.session_state.get(ASSET_PANEL_SOURCE_LABEL, "") or "")
            panel_timings = dict(st.session_state.get(STEP4_PANEL_TIMINGS_KEY, {}) or {})
            panel_timings["reuse_existing_panel"] = True
            with st.expander("Data panel preview", expanded=False):
                _render_data_panel_preview(panel_df)

    update_asset_panel_state(
        panel_df,
        panel_source_label,
        panel_error,
        panel_signature=current_panel_signature,
        timings=panel_timings,
        build_trigger=build_trigger if should_rebuild_panel else "reuse_existing_panel",
    )

    if bool(st.session_state.get(ASSET_PANEL_READY, False)):
        st.success("Portfolio and market data ready.")
        st.caption(f"Panel ready · source: {st.session_state.get(ASSET_PANEL_SOURCE_LABEL, '')}")
        with st.expander("Diagnostics (advanced)", expanded=False):
            ready_panel_for_diag = st.session_state.get(ASSET_PANEL_DF)
            tickers_to_download = list(st.session_state.get("_step4_yahoo_tickers_to_download", []) or [])
            if tickers_to_download:
                _render_asset_diagnostics(requested_assets=tickers_to_download, ready_panel=ready_panel_for_diag)
            else:
                latest_union_caption = str(st.session_state.get("_step4_latest_union_caption", "") or "")
                if latest_union_caption:
                    st.caption(latest_union_caption)
                if isinstance(ready_panel_for_diag, pd.DataFrame) and not ready_panel_for_diag.empty:
                    _render_asset_diagnostics(requested_assets=[], ready_panel=ready_panel_for_diag)

            timing_summary = format_step4_timing_summary(st.session_state.get(STEP4_PANEL_TIMINGS_KEY, {}) or {})
            if timing_summary:
                st.caption("Step 4 timings · " + timing_summary)
            current_signature = st.session_state.get(STEP4_PANEL_SIGNATURE_KEY, "")
            if current_signature:
                st.caption("Panel input signature is stored for cache/rebuild checks.")

            ready_panel = st.session_state.get(ASSET_PANEL_DF)
            if isinstance(ready_panel, pd.DataFrame) and not ready_panel.empty:
                st.download_button(
                    "Download prepared market data",
                    data=panel_df_to_csv_bytes(ready_panel),
                    file_name=panel_download_filename(
                        st.session_state.get(ASSET_PANEL_SOURCE_LABEL, "panel"),
                        selected_strategy,
                        str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly"),
                    ),
                    mime="text/csv",
                    key="step4_download_panel_csv",
                )
    else:
        st.error("Step 5 is hard-gated until the asset panel is ready. Load a valid Yahoo Finance panel first.")
        if panel_error:
            st.caption(f"Latest panel error: {panel_error}")

    payload = build_step4_universe_payload_from_state()

    left, right = st.columns(2)
    with left:
        if st.button("Back to Step 3", key="step4_back"):
            st.session_state[CURRENT_STEP] = 3
            st.rerun()
    with right:
        continue_disabled = not bool(st.session_state.get(ASSET_PANEL_READY, False))
        if st.button("Continue to Investment Engine", key="step4_continue", disabled=continue_disabled):
            st.session_state["step4_universe_payload"] = payload
            st.session_state[CURRENT_STEP] = 5
            st.rerun()

    st.info("Finish this step by continuing to the Investment Engine workspace.")
