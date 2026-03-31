# src/explain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Technical explanation (Step 4 expander)
# ============================================================
@dataclass
class ExplanationInputs:
    income: float
    fixed_expenses: float
    variable_expenses: float
    weeks: int
    # IMPORTANT: in the current UI, these represent the *behavioural cut* (not the target savings)
    delta_a: float
    delta_b: float
    variability_pct: float  # e.g., 30 for ±30%
    seed: int
    iters: int


def _band_width(lower: float, upper: float) -> float:
    return float(upper - lower)


def _safe_pct_change(new: float, base: float) -> Optional[float]:
    if base == 0:
        return None
    return float((new - base) / base * 100.0)


def _fmt_gbp0(x: float) -> str:
    return f"£{round(float(x)):,.0f}"


def _fmt_pct(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{p:.1f}%"


def format_money_range(low: float, high: float) -> str:
    return f"{_fmt_gbp0(low)} to {_fmt_gbp0(high)}"


@dataclass
class SinglePlanExplanationInputs:
    weeks: int
    variability_pct: float  # e.g., 30 for ±30%
    seed: int
    iters: int


def build_single_plan_explanation(
    *,
    baseline_final: float,
    conservative_final: float,
    expected_final: float,
    optimistic_final: float,
    inputs: SinglePlanExplanationInputs,
) -> str:
    """
    Human-readable explanation (technical) for the single-plan Step 3 flow.

    This version is aligned with the current UX:
    - one plan
    - three outcome views from the Monte Carlo distribution
      (conservative / expected / optimistic)
    """
    baseline_final = float(round(float(baseline_final)))
    conservative_final = float(round(float(conservative_final)))
    expected_final = float(round(float(expected_final)))
    optimistic_final = float(round(float(optimistic_final)))

    conservative_diff = conservative_final - baseline_final
    expected_diff = expected_final - baseline_final
    optimistic_diff = optimistic_final - baseline_final

    spread_width = optimistic_final - conservative_final

    conservative_pct = _safe_pct_change(conservative_final, baseline_final)
    expected_pct = _safe_pct_change(expected_final, baseline_final)
    optimistic_pct = _safe_pct_change(optimistic_final, baseline_final)

    def _fmt_money_delta(x: float) -> str:
        sign = "+" if x >= 0 else "−"
        return f"{sign}£{abs(round(float(x))):,.0f}"

    def _fmt_pct_delta(p: Optional[float]) -> str:
        if p is None:
            return ""
        return f", {p:+.1f}%"

    # ---- Dynamic interpretation layer (new) ----
    baseline_abs = abs(baseline_final) if abs(baseline_final) > 1e-9 else 1.0
    spread_ratio = float(spread_width / baseline_abs)

    if expected_final >= baseline_final and conservative_final >= baseline_final:
        if spread_ratio <= 0.03:
            interpretation = (
                "Overall, **your plan looks robust**: even the conservative outcome stays above baseline, "
                "and the uncertainty range is relatively tight."
            )
        else:
            interpretation = (
                "Overall, **your plan looks positive but somewhat fragile**: it improves on baseline even in the conservative case, "
                "but the uncertainty range is still meaningful."
            )
    elif expected_final >= baseline_final and conservative_final < baseline_final:
        if spread_ratio <= 0.03:
            interpretation = (
                "Overall, **your plan looks promising but downside-sensitive**: the expected path improves on baseline, "
                "but a weaker run could still finish below it."
            )
        else:
            interpretation = (
                "Overall, **your plan looks fragile**: the expected path is better than baseline, "
                "but the conservative outcome drops below it and the uncertainty range is fairly wide."
            )
    elif expected_final < baseline_final and optimistic_final > baseline_final:
        interpretation = (
            "Overall, **your plan is under pressure**: the central expectation is below baseline, "
            "and it would take a stronger-than-usual outcome to beat it."
        )
    else:
        interpretation = (
            "Overall, **your plan does not currently look robust versus baseline**: "
            "both the expected and conservative outcomes remain below it."
        )

    lines: List[str] = []
    lines.append(f"Over {inputs.weeks} weeks, the baseline ends at **{_fmt_gbp0(baseline_final)}**.")
    lines.append("Your plan is evaluated using a Monte Carlo simulation with variable spending uncertainty.")
    lines.append(
        f"- **Conservative (10th percentile)**: **{_fmt_gbp0(conservative_final)}** "
        f"({_fmt_money_delta(conservative_diff)} vs baseline{_fmt_pct_delta(conservative_pct)})."
    )
    lines.append(
        f"- **Expected (median / 50th percentile)**: **{_fmt_gbp0(expected_final)}** "
        f"({_fmt_money_delta(expected_diff)} vs baseline{_fmt_pct_delta(expected_pct)})."
    )
    lines.append(
        f"- **Optimistic (90th percentile)**: **{_fmt_gbp0(optimistic_final)}** "
        f"({_fmt_money_delta(optimistic_diff)} vs baseline{_fmt_pct_delta(optimistic_pct)})."
    )

    lines.append(interpretation)

    lines.append(
        f"Uncertainty comes from **variable spending fluctuations** (±{float(inputs.variability_pct):.0f}%). "
        f"The full 10–90% range at the end of the horizon is about **{_fmt_gbp0(spread_width)}**."
    )
    if float(inputs.variability_pct) >= 25:
        lines.append(
            "Because variability is relatively high, outcomes are sensitive to real-life spending behaviour—"
            "more consistent weeks can narrow the range."
        )
    else:
        lines.append(
            "Because variability is relatively low, the forecast band stays tighter—"
            "results are more predictable given the same inputs."
        )
    lines.append(
        f"Reproducibility note: results are generated with seed **{int(inputs.seed)}** "
        f"over **{int(inputs.iters)}** Monte Carlo iterations."
    )

    return "\n\n".join(lines)


def build_explanation(
    base_final: float,
    a_final_mean: float,
    a_final_low: float,
    a_final_high: float,
    b_final_mean: float,
    b_final_low: float,
    b_final_high: float,
    inputs: ExplanationInputs,
) -> str:
    """
    Human-readable explanation (technical).
    Note: we keep language like "saves £X/week", but internally X here is the
    *discretionary cut* (behaviour lever) used by the simulator.
    """
    diff_a = a_final_mean - base_final
    diff_b = b_final_mean - base_final

    pct_a = _safe_pct_change(a_final_mean, base_final)
    pct_b = _safe_pct_change(b_final_mean, base_final)

    winner = "Scenario A" if a_final_mean >= b_final_mean else "Scenario B"
    win_diff = abs(a_final_mean - b_final_mean)

    width_a = _band_width(a_final_low, a_final_high)
    width_b = _band_width(b_final_low, b_final_high)

    lines: List[str] = []

    def _fmt_money(x: float) -> str:
        sign = "+" if x >= 0 else "−"
        return f"{sign}£{abs(x):,.2f}"

    lines.append(f"Over {inputs.weeks} weeks, the baseline ends at **£{base_final:,.2f}**.")
    lines.append(
        f"Scenario assumptions: **A changes spending by £{inputs.delta_a:,.0f}/week**, "
        f"**B changes spending by £{inputs.delta_b:,.0f}/week** "
        f"(variable spending uncertainty set to ±{inputs.variability_pct:.0f}%)."
    )

    if pct_a is None:
        lines.append(f"**Scenario A** ends at **£{a_final_mean:,.2f}** ({_fmt_money(diff_a)} vs baseline).")
    else:
        lines.append(
            f"**Scenario A** ends at **£{a_final_mean:,.2f}** "
            f"({_fmt_money(diff_a)} vs baseline, {pct_a:+.1f}%)."
        )

    if pct_b is None:
        lines.append(f"**Scenario B** ends at **£{b_final_mean:,.2f}** ({_fmt_money(diff_b)} vs baseline).")
    else:
        lines.append(
            f"**Scenario B** ends at **£{b_final_mean:,.2f}** "
            f"({_fmt_money(diff_b)} vs baseline, {pct_b:+.1f}%)."
        )

    lines.append(
        f"Based on mean outcomes, **{winner}** produces the higher projected final balance "
        f"(gap ≈ **£{win_diff:,.2f}**)."
    )

    if inputs.delta_a == inputs.delta_b:
        if inputs.delta_a > 0:
            lines.append(
                f"Both scenarios apply the same behavioural change (**£{inputs.delta_a:,.0f}/week**), "
                "so remaining differences are driven mainly by stochastic variability."
            )
        else:
            lines.append(
                "No additional behavioural change is applied in either scenario, "
                "so differences are driven mainly by spending variability."
            )
    else:
        stronger = "Scenario A" if inputs.delta_a > inputs.delta_b else "Scenario B"
        stronger_delta = max(inputs.delta_a, inputs.delta_b)
        lines.append(
            f"The main driver is the **behavioural lever**: **{stronger}** applies the larger adjustment "
            f"(**£{stronger_delta:,.0f}/week**), which compounds over time."
        )

    lines.append(
        f"Uncertainty comes from **variable spending fluctuations** (±{inputs.variability_pct:.0f}%). "
        f"At week {inputs.weeks}, the uncertainty range width is about **£{width_a:,.2f}** for Scenario A "
        f"and **£{width_b:,.2f}** for Scenario B (10–90% band)."
    )

    wider = "Scenario A" if width_a >= width_b else "Scenario B"
    lines.append(
        f"The uncertainty band is slightly wider for **{wider}** at the final week, "
        "reflecting sensitivity to variable spending noise."
    )

    if inputs.variability_pct >= 25:
        lines.append(
            "Because variability is relatively high, outcomes are sensitive to spending behaviour—"
            "more consistent spending can narrow the uncertainty band."
        )
    else:
        lines.append(
            "Because variability is relatively low, the forecast band stays tighter—"
            "results are more predictable given the same inputs."
        )

    lines.append(
        f"Reproducibility note: results are generated with seed **{inputs.seed}** "
        f"over **{inputs.iters}** Monte Carlo iterations."
    )

    return "\n\n".join(lines)


# ============================================================
# Human-first reflection (Step 4)
# ============================================================
@dataclass
class ReflectionMetrics:
    weeks: int
    base_final: float

    a_mean: float
    a_low: float
    a_high: float

    b_mean: float
    b_low: float
    b_high: float

    # Target-first semantics (Option B)
    target_a_weekly: float
    target_b_weekly: float
    cut_a_weekly: float
    cut_b_weekly: float
    margin_a_weekly: float
    margin_b_weekly: float

    # Back-compat (treat as REAL cut)
    delta_a_weekly: float
    delta_b_weekly: float

    a_gain: float
    b_gain: float
    gap_ab: float

    a_pct: Optional[float]
    b_pct: Optional[float]

    winner: str   # "Scenario A" | "Scenario B" | "Tie"
    safer: str    # "Scenario A" | "Scenario B" | "Tie"


def compute_reflection_metrics(
    *,
    weeks: int,
    base_final: float,
    a_mean: float,
    a_low: float,
    a_high: float,
    b_mean: float,
    b_low: float,
    b_high: float,
    # New params (preferred)
    target_a_weekly: Optional[float] = None,
    target_b_weekly: Optional[float] = None,
    cut_a_weekly: Optional[float] = None,
    cut_b_weekly: Optional[float] = None,
    margin_a_weekly: Optional[float] = None,
    margin_b_weekly: Optional[float] = None,
    # Old params (fallback)
    delta_a_weekly: float = 0.0,
    delta_b_weekly: float = 0.0,
) -> ReflectionMetrics:
    if target_a_weekly is None:
        target_a_weekly = float(delta_a_weekly)
    if target_b_weekly is None:
        target_b_weekly = float(delta_b_weekly)

    if cut_a_weekly is None:
        cut_a_weekly = float(delta_a_weekly)
    if cut_b_weekly is None:
        cut_b_weekly = float(delta_b_weekly)

    if margin_a_weekly is None:
        margin_a_weekly = max(float(target_a_weekly) - float(cut_a_weekly), 0.0)
    if margin_b_weekly is None:
        margin_b_weekly = max(float(target_b_weekly) - float(cut_b_weekly), 0.0)

    a_gain = float(a_mean - base_final)
    b_gain = float(b_mean - base_final)
    gap_ab = float(b_mean - a_mean)

    a_pct = _safe_pct_change(a_mean, base_final)
    b_pct = _safe_pct_change(b_mean, base_final)

    if a_mean > b_mean:
        winner = "Scenario A"
    elif b_mean > a_mean:
        winner = "Scenario B"
    else:
        winner = "Tie"

    if a_low > b_low:
        safer = "Scenario A"
    elif b_low > a_low:
        safer = "Scenario B"
    else:
        safer = "Tie"

    return ReflectionMetrics(
        weeks=int(weeks),
        base_final=float(base_final),

        a_mean=float(a_mean),
        a_low=float(a_low),
        a_high=float(a_high),

        b_mean=float(b_mean),
        b_low=float(b_low),
        b_high=float(b_high),

        target_a_weekly=float(target_a_weekly),
        target_b_weekly=float(target_b_weekly),
        cut_a_weekly=float(cut_a_weekly),
        cut_b_weekly=float(cut_b_weekly),
        margin_a_weekly=float(margin_a_weekly),
        margin_b_weekly=float(margin_b_weekly),

        delta_a_weekly=float(cut_a_weekly),
        delta_b_weekly=float(cut_b_weekly),

        a_gain=a_gain,
        b_gain=b_gain,
        gap_ab=gap_ab,
        a_pct=a_pct,
        b_pct=b_pct,
        winner=winner,
        safer=safer,
    )


# ============================================================
# Step 3 → Step 4 transition helper
# ============================================================

def build_step4_transition_text(
    *,
    weekly_margin: float,
    monthly_equivalent: Optional[float] = None,
    investment_enabled: bool = True,
) -> str:
    """
    Small narrative bridge between the weekly budgeting layer and the optional
    long-horizon investment layer.

    Parameters
    ----------
    weekly_margin:
        Current weekly margin after essentials / budgeting logic.
    monthly_equivalent:
        Optional monthly equivalent. If None, computed from weekly_margin.
    investment_enabled:
        Whether the Step 4 investment module is currently available in the UI.
    """
    wm = float(weekly_margin)
    mm = float(monthly_equivalent) if monthly_equivalent is not None else float(weekly_to_monthly_for_text(wm))

    if not investment_enabled:
        return (
            f"Your current weekly margin is **{_fmt_gbp0(wm)}/week** "
            f"(about **{_fmt_gbp0(mm)}/month**). "
            "If you stabilise this consistently, it can later become the basis for longer-term savings or investing decisions."
        )

    if wm <= 0:
        return (
            f"Right now your margin is **{_fmt_gbp0(wm)}/week** "
            f"(about **{_fmt_gbp0(mm)}/month**). "
            "That means the immediate priority is to stabilise the weekly budget first. "
            "Once the margin turns positive and sustainable, you can use Step 4 to explore long-term growth scenarios."
        )

    if wm < 25:
        return (
            f"You currently have a positive margin of about **{_fmt_gbp0(wm)}/week** "
            f"(roughly **{_fmt_gbp0(mm)}/month**). "
            "That creates a small but real base for future investing. "
            "Step 4 can help you explore what even modest monthly contributions might look like over several years."
        )

    if wm < 75:
        return (
            f"You currently have a weekly margin of around **{_fmt_gbp0(wm)}/week**, "
            f"which is about **{_fmt_gbp0(mm)}/month**. "
            "That means your short-term budget is not only functioning, but also generating investable capacity. "
            "In Step 4, you can test what could happen if part of that amount were invested monthly over time."
        )

    return (
        f"Your weekly margin is about **{_fmt_gbp0(wm)}/week** "
        f"(around **{_fmt_gbp0(mm)}/month**), which gives you meaningful room to think beyond short-term budgeting. "
        "Step 4 lets you explore how part of that surplus could compound over the long run under different risk profiles."
    )


def weekly_to_monthly_for_text(amount_weekly: float) -> float:
    return float(amount_weekly) * (52.0 / 12.0)


# ============================================================
# 🧠 Intelligent structural deficit tips
# ============================================================
def build_structural_deficit_tips(
    *,
    breakdown: Dict[str, Any],
    deficit_w: float,
    top_n: int = 3,
) -> str:
    """
    Builds data-driven tips using what the user actually entered.

    Expected breakdown keys (best-effort, optional):
      - income_w: float
      - fixed_total_w: float  (fixed + variable essentials)
      - discretionary_w: float
      - margin_w: float (can be negative)
      - fixed_items_rows: list[dict] with keys: name, weekly
      - variable_items_rows_weekly: list[dict] with keys: name, weekly
      - variable_parts: dict with keys like utilities_weekly, commute_weekly, groceries_weekly, household_weekly (optional)
    """
    income = float(breakdown.get("income_w", 0.0) or 0.0)
    essentials = float(breakdown.get("fixed_total_w", 0.0) or 0.0)
    disc = float(breakdown.get("discretionary_w", 0.0) or 0.0)

    fixed_items = breakdown.get("fixed_items_rows") or []
    variable_items = breakdown.get("variable_items_rows_weekly") or []

    CANON = {
        "Commute": "Commuting",
        "Commute Weekly": "Commuting",
        "Utilities Base": "Utilities",
        "Utilities Base Weekly": "Utilities",
        "Utilities Weekly": "Utilities",
        "Groceries Weekly": "Groceries",
        "Household Weekly": "Household basics",
        "Household Basics Weekly": "Household basics",
    }

    def norm_name(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return "Unnamed"
        return CANON.get(s, s)

    def pct(x: float) -> Optional[float]:
        if income <= 0:
            return None
        return float(x / income * 100.0)

    ranked: List[Dict[str, Any]] = []

    def push_items(rows: list, src: str):
        for r in rows:
            try:
                name = norm_name(str(r.get("name", "")).strip())
                weekly = float(r.get("weekly", 0.0) or 0.0)
                if weekly > 0:
                    ranked.append({"name": name, "weekly": weekly, "src": src})
            except Exception:
                continue

    push_items(fixed_items, "fixed")
    push_items(variable_items, "variable")

    ranked.sort(key=lambda d: d["weekly"], reverse=True)
    top_items = ranked[: max(0, int(top_n))]

    var_parts = breakdown.get("variable_parts") or {}

    def _vp(key: str) -> float:
        try:
            return float(var_parts.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    utilities_w = _vp("utilities_weekly")
    commute_w = _vp("commute_weekly")
    groceries_w = _vp("groceries_weekly")
    household_w = _vp("household_weekly")

    lines: List[str] = []
    lines.append("## Tips to fix a structural deficit (based on your numbers)")
    lines.append("")
    lines.append(
        f"- Your weekly shortfall is about **{_fmt_gbp0(deficit_w)}/week** "
        "(income does not cover essentials)."
    )

    if disc > 0:
        cover_pct_disc = min(disc / deficit_w * 100.0, 100.0) if deficit_w > 0 else 0.0
        lines.append(
            f"- Discretionary is **{_fmt_gbp0(disc)}/week**. "
            f"Even cutting it to £0 would cover only **{cover_pct_disc:.0f}%** of the deficit."
        )
    else:
        lines.append("- Discretionary is already **£0/week**, so lifestyle cuts alone cannot fix this.")

    if income > 0:
        ess_pct = pct(essentials) or 0.0
        lines.append(
            f"- Essentials are about **{ess_pct:.0f}% of income** (**{_fmt_gbp0(essentials)}/w**). "
            "As a reference point (not a rule), many budgets aim for **50–65%**."
        )

    if top_items:
        lines.append("")
        lines.append("### Your biggest weekly items (ranked from your breakdown)")
        for it in top_items:
            share = pct(float(it["weekly"]))
            src_tag = "Fixed" if it.get("src") == "fixed" else "Variable"
            if share is None:
                lines.append(f"- {it['name']}: **{_fmt_gbp0(it['weekly'])}/w** ({src_tag})")
            else:
                lines.append(f"- {it['name']}: **{_fmt_gbp0(it['weekly'])}/w** (~{share:.0f}% of income, {src_tag})")

        biggest = float(top_items[0]["weekly"])
        if deficit_w > 0:
            multiple = biggest / deficit_w
            cover_pct = min(multiple * 100.0, 100.0)

            lines.append("")
            lines.append(
                f"**High leverage:** your largest item is **{_fmt_gbp0(biggest)}/w**, "
                f"which equals about **{cover_pct:.0f}%** of the deficit."
            )

            if multiple >= 1.0:
                lines.append(
                    f"- Reducing this single item by **{_fmt_gbp0(deficit_w)}/w** "
                    "would eliminate the shortfall entirely."
                )
            else:
                lines.append(
                    f"- Even eliminating it completely would cover only "
                    f"about **{cover_pct:.0f}%** of the deficit."
                )

            if len(top_items) >= 2:
                second = float(top_items[1]["weekly"])
                combined = biggest + second
                combined_pct = min(combined / deficit_w * 100.0, 100.0)
                lines.append(
                    f"- Your top two items combined represent about **{combined_pct:.0f}%** of the deficit."
                )

            lines.append("")
            lines.append("### What “break-even” means in your terms (equivalences)")
            eq_bits: List[str] = []
            for it in top_items[:3]:
                w = float(it["weekly"])
                if w > 0:
                    frac = deficit_w / w
                    eq_bits.append(f"≈ **{frac:.2f}×** your **{it['name']}** ({_fmt_gbp0(w)}/w)")
            if eq_bits:
                lines.append(f"- To reach break-even you need about **{_fmt_gbp0(deficit_w)}/w** improvement, which is:")
                for b in eq_bits:
                    lines.append(f"  - {b}")

        lines.append("")
        lines.append("### Quick quantified targets (examples)")
        lines.append(
            f"- Break-even requires roughly **{_fmt_gbp0(deficit_w)}/w** improvement.\n"
            f"- Example split: reduce the top item by **{_fmt_gbp0(deficit_w/2)}/w** "
            f"and find the other **{_fmt_gbp0(deficit_w/2)}/w** across the next 1–2 items."
        )
    else:
        lines.append("")
        lines.append(
            "### Make tips more specific\n"
            "Open **“Build my fixed expenses”** in Step 1 and enter your major items "
            "(rent, council tax, transport, etc.). Then come back to Step 2 — the tips will list your top drivers explicitly."
        )

    if any(x > 0 for x in [utilities_w, commute_w, groceries_w, household_w]):
        lines.append("")
        lines.append("### Variable essentials (from your inputs)")
        if utilities_w > 0:
            lines.append(f"- Utilities: **{_fmt_gbp0(utilities_w)}/w**")
        if commute_w > 0:
            lines.append(f"- Commuting: **{_fmt_gbp0(commute_w)}/w**")
        if groceries_w > 0:
            lines.append(f"- Groceries: **{_fmt_gbp0(groceries_w)}/w**")
        if household_w > 0:
            lines.append(f"- Household basics: **{_fmt_gbp0(household_w)}/w**")
        lines.append(
            "If one of these stands out, it’s a good candidate for a quick review "
            "(plans, shopping pattern, travel mode)."
        )

    lines.append("")
    lines.append("### Minimum improvement needed to break even")
    lines.append(
        f"- You need to improve your weekly position by about **{_fmt_gbp0(deficit_w)}/w** "
        "(via higher income, lower essentials, or a mix)."
    )

    lines.append("")
    lines.append("### Inside the app (what to do next)")
    lines.append(
        "- Step 1 → use the fixed/variable breakdowns so the app can identify your top items.\n"
        "- Try a realistic reduction on the top 1–3 weekly items, then re-check the Margin.\n"
        "- Once Margin ≥ £0/w, Step 2 scenarios become actionable."
    )
    lines.append(
        "_Note: tips are not personalised financial advice (the app can’t see contracts, eligibility, or local options)._"
    )

    return "\n".join(lines).strip()


def build_human_reflection_text(
    metrics: ReflectionMetrics,
    *,
    shock_events: Optional[list] = None,
    breakdown: Optional[Dict[str, Any]] = None,
    include_step4_transition: bool = False,
    investment_enabled: bool = True,
) -> Dict[str, str]:
    """
    Returns text blocks for Streamlit rendering (keeps app.py thin).

    Keys:
      - what_you_get
      - pick_primary
      - pick_secondary
      - shock_note
      - variability
      - structural_deficit (optional)
      - step4_transition (optional)
    """
    shock_note = ""

    def _coverage_line(target: float, margin: float, cut: float) -> str:
        if float(cut) <= 0:
            return f"  - Covered fully by existing margin (**{_fmt_gbp0(margin)}/w**) — **no spending cuts**."
        return (
            f"  - Covered by margin: **{_fmt_gbp0(margin)}/w**\n"
            f"  - Requires cutting discretionary by **{_fmt_gbp0(cut)}/w**."
        )

    what_you_get_lines = [
        f"- If you **change nothing**, you end around **{_fmt_gbp0(metrics.base_final)}** after **{metrics.weeks} weeks**.",
        f"- **Scenario A**: target **save {_fmt_gbp0(metrics.target_a_weekly)}/week** → end around **{_fmt_gbp0(metrics.a_mean)}** "
        f"(≈ **{_fmt_gbp0(metrics.a_gain)} more** than baseline, **{_fmt_pct(metrics.a_pct)}**).",
        _coverage_line(metrics.target_a_weekly, metrics.margin_a_weekly, metrics.cut_a_weekly),
        f"- **Scenario B**: target **save {_fmt_gbp0(metrics.target_b_weekly)}/week** → end around **{_fmt_gbp0(metrics.b_mean)}** "
        f"(≈ **{_fmt_gbp0(metrics.b_gain)} more** than baseline, **{_fmt_pct(metrics.b_pct)}**).",
        _coverage_line(metrics.target_b_weekly, metrics.margin_b_weekly, metrics.cut_b_weekly),
    ]
    what_you_get = "\n".join(what_you_get_lines)

    if metrics.winner == "Tie":
        pick_primary = (
            "Both scenarios are very close on the expected outcome. "
            "Pick the target that feels easier to stick to every week."
        )
        pick_secondary = ""
    else:
        pick_primary = (
            f"On average, **{metrics.winner}** leaves you with more money. "
            f"The gap vs the other option is about **{_fmt_gbp0(abs(metrics.gap_ab))}**."
        )
        pick_secondary = ""
        if metrics.safer != "Tie" and metrics.safer != metrics.winner:
            pick_secondary = (
                f"However, the **safer worst-case** (lower-bound) looks like **{metrics.safer}**. "
                "So one option wins on average, but the other is slightly more resilient in a bad draw."
            )

    if shock_events:
        valid = [r for r in shock_events if float(r.get("amount", 0.0) or 0.0) > 0]
        if valid:
            total = sum(float(r.get("amount", 0.0) or 0.0) for r in valid)
            count = len(valid)
            worst = max(valid, key=lambda r: float(r.get("amount", 0.0) or 0.0))
            shock_note = (
                f"One-off events included: **{count} event(s)** "
                f"(total impact **{_fmt_gbp0(total)}**).\n\n"
                f"Largest event: **{_fmt_gbp0(float(worst.get('amount', 0.0) or 0.0))}** "
                f"in week **{int(worst.get('week', 0) or 0)}**."
            )

    variability = "\n".join(
        [
            "### Results can vary (because real weeks aren’t identical)",
            f"- Scenario A likely ends somewhere around **{format_money_range(metrics.a_low, metrics.a_high)}**.",
            f"- Scenario B likely ends somewhere around **{format_money_range(metrics.b_low, metrics.b_high)}**.",
        ]
    )

    structural_msg = ""
    if breakdown is not None:
        margin_w = float(breakdown.get("margin_w", 0.0) or 0.0)
        disc_w = float(breakdown.get("discretionary_w", 0.0) or 0.0)
        structural_deficit = (margin_w + disc_w) < 0
        if structural_deficit:
            structural_msg = build_structural_deficit_tips(
                breakdown=breakdown,
                deficit_w=abs(margin_w),
            )

    step4_transition = ""
    if include_step4_transition:
        best_weekly_capacity = max(
            float(metrics.target_a_weekly),
            float(metrics.target_b_weekly),
            0.0,
        )
        step4_transition = build_step4_transition_text(
            weekly_margin=best_weekly_capacity,
            monthly_equivalent=weekly_to_monthly_for_text(best_weekly_capacity),
            investment_enabled=investment_enabled,
        )

    return {
        "what_you_get": what_you_get,
        "pick_primary": pick_primary,
        "pick_secondary": pick_secondary,
        "shock_note": shock_note,
        "variability": variability,
        "structural_deficit": structural_msg,
        "step4_transition": step4_transition,
    }

# ============================================================
# Step 7 — Investment strategy explanation
# ============================================================
@dataclass
class InvestmentExplanationInputs:
    performance_summary: Dict[str, Any]
    risk_summary: Dict[str, Any]
    investment_context: Dict[str, Any]
    projection_summary: Optional[Dict[str, Any]] = None


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _fmt_pct1(value: Optional[float], *, already_fraction: bool = True) -> str:
    if value is None:
        return "—"
    scaled = float(value) * 100.0 if already_fraction else float(value)
    return f"{scaled:.1f}%"


def _fmt_money0(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"£{round(float(value)):,.0f}"


def _pick_first_numeric(mapping: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in mapping:
            value = _safe_float(mapping.get(key))
            if value is not None:
                return value
    return None


def _infer_strategy_profile(cagr: Optional[float], vol: Optional[float], max_dd: Optional[float], sharpe: Optional[float]) -> Tuple[str, str]:
    score = 0.0
    if vol is not None:
        if vol <= 0.10:
            score -= 2.0
        elif vol <= 0.16:
            score -= 0.5
        elif vol >= 0.24:
            score += 2.0
        elif vol >= 0.18:
            score += 1.0
    if max_dd is not None:
        dd = abs(float(max_dd))
        if dd <= 0.12:
            score -= 1.5
        elif dd <= 0.20:
            score -= 0.5
        elif dd >= 0.35:
            score += 2.0
        elif dd >= 0.25:
            score += 1.0
    if cagr is not None:
        if cagr >= 0.10:
            score += 1.0
        elif cagr <= 0.04:
            score -= 0.5
    if sharpe is not None:
        if sharpe >= 0.90:
            score -= 0.25
        elif sharpe <= 0.30:
            score += 0.25

    if score <= -1.5:
        return (
            "Conservative",
            "This looks more defensive than return-seeking: the profile suggests a strategy that is trying to limit damage first and grow second.",
        )
    if score >= 1.5:
        return (
            "Growth-like",
            "This looks more return-seeking than defensive: the profile suggests a strategy willing to accept bigger swings in pursuit of higher long-run upside.",
        )
    return (
        "Balanced",
        "This sits in the middle: it is not especially defensive, but it is not acting like a fully aggressive growth strategy either.",
    )


def _build_cagr_line(cagr: Optional[float], projection_summary: Dict[str, Any]) -> str:
    if cagr is None:
        return "- **CAGR**: the app could not read a stable annual growth estimate from this run."
    expected_terminal = _pick_first_numeric(projection_summary, "expected_terminal", "median_terminal")
    current_savings = _pick_first_numeric(projection_summary, "starting_value", "starting_wealth", "current_savings")
    monthly_contribution = _pick_first_numeric(projection_summary, "monthly_contribution")
    years = _pick_first_numeric(projection_summary, "horizon_years")
    real_example = ""
    if expected_terminal is not None and years is not None and years > 0:
        real_example = f" In your current projection, that translates into an expected pot around **{_fmt_money0(expected_terminal)}** over about **{int(round(years))} years**."
    elif cagr is not None:
        illustrative = 10000.0 * ((1.0 + float(cagr)) ** 10)
        real_example = f" A simple illustration is **£10,000** growing to roughly **{_fmt_money0(illustrative)}** over 10 years if that average rate held."
    contrib_note = ""
    if monthly_contribution is not None and monthly_contribution > 0:
        contrib_note = f" This is being supported by ongoing contributions of about **{_fmt_money0(monthly_contribution)}/month**."
    elif current_savings is not None:
        contrib_note = f" Think of it as the growth pace applied to a starting pot of about **{_fmt_money0(current_savings)}**."
    return f"- **CAGR ({_fmt_pct1(cagr)})**: this is the strategy's average long-run growth speed, not a promise of what happens every year.{real_example}{contrib_note}"


def _build_vol_line(vol: Optional[float]) -> str:
    if vol is None:
        return "- **Volatility**: the app could not read a stable volatility estimate from this run."
    if vol <= 0.10:
        tone = "day-to-day and year-to-year movement is relatively contained for an investment strategy"
    elif vol <= 0.18:
        tone = "you should expect visible swings, but not the kind of turbulence usually associated with very aggressive portfolios"
    else:
        tone = "the ride can be rough, with meaningful ups and downs even when the long-run story still looks okay"
    yearly_move = 10000.0 * float(vol)
    return f"- **Volatility ({_fmt_pct1(vol)})**: this is the amount of noise around the average path. In practice, **{tone}**. On a **£10,000** pot, that is roughly the difference between a fairly calm year and a year that swings by around **{_fmt_money0(yearly_move)}** either way."


def _build_drawdown_line(max_dd: Optional[float]) -> str:
    if max_dd is None:
        return "- **Max drawdown**: the app could not read a stable drawdown estimate from this run."
    dd = abs(float(max_dd))
    start = 10000.0
    trough = start * (1.0 - dd)
    if dd <= 0.12:
        tone = "That is uncomfortable, but still within the zone many defensive or balanced investors can tolerate."
    elif dd <= 0.25:
        tone = "That is a real setback: many users say they are fine with risk until they actually live through a drop like this."
    else:
        tone = "That is severe. A strategy with this kind of drawdown can be hard to stick with emotionally, even if the maths later recovers."
    return f"- **Max drawdown ({_fmt_pct1(dd)})**: this is the worst peak-to-trough fall seen in the tested path. In money terms, **£10,000** could temporarily fall to about **{_fmt_money0(trough)}**. {tone}"


def _build_sharpe_line(sharpe: Optional[float]) -> str:
    if sharpe is None:
        return "- **Sharpe**: the app could not read a stable Sharpe estimate from this run."
    if sharpe >= 1.0:
        tone = "That usually means the strategy has been paid reasonably well for the risk it took."
    elif sharpe >= 0.5:
        tone = "That is a workable middle ground: there may be value here, but the reward per unit of risk is not exceptional."
    else:
        tone = "That suggests the strategy is taking risk without being paid especially well for it."
    return f"- **Sharpe ({sharpe:.2f})**: this is a rough 'efficiency' score for risk versus return. Higher is better because it means the ups have been more worth the stress. {tone}"


def _build_educational_comparison(profile: str, cagr: Optional[float], vol: Optional[float], max_dd: Optional[float]) -> str:
    current = []
    if cagr is not None:
        current.append(f"growth around **{_fmt_pct1(cagr)}**")
    if vol is not None:
        current.append(f"volatility around **{_fmt_pct1(vol)}**")
    if max_dd is not None:
        current.append(f"worst fall around **{_fmt_pct1(abs(max_dd))}**")
    current_text = ", ".join(current) if current else "mixed risk/return characteristics"
    return (
        "### Educational comparison points\n\n"
        "- **Cash-like**: very low growth, very low movement, but inflation can quietly erode purchasing power.\n"
        "- **Balanced**: moderate growth with noticeable but usually tolerable drawdowns.\n"
        "- **Growth**: stronger long-run upside, but deeper falls and a bumpier ride.\n"
        "- **Aggressive**: highest upside potential, but also the easiest profile to abandon after a bad year.\n\n"
        f"Your current run looks closest to **{profile}**, with {current_text}. So the key question is not just 'can it earn more?', but 'could a real person live through the bad stretches without bailing out at the worst moment?'"
    )


def _build_projection_section(projection_summary: Dict[str, Any]) -> str:
    if not projection_summary:
        return ""
    expected_terminal = _pick_first_numeric(projection_summary, "expected_terminal")
    median_terminal = _pick_first_numeric(projection_summary, "median_terminal")
    p10_terminal = _pick_first_numeric(projection_summary, "p10_terminal")
    p90_terminal = _pick_first_numeric(projection_summary, "p90_terminal")
    expected_profit = _pick_first_numeric(projection_summary, "expected_profit")
    total_contributed = _pick_first_numeric(projection_summary, "total_contributed")
    loss_prob = _pick_first_numeric(projection_summary, "probability_of_loss_vs_contributions")
    goal_prob = _pick_first_numeric(projection_summary, "probability_of_reaching_goal")

    lines = ["### What the long-term projection is saying"]
    if expected_terminal is not None or median_terminal is not None:
        parts = []
        if expected_terminal is not None:
            parts.append(f"expected ending wealth around **{_fmt_money0(expected_terminal)}**")
        if median_terminal is not None:
            parts.append(f"median path around **{_fmt_money0(median_terminal)}**")
        lines.append("- The projection currently points to " + " and ".join(parts) + ".")
    if p10_terminal is not None and p90_terminal is not None:
        lines.append(f"- A reasonable bad-to-good range is roughly **{_fmt_money0(p10_terminal)} to {_fmt_money0(p90_terminal)}**. That wide gap is a reminder that long-run averages still come with uncertainty.")
    if total_contributed is not None and expected_profit is not None:
        lines.append(f"- Of the projected outcome, about **{_fmt_money0(total_contributed)}** comes from contributions and around **{_fmt_money0(expected_profit)}** comes from investment growth.")
    if loss_prob is not None:
        lines.append(f"- The model estimates a **{_fmt_pct1(loss_prob)}** chance of ending below total contributions. That does not mean disaster is likely, but it does mean losses are a real part of the distribution.")
    if goal_prob is not None:
        lines.append(f"- If you set a wealth goal, the current estimated chance of reaching it is **{_fmt_pct1(goal_prob)}**.")
    return "\n\n".join(lines)


def _solve_implied_annual_return(*, target_terminal: Optional[float], starting_value: Optional[float], monthly_contribution: Optional[float], horizon_years: Optional[float]) -> Optional[float]:
    if target_terminal is None or horizon_years is None:
        return None
    years = float(horizon_years)
    if years <= 0.0:
        return None
    target = float(target_terminal)
    pv = float(starting_value or 0.0)
    pmt = float(monthly_contribution or 0.0)
    n_months = int(round(years * 12.0))
    if n_months <= 0:
        return None

    def future_value(annual_rate: float) -> float:
        monthly_rate = annual_rate / 12.0
        if abs(monthly_rate) <= 1e-12:
            return pv + pmt * n_months
        growth = (1.0 + monthly_rate) ** n_months
        contrib_leg = pmt * ((growth - 1.0) / monthly_rate)
        return pv * growth + contrib_leg

    lo = -0.90
    hi = 1.20
    fv_lo = future_value(lo)
    fv_hi = future_value(hi)
    if target < fv_lo or target > fv_hi:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fv_mid = future_value(mid)
        if fv_mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _build_long_term_behaviour_section(
    *,
    profile: str,
    cagr: Optional[float],
    max_dd: Optional[float],
    monthly_contribution: Optional[float],
    baseline_monthly: Optional[float],
    required_cut_monthly: Optional[float],
    projection_summary: Dict[str, Any],
) -> str:
    horizon_years = _pick_first_numeric(projection_summary, "horizon_years")
    starting_value = _pick_first_numeric(
        projection_summary,
        "starting_value",
        "starting_wealth",
        "current_savings",
    )
    expected_terminal = _pick_first_numeric(projection_summary, "expected_terminal")
    median_terminal = _pick_first_numeric(projection_summary, "median_terminal")
    p10_terminal = _pick_first_numeric(projection_summary, "p10_terminal")
    p90_terminal = _pick_first_numeric(projection_summary, "p90_terminal")
    goal_amount = _pick_first_numeric(projection_summary, "goal_amount", "wealth_goal")

    target_terminal = expected_terminal if expected_terminal is not None else median_terminal
    implied_return = _solve_implied_annual_return(
        target_terminal=target_terminal,
        starting_value=starting_value,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
    )
    implied_return_low = _solve_implied_annual_return(
        target_terminal=p10_terminal,
        starting_value=starting_value,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
    )
    implied_return_high = _solve_implied_annual_return(
        target_terminal=p90_terminal,
        starting_value=starting_value,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
    )

    lines: List[str] = ["### 4. What this strategy asks from you over the next 20–30 years"]

    behaviour_intro = {
        "Conservative": "This setup is asking for patience and consistency more than heroics. The main job is to keep contributing and not expect spectacular growth every year.",
        "Balanced": "This setup is asking for steady behaviour: keep contributing, accept that some years will look disappointing, and let time do most of the work.",
        "Growth-like": "This setup is asking for real emotional tolerance. The long-run upside only matters if you can keep contributing and avoid bailing out during ugly periods.",
    }
    lines.append(behaviour_intro.get(profile, "This setup mainly asks for consistency, patience, and realistic expectations over a long horizon."))

    discipline_bits: List[str] = []
    if monthly_contribution is not None and monthly_contribution > 0:
        discipline_bits.append(f"keep contributing about **{_fmt_money0(monthly_contribution)}/month**")
    if baseline_monthly is not None and baseline_monthly > 0:
        discipline_bits.append(f"remember that your underlying saving capacity is around **{_fmt_money0(baseline_monthly)}/month**")
    if required_cut_monthly is not None and required_cut_monthly > 0:
        discipline_bits.append(f"which currently relies on roughly **{_fmt_money0(required_cut_monthly)}/month** of behavioural cuts staying in place")
    if discipline_bits:
        lines.append("- **Contribution discipline**: " + ", ".join(discipline_bits) + ".")
        if monthly_contribution is not None and monthly_contribution > 0 and horizon_years is not None and horizon_years >= 5:
            extra_contrib = monthly_contribution * 0.10
            extra_direct = extra_contrib * 12.0 * float(horizon_years)
            lines.append(
                f"- **If contributions rise over time**: even adding about **{_fmt_money0(extra_contrib)}/month** more than today would mean roughly **{_fmt_money0(extra_direct)}** of extra direct contributions over **{int(round(horizon_years))} years**, before compounding is even counted."
            )
            pause_years = 3
            missed_direct = monthly_contribution * 12.0 * pause_years
            lines.append(
                f"- **If you pause for a while**: stopping contributions for **{pause_years} years** would remove about **{_fmt_money0(missed_direct)}** of direct money from the plan, plus the growth that money could have earned later."
            )

    if max_dd is not None:
        dd = abs(float(max_dd))
        trough_10k = 10000.0 * (1.0 - dd)
        lines.append(
            f"- **Behavioural requirement**: the tested path suggests you may need to live through drawdowns around **{_fmt_pct1(dd)}**. In plain English, **£10,000** could temporarily become about **{_fmt_money0(trough_10k)}** without the strategy necessarily being 'broken'."
        )

    if implied_return is not None:
        return_text = f"- **Return path realism**: to end near **{_fmt_money0(target_terminal)}** over about **{int(round(horizon_years or 0))} years**, this setup roughly needs something like **{_fmt_pct1(implied_return)} annualised** from the invested capital, given the current contribution pattern."
        if implied_return_low is not None and implied_return_high is not None:
            lo = min(implied_return_low, implied_return_high)
            hi = max(implied_return_low, implied_return_high)
            return_text += f" A wider bad-to-good projection band roughly maps to something like **{_fmt_pct1(lo)} to {_fmt_pct1(hi)} annualised**, which is another way of saying the path matters a lot."
        lines.append(return_text)
    elif target_terminal is not None and horizon_years is not None:
        lines.append(
            f"- **Return path realism**: the model points to a terminal wealth near **{_fmt_money0(target_terminal)}** over about **{int(round(horizon_years))} years**, but that should be read as a plausible scenario, not a required or guaranteed rate of return."
        )

    if goal_amount is not None and goal_amount > 0:
        lines.append(
            f"- **Reality check versus goals**: if your real aim is around **{_fmt_money0(goal_amount)}**, treat the projection as a probability exercise. Below the implied path, the plan likely falls short; stronger returns or higher contributions make the goal easier."
        )

    practical_lines: List[str] = []
    if monthly_contribution is not None and monthly_contribution > 0:
        practical_lines.append(f"keep contributing around **{_fmt_money0(monthly_contribution)}/month** unless your budget genuinely changes")
    practical_lines.append("review the plan on a slow cadence, like once or twice a year, rather than reacting every week")
    if max_dd is not None:
        practical_lines.append(f"expect occasional painful periods in the zone of **{_fmt_pct1(abs(max_dd))}** drawdowns")
    practical_lines.append("do not treat the median or expected projection as a promise")
    if monthly_contribution is not None and monthly_contribution > 0:
        practical_lines.append("remember that increasing contributions is often more powerful than trying to squeeze out an extra 1–2% of return")
    lines.append("- **Practical plan**: " + "; ".join(practical_lines) + ".")

    return "\n\n".join(lines)


def build_investment_strategy_explanation(inputs: InvestmentExplanationInputs) -> str:
    performance_summary = dict(inputs.performance_summary or {})
    risk_summary = dict(inputs.risk_summary or {})
    investment_context = dict(inputs.investment_context or {})
    projection_summary = dict(inputs.projection_summary or {})

    cagr = _pick_first_numeric(performance_summary, "cagr")
    vol = _pick_first_numeric(
        performance_summary,
        "annual_volatility",
        "annualized_volatility",
        "volatility",
    )
    max_dd = _pick_first_numeric(performance_summary, "max_drawdown")
    sharpe = _pick_first_numeric(performance_summary, "sharpe")
    monthly_contribution = _pick_first_numeric(investment_context, "monthly_contribution")
    weekly_equivalent = _pick_first_numeric(investment_context, "weekly_equivalent")
    baseline_monthly = _pick_first_numeric(investment_context, "baseline_monthly")
    required_cut_monthly = _pick_first_numeric(investment_context, "required_cut_monthly")

    profile, profile_text = _infer_strategy_profile(cagr, vol, max_dd, sharpe)
    risk_note = ""
    if risk_summary:
        risk_note = " The run also includes extra risk diagnostics in `risk_summary`, but this section keeps the explanation focused on the investor experience rather than backend detail."

    lines: List[str] = []
    lines.append("**Step 7 — Understand your strategy**")
    lines.append(f"### 1. Strategy profile\n\n**{profile}**. {profile_text}{risk_note}")

    context_bits = []
    if monthly_contribution is not None and monthly_contribution > 0:
        context_bits.append(f"you are currently feeding the strategy about **{_fmt_money0(monthly_contribution)}/month**")
    if weekly_equivalent is not None and weekly_equivalent > 0:
        context_bits.append(f"which is roughly **{_fmt_money0(weekly_equivalent)}/week**")
    if baseline_monthly is not None and baseline_monthly > 0:
        context_bits.append(f"with baseline saving capacity around **{_fmt_money0(baseline_monthly)}/month**")
    if required_cut_monthly is not None and required_cut_monthly > 0:
        context_bits.append(f"and behavioural cuts of about **{_fmt_money0(required_cut_monthly)}/month**")
    if context_bits:
        lines.append("### 2. What this setup means in context\n\nRight now " + ", ".join(context_bits) + ".")

    lines.append(
        "### 3. What the core metrics mean in practice\n\n" +
        "\n".join([
            _build_cagr_line(cagr, projection_summary),
            _build_vol_line(vol),
            _build_drawdown_line(max_dd),
            _build_sharpe_line(sharpe),
        ])
    )

    projection_section = _build_projection_section(projection_summary)
    if projection_section:
        lines.append(projection_section)

    lines.append(
        _build_long_term_behaviour_section(
            profile=profile,
            cagr=cagr,
            max_dd=max_dd,
            monthly_contribution=monthly_contribution,
            baseline_monthly=baseline_monthly,
            required_cut_monthly=required_cut_monthly,
            projection_summary=projection_summary,
        )
    )
    lines.append(_build_educational_comparison(profile, cagr, vol, max_dd))
    lines.append("### Bottom line\n\nThis step is not telling you whether the strategy is 'good' in the abstract. It is helping you judge whether the likely growth, the depth of temporary losses, the contribution discipline required, and the emotional difficulty of staying invested actually fit the kind of investor experience you want to simulate.")
    return "\n\n".join(lines)
