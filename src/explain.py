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
    return f"£{x:,.0f}"


def _fmt_pct(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{p:.1f}%"


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

    # ------------------------------------------------------------
    # ✅ Label normalisation (makes tips feel "personal", not generic)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Collect + rank items (fixed + variable)
    # ------------------------------------------------------------
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

    # Optional: also use raw variable_parts (nice-to-have)
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

    # ------------------------------------------------------------
    # Start building text
    # ------------------------------------------------------------
    lines: List[str] = []
    lines.append("## Tips to fix a structural deficit (based on your numbers)")
    lines.append("")
    lines.append(
        f"- Your weekly shortfall is about **{_fmt_gbp0(deficit_w)}/week** "
        "(income does not cover essentials)."
    )

    # Discretionary coverage
    if disc > 0:
        cover_pct_disc = min(disc / deficit_w * 100.0, 100.0) if deficit_w > 0 else 0.0
        lines.append(
            f"- Discretionary is **{_fmt_gbp0(disc)}/week**. "
            f"Even cutting it to £0 would cover only **{cover_pct_disc:.0f}%** of the deficit."
        )
    else:
        lines.append("- Discretionary is already **£0/week**, so lifestyle cuts alone cannot fix this.")

    # Essentials ratio (reference only)
    if income > 0:
        ess_pct = pct(essentials) or 0.0
        lines.append(
            f"- Essentials are about **{ess_pct:.0f}% of income** (**{_fmt_gbp0(essentials)}/w**). "
            "As a reference point (not a rule), many budgets aim for **50–65%**."
        )

    # ------------------------------------------------------------
    # Top drivers (the “smart” part)
    # ------------------------------------------------------------
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

        # ✅ Leverage block (robust)
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

            # Top 2 combined
            if len(top_items) >= 2:
                second = float(top_items[1]["weekly"])
                combined = biggest + second
                combined_pct = min(combined / deficit_w * 100.0, 100.0)
                lines.append(
                    f"- Your top two items combined represent about **{combined_pct:.0f}%** of the deficit."
                )

            # ✅ NEW: “equivalences” block (feels very personalised)
            lines.append("")
            lines.append("### What “break-even” means in your terms (equivalences)")
            # Need improvement = deficit_w
            # Express as fraction of top drivers
            eq_bits: List[str] = []
            for it in top_items[:3]:
                w = float(it["weekly"])
                if w > 0:
                    frac = deficit_w / w
                    # show nice rounded
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

    # Optional variable essentials section (only if variable_parts exists)
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
            f"- Scenario A likely ends somewhere around **{_fmt_gbp0(metrics.a_low)} to {_fmt_gbp0(metrics.a_high)}**.",
            f"- Scenario B likely ends somewhere around **{_fmt_gbp0(metrics.b_low)} to {_fmt_gbp0(metrics.b_high)}**.",
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

    return {
        "what_you_get": what_you_get,
        "pick_primary": pick_primary,
        "pick_secondary": pick_secondary,
        "shock_note": shock_note,
        "variability": variability,
        "structural_deficit": structural_msg,
    }
