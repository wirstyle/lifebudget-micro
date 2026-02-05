# src/explain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


# -----------------------------
# Inputs used for "technical" explanation
# -----------------------------
@dataclass
class ExplanationInputs:
    income: float
    fixed_expenses: float
    variable_expenses: float
    weeks: int
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
    Human-readable explanation (technical):
    - Baseline vs Scenario A vs Scenario B
    - Driver reasoning (behavioural lever vs variability)
    - Uncertainty behaviour (band width)

    Note:
    In the technical explanation we keep the language "saves £X/week" because at UI-level
    X represents the *weekly savings target* (even if internally the simulator receives a cut).
    """

    diff_a = a_final_mean - base_final
    diff_b = b_final_mean - base_final

    pct_a = _safe_pct_change(a_final_mean, base_final)
    pct_b = _safe_pct_change(b_final_mean, base_final)

    winner = "Scenario A" if a_final_mean >= b_final_mean else "Scenario B"
    win_diff = abs(a_final_mean - b_final_mean)

    width_a = _band_width(a_final_low, a_final_high)
    width_b = _band_width(b_final_low, b_final_high)

    lines = []

    def _fmt_money(x: float) -> str:
        sign = "+" if x >= 0 else "−"
        return f"{sign}£{abs(x):,.2f}"

    # 1) What happened
    lines.append(f"Over {inputs.weeks} weeks, the baseline ends at **£{base_final:,.2f}**.")

    # A/B assumptions (explicit parameter traceability)
    lines.append(
        f"Scenario assumptions: **A saves £{inputs.delta_a:,.0f}/week**, "
        f"**B saves £{inputs.delta_b:,.0f}/week** "
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

    # 2) Why it happened (driver logic consistent with A/B deltas)
    if inputs.delta_a == inputs.delta_b:
        if inputs.delta_a > 0:
            lines.append(
                f"Both scenarios apply the same behavioural change (**£{inputs.delta_a:,.0f}/week** extra savings), "
                "so remaining differences are driven mainly by stochastic variability."
            )
        else:
            lines.append(
                "No additional savings are applied in either scenario, so differences are driven mainly by spending variability."
            )
    else:
        stronger = "Scenario A" if inputs.delta_a > inputs.delta_b else "Scenario B"
        stronger_delta = max(inputs.delta_a, inputs.delta_b)
        lines.append(
            f"The main driver of change is the **behavioural savings lever**: **{stronger}** applies the larger adjustment "
            f"(**£{stronger_delta:,.0f}/week**), which increases weekly surplus and compounds over time."
        )

    # Uncertainty reasoning (band widths)
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

    # 3) What it means (practical takeaway)
    if inputs.variability_pct >= 25:
        lines.append(
            "Because variability is relatively high, outcomes are sensitive to spending behaviour—"
            "reducing variability (more consistent spending) can narrow the uncertainty band."
        )
    else:
        lines.append(
            "Because variability is relatively low, the forecast band stays tighter—"
            "results are more predictable given the same inputs."
        )

    lines.append(
        f"Reproducibility note: results are generated with seed **{inputs.seed}** over **{inputs.iters}** Monte Carlo iterations."
    )

    return "\n\n".join(lines)


# -----------------------------
# Step 4 (HUMAN-FIRST) helpers
# -----------------------------
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

    # NEW semantics (target policy traceability)
    target_a_weekly: float
    target_b_weekly: float
    cut_a_weekly: float
    cut_b_weekly: float
    margin_a_weekly: float
    margin_b_weekly: float

    # Back-compat (keep, but treat as the REAL cut)
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
    # NEW params (preferred)
    target_a_weekly: Optional[float] = None,
    target_b_weekly: Optional[float] = None,
    cut_a_weekly: Optional[float] = None,
    cut_b_weekly: Optional[float] = None,
    margin_a_weekly: Optional[float] = None,
    margin_b_weekly: Optional[float] = None,
    # OLD params (fallback)
    delta_a_weekly: float = 0.0,
    delta_b_weekly: float = 0.0,
) -> ReflectionMetrics:
    """
    Computes summary metrics for the human-first reflection.

    New semantics:
    - target_*_weekly = desired savings goal per week
    - margin_*_weekly = portion covered by existing weekly margin
    - cut_*_weekly    = required discretionary cut per week (real behaviour)

    Backwards compatible:
    - If target_* not provided, assumes target == cut == delta_*_weekly and margin == 0.
    """

    # Fallback mapping if new fields weren't supplied
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

        # keep as “real cut”
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


def _fmt_gbp0(x: float) -> str:
    return f"£{x:,.0f}"


def _fmt_pct(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{p:.1f}%"


def build_human_reflection_text(
    metrics: ReflectionMetrics,
    *,
    shock_enabled: bool,
    shock_amount: float,
    shock_week: int,
) -> Dict[str, str]:
    """
    Returns text blocks for Streamlit rendering (keeps app.py thin).
    Keys:
      - what_you_get
      - pick_primary
      - pick_secondary
      - shock_note
      - variability
    """

    def _coverage_line(target: float, margin: float, cut: float) -> str:
        # margin is already “free” coverage; cut is behavioural sacrifice
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

    shock_note = ""
    if shock_enabled and float(shock_amount) > 0:
        shock_note = (
            f"One-off event included: **{_fmt_gbp0(float(shock_amount))}** happens in **week {int(shock_week)}** "
            "(it reduces the balance from that week onward)."
        )

    variability = "\n".join(
        [
            "### Results can vary (because real weeks aren’t identical)",
            f"- Scenario A likely ends somewhere around **{_fmt_gbp0(metrics.a_low)} to {_fmt_gbp0(metrics.a_high)}**.",
            f"- Scenario B likely ends somewhere around **{_fmt_gbp0(metrics.b_low)} to {_fmt_gbp0(metrics.b_high)}**.",
        ]
    )

    return {
        "what_you_get": what_you_get,
        "pick_primary": pick_primary,
        "pick_secondary": pick_secondary,
        "shock_note": shock_note,
        "variability": variability,
    }
