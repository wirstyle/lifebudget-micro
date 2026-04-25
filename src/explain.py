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




def build_step3_next_actions_markdown(
    *,
    intent: str,
    margin_w: float,
    target_a: float,
    disc: float,
    structural_deficit: bool,
    short_term_goal_amount: float = 0.0,
    short_term_goal_weeks: int = 0,
    short_term_goal_required_weekly: float = 0.0,
    structural_deficit_tips: str = "",
) -> Dict[str, str]:
    """
    Narrative block for Step 3: 'What to do next'.

    Returns:
      {
        "level": "success" | "info" | "warning" | "error",
        "title": str,
        "body": markdown_text,
      }
    """
    intent_name = str(intent or "not_sure_yet")
    margin_w = float(margin_w or 0.0)
    target_a = float(target_a or 0.0)
    disc = float(disc or 0.0)
    short_term_goal_amount = float(short_term_goal_amount or 0.0)
    short_term_goal_weeks = int(short_term_goal_weeks or 0)
    short_term_goal_required_weekly = float(short_term_goal_required_weekly or 0.0)

    if structural_deficit:
        body_lines: List[str] = [
            "You’re in structural deficit territory. The immediate goal is not optimisation — it is getting back to a sustainable weekly position first."
        ]
        if structural_deficit_tips:
            body_lines.append(structural_deficit_tips)
        return {
            "level": "warning",
            "title": "What to do next (short)",
            "body": "\n\n".join(body_lines),
        }

    need = max(target_a - max(margin_w, 0.0), 0.0)

    if intent_name == "avoid_overspending":
        if need <= 0.0:
            return {
                "level": "success",
                "title": "What to do next (short)",
                "body": "Your cash flow already looks stable enough to stay at or above break-even under the current assumptions. This mode is about protecting that stability.",
            }
        if need <= disc:
            return {
                "level": "info",
                "title": "What to do next (short)",
                "body": f"To make break-even more resilient, reduce flexible spending by about **{_fmt_gbp0(need)}/week** and test the plan again with one-off events.",
            }
        return {
            "level": "error",
            "title": "What to do next (short)",
            "body": "Even this stabilisation-first mode still needs a structural change because discretionary cuts alone are not enough.",
        }

    if intent_name == "save_more_each_week":
        if need <= 0.0:
            return {
                "level": "success",
                "title": "What to do next (short)",
                "body": "Your chosen savings pace is already covered by your current margin. You can push it a little higher if you want a stronger weekly savings habit.",
            }
        if need <= disc:
            return {
                "level": "info",
                "title": "What to do next (short)",
                "body": f"To save more each week, aim to free up about **{_fmt_gbp0(need)}/week** from discretionary spending and check whether the uncertainty band still feels acceptable.",
            }
        return {
            "level": "error",
            "title": "What to do next (short)",
            "body": "Your current stretch target is too aggressive for discretionary cuts alone. You’ll need either lower essentials or more income to sustain it.",
        }

    if intent_name == "reach_target_balance":
        if short_term_goal_amount > 0.0:
            implied_cash_only = float(target_a) * float(max(short_term_goal_weeks, 1))
            if target_a >= short_term_goal_required_weekly and short_term_goal_required_weekly > 0.0:
                return {
                    "level": "success",
                    "title": "What to do next (short)",
                    "body": (
                        f"At **{_fmt_gbp0(target_a)}/week**, you are pacing fast enough to build a short-term cash buffer of about "
                        f"**{_fmt_gbp0(short_term_goal_amount)}** within roughly **{short_term_goal_weeks} weeks** in cash-only terms."
                    ),
                }
            return {
                "level": "info",
                "title": "What to do next (short)",
                "body": (
                    f"Your cash-buffer goal implies about **{_fmt_gbp0(short_term_goal_required_weekly)}/week**. "
                    f"Your current Plan A target would build roughly **{_fmt_gbp0(implied_cash_only)}** over **{short_term_goal_weeks} weeks** before any investing logic from later steps."
                ),
            }
        if need <= 0.0:
            return {
                "level": "success",
                "title": "What to do next (short)",
                "body": "Your current weekly target already supports the short-term goal framing you selected.",
            }
        return {
            "level": "info",
            "title": "What to do next (short)",
            "body": f"Your current target still needs about **{_fmt_gbp0(need)}/week** more than the baseline to support the short-term goal framing.",
        }

    # default = not_sure_yet
    if need <= 0.0:
        return {
            "level": "success",
            "title": "What to do next (short)",
            "body": "Plan A is already covered by your current margin. Consider setting a slightly higher target.",
        }
    if need <= disc:
        return {
            "level": "info",
            "title": "What to do next (short)",
            "body": f"To reach Plan A (**{_fmt_gbp0(target_a)}/week**), aim to reduce discretionary by about **{_fmt_gbp0(need)}/week**. Start with a small, consistent cut rather than a perfect plan.",
        }
    return {
        "level": "error",
        "title": "What to do next (short)",
        "body": "Plan A is not achievable through discretionary cuts alone. You’ll need a structural change (income ↑ or essentials ↓).",
    }

def build_step3_technical_details_markdown(
    *,
    baseline_df,
    plan_df,
    weeks: int,
    iterations: int,
    variability_frac: float,
    uncertainty_preset: str,
    seed: int,
    target_a: float,
    short_term_goal_amount: float = 0.0,
    short_term_goal_weeks: int | None = None,
    short_term_goal_required_weekly: float = 0.0,
) -> str:
    """
    Full Step 3 technical markdown for the current single-plan flow.
    Keeps Step 3 as UI while narrative + interpretation live in explain.py.
    """
    import pandas as pd

    baseline_final = 0.0
    conservative_final = 0.0
    expected_final = 0.0
    optimistic_final = 0.0

    if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty and "Balance" in baseline_df.columns:
        xs = pd.to_numeric(baseline_df["Balance"], errors="coerce").dropna()
        if not xs.empty:
            baseline_final = float(xs.iloc[-1])

    if isinstance(plan_df, pd.DataFrame) and not plan_df.empty:
        if "Lower" in plan_df.columns:
            xs = pd.to_numeric(plan_df["Lower"], errors="coerce").dropna()
            if not xs.empty:
                conservative_final = float(xs.iloc[-1])
        if "Mean" in plan_df.columns:
            xs = pd.to_numeric(plan_df["Mean"], errors="coerce").dropna()
            if not xs.empty:
                expected_final = float(xs.iloc[-1])
        if "Upper" in plan_df.columns:
            xs = pd.to_numeric(plan_df["Upper"], errors="coerce").dropna()
            if not xs.empty:
                optimistic_final = float(xs.iloc[-1])

    inputs = SinglePlanExplanationInputs(
        weeks=int(weeks),
        variability_pct=float(variability_frac) * 100.0,
        seed=int(seed),
        iters=int(iterations),
    )

    core_text = build_single_plan_explanation(
        baseline_final=float(baseline_final),
        conservative_final=float(conservative_final),
        expected_final=float(expected_final),
        optimistic_final=float(optimistic_final),
        inputs=inputs,
    )

    implied_target_balance = float(target_a) * float(max(int(weeks), 1))
    expected_vs_target = float(expected_final - implied_target_balance)
    conservative_vs_target = float(conservative_final - implied_target_balance)
    range_width = float(optimistic_final - conservative_final)

    def _risk_label(range_width_value: float, horizon_weeks: int) -> str:
        if int(horizon_weeks) <= 0:
            return "Unknown"
        weekly_spread = float(range_width_value) / float(horizon_weeks)
        if weekly_spread < 10.0:
            return "Low"
        if weekly_spread < 25.0:
            return "Moderate"
        return "High"

    def _confidence_label(range_width_value: float, baseline_final_value: float) -> str:
        base = max(abs(float(baseline_final_value)), 1.0)
        ratio = float(range_width_value) / base
        if ratio <= 0.10:
            return "High"
        if ratio <= 0.20:
            return "Medium"
        return "Low"

    risk_level = _risk_label(range_width, int(weeks))
    confidence = _confidence_label(range_width, baseline_final)

    lines: List[str] = []
    lines.append("## Technical explanation (Baseline vs your plan)")
    if core_text:
        lines.append(core_text)

    lines.append("### What this means for you")
    lines.append(f"- In a typical scenario, you end up with about **£{expected_final - baseline_final:+,.0f}** versus baseline.")
    lines.append(f"- But in a bad scenario, you could end up about **£{abs(conservative_final - baseline_final):,.0f}** worse than baseline.")
    lines.append("- This means your plan has upside, but also meaningful downside risk.")

    lines.append("### This is the trade-off")
    lines.append(f"- **Potential upside:** about **£{expected_final - baseline_final:+,.0f}** in the expected case.")
    lines.append(f"- **Downside risk:** about **£{conservative_final - baseline_final:+,.0f}** in the conservative case.")

    lines.append("### Interpretation")
    lines.append(f"- About **1 in 10** outcomes end below **£{conservative_final:,.0f}**.")
    lines.append(f"- About **1 in 2** outcomes are around **£{expected_final:,.0f}**.")
    lines.append(f"- About **1 in 10** outcomes exceed **£{optimistic_final:,.0f}**.")

    lines.append("### Comparison with your Step 2 target")
    lines.append(f"Your weekly target implies about **£{implied_target_balance:,.0f}** by week {int(weeks)}.")
    if expected_vs_target >= 0:
        lines.append(f"- **Expected outcome:** **£{expected_final:,.0f}** → about **£{expected_vs_target:,.0f} above** your target path.")
    else:
        lines.append(f"- **Expected outcome:** **£{expected_final:,.0f}** → about **£{abs(expected_vs_target):,.0f} below** your target path.")
    if conservative_vs_target >= 0:
        lines.append(f"- **Conservative outcome:** **£{conservative_final:,.0f}** → about **£{conservative_vs_target:,.0f} above** your target path.")
    else:
        lines.append(f"- **Conservative outcome:** **£{conservative_final:,.0f}** → about **£{abs(conservative_vs_target):,.0f} below** your target path.")

    lines.append("### Range interpretation")
    if range_width <= max(200.0, implied_target_balance * 0.10):
        lines.append(
            f"This range (~**£{range_width:,.0f}**) is relatively contained for a {int(weeks)}-week horizon, "
            "which suggests the plan is not extremely sensitive to spending variability."
        )
    else:
        lines.append(
            f"This range (~**£{range_width:,.0f}**) is relatively wide for a {int(weeks)}-week horizon, "
            "which indicates sensitivity to spending variability."
        )

    lines.append("### Risk label")
    lines.append(f"**Risk level:** {risk_level}")
    lines.append(f"**Confidence:** {confidence}")

    if float(short_term_goal_amount or 0.0) > 0.0:
        resolved_goal_weeks = int(short_term_goal_weeks or weeks)
        lines.append(
            f"Short-term cash-buffer overlay: this scenario is also being compared against a cash-only goal of "
            f"**£{float(short_term_goal_amount):,.0f}** over **{resolved_goal_weeks} weeks**, "
            f"which implies about **£{float(short_term_goal_required_weekly or 0.0):,.0f}/week**."
        )

    lines.append(
        f"Your plan is evaluated with the real short-horizon Monte Carlo layer using **{int(iterations)}** iterations "
        f"and variability of about **±{int(round(float(variability_frac) * 100.0))}%** "
        f"on the combined variable + discretionary spending bucket ({str(uncertainty_preset)})."
    )
    lines.append(
        f"Reproducibility note: results are generated with seed **{int(seed)}** over **{int(iterations)}** Monte Carlo iterations."
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


# ============================================================
# Engine structural explanation (5G)
# ============================================================

def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            mapped = to_dict()
            if isinstance(mapped, dict):
                return dict(mapped)
        except Exception:
            pass
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_num(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _fmt_signed_pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:+.1f}%"


def _label_from_concentration(cfg_map: Dict[str, Any]) -> str:
    top_k = _safe_num(cfg_map.get("top_k"))
    weight_shrink = _safe_num(cfg_map.get("weight_shrink"))
    temperature = _safe_num(cfg_map.get("temperature"))
    if top_k is not None and top_k <= 8:
        return "fairly concentrated"
    if top_k is not None and top_k >= 18:
        return "broadly diversified"
    if weight_shrink is not None and weight_shrink >= 0.20:
        return "more diversified"
    if temperature is not None and temperature <= 0.75:
        return "more concentrated"
    return "moderately diversified"


def _build_metric_tradeoff_lines(perf: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    supports: List[str] = []
    tradeoffs: List[str] = []
    sharpe = _safe_num(perf.get("sharpe"))
    cagr = _safe_num(perf.get("cagr"))
    max_dd = _safe_num(perf.get("max_drawdown", perf.get("max_dd")))
    turnover = _safe_num(perf.get("mean_turnover"))
    diversification = _safe_num(perf.get("diversification", perf.get("mean_diversification_ratio")))
    ann_vol = _safe_num(perf.get("annual_volatility", perf.get("annualized_volatility")))

    if sharpe is not None:
        if sharpe >= 1.0:
            supports.append("risk-adjusted return is strong, so the strategy is not relying only on raw upside")
        elif sharpe >= 0.5:
            supports.append("risk-adjusted return is positive, which suggests the edge is not purely nominal CAGR")
        else:
            tradeoffs.append("risk-adjusted return is still modest, so headline gains should be interpreted carefully")

    if max_dd is not None:
        dd_abs = abs(max_dd)
        if dd_abs <= 0.10:
            supports.append("drawdown stayed contained, which points to effective risk control")
        elif dd_abs >= 0.20:
            tradeoffs.append("drawdown is materially elevated, so the path may be harder to tolerate in practice")

    if turnover is not None:
        if turnover <= 0.20:
            supports.append("turnover stayed relatively controlled, which helps defend robustness after costs")
        elif turnover >= 0.60:
            tradeoffs.append("turnover is quite high, which increases the risk that part of the edge is fragile or cost-sensitive")

    if diversification is not None:
        if diversification >= 0.60:
            supports.append("diversification remained meaningful, so performance is less likely to come from one narrow bet")
        elif diversification <= 0.30:
            tradeoffs.append("diversification is limited, so results may rely on a narrower set of exposures")

    if cagr is not None and ann_vol is not None:
        if cagr >= 0.08 and ann_vol <= 0.18:
            supports.append("return and volatility stayed in a relatively balanced range")
        elif cagr >= 0.10 and ann_vol >= 0.22:
            tradeoffs.append("upside appears to come with a clear volatility cost")

    return supports, tradeoffs


def explain_why_config_works(
    cfg: Any,
    results: Any,
    coherence: Any,
    philosophy: Any,
) -> Dict[str, Any]:
    """Build a defendable natural-language explanation for engine results.

    Returns a structured payload with short headline, key drivers, trade-offs,
    coherence framing and a pre-rendered markdown block for the UI.
    """
    cfg_map = _coerce_mapping(cfg)
    results_map = _coerce_mapping(results)
    coherence_map = _coerce_mapping(coherence)
    philosophy_name = str(philosophy or coherence_map.get("philosophy") or "Balanced")

    perf = _coerce_mapping(results_map.get("performance_summary", results_map))
    sharpe = _safe_num(perf.get("sharpe"))
    cagr = _safe_num(perf.get("cagr"))
    max_dd = _safe_num(perf.get("max_drawdown", perf.get("max_dd")))
    turnover = _safe_num(perf.get("mean_turnover"))
    ann_vol = _safe_num(perf.get("annual_volatility", perf.get("annualized_volatility")))

    signal_mode = str(cfg_map.get("signal_mode") or "unknown")
    overlay_mode = str(cfg_map.get("probabilistic_mode") or cfg_map.get("probabilistic_mode_effective") or "none")
    top_k = _safe_num(cfg_map.get("top_k"))
    covariance_hint = "on" if bool(cfg_map.get("ewma_sigma", False)) or (_safe_num(cfg_map.get("correlation_penalty_strength")) or 0.0) > 1e-9 or _safe_num(cfg_map.get("target_portfolio_vol_monthly")) is not None else "off"
    concentration_label = _label_from_concentration(cfg_map)

    drivers: List[str] = []
    tradeoffs: List[str] = []

    # Structural driver lines
    if signal_mode != "unknown":
        drivers.append(f"the engine is leaning on **{signal_mode}** as its primary signal contract")
    if top_k is not None and top_k > 0:
        drivers.append(f"portfolio construction uses **top_k ≈ {int(round(top_k))}**, which makes the posture **{concentration_label}**")
    if overlay_mode != "none":
        drivers.append(f"a **{overlay_mode}** overlay is active, so the final weights are not driven by point estimates alone")
    else:
        drivers.append("the result is coming mostly from the core ranking/allocation engine rather than from an extra overlay layer")
    if covariance_hint == "on":
        drivers.append("risk control is being supported by covariance / volatility-aware guardrails")
    else:
        tradeoffs.append("covariance-style risk control looks limited, so path risk may rely more on the base signal than on explicit portfolio defence")

    metric_supports, metric_tradeoffs = _build_metric_tradeoff_lines(perf)
    drivers.extend(metric_supports)
    tradeoffs.extend(metric_tradeoffs)

    coherence_label = str(coherence_map.get("label") or "unavailable")
    coherence_score = _safe_num(coherence_map.get("score_continuous"))
    coherence_reasons = [str(x) for x in list(coherence_map.get("reasons") or []) if str(x).strip()]
    coherence_warnings = [str(x) for x in list(coherence_map.get("warnings") or []) if str(x).strip()]

    coherence_sentence = f"This configuration is **{coherence_label}** relative to the **{philosophy_name}** philosophy"
    if coherence_score is not None:
        coherence_sentence += f" (coherence score {coherence_score:.2f})"
    coherence_sentence += "."

    if coherence_reasons:
        drivers.append(f"structurally, it fits because **{coherence_reasons[0]}**")
    if coherence_warnings:
        tradeoffs.append(f"the main structural warning is: **{coherence_warnings[0]}**")

    # Headline / summary
    headline_bits: List[str] = []
    if sharpe is not None:
        headline_bits.append(f"Sharpe {sharpe:.2f}")
    if cagr is not None:
        headline_bits.append(f"CAGR {_fmt_signed_pct(cagr * 100.0) if abs(cagr) <= 1.0 else f'{cagr:.2f}'}")
    headline = " · ".join(headline_bits) if headline_bits else "Engine explanation"

    summary_lines: List[str] = []
    if sharpe is not None and sharpe >= 1.0:
        summary_lines.append("Sharpe improved because the signal and portfolio construction appear to be working together, not just because the strategy took more raw risk.")
    elif sharpe is not None and cagr is not None and cagr > 0:
        summary_lines.append("The strategy is generating positive returns, but the case is strongest when read as a balance between return and risk rather than CAGR alone.")
    else:
        summary_lines.append("The current result is more exploratory than fully convincing, so the explanation should be read as a structural interpretation rather than a claim of strong edge.")

    if max_dd is not None:
        if abs(max_dd) <= 0.10:
            summary_lines.append("Drawdown stayed controlled because the configuration keeps the risk posture reasonably contained.")
        elif abs(max_dd) >= 0.20:
            summary_lines.append("Drawdown stayed difficult to contain, which suggests the upside is being bought with a materially rougher path.")

    summary_lines.append(coherence_sentence)

    if cagr is not None and max_dd is not None:
        if cagr > 0 and abs(max_dd) <= 0.12:
            tradeoffs.insert(0, "main trade-off: some upside may have been sacrificed in exchange for cleaner risk control")
        elif cagr > 0 and abs(max_dd) > 0.18:
            tradeoffs.insert(0, "main trade-off: higher upside is coming with a clearly heavier drawdown burden")

    # Deduplicate while preserving order
    def _dedupe(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in items:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    drivers = _dedupe(drivers)[:5]
    tradeoffs = _dedupe(tradeoffs)[:4]
    summary_lines = _dedupe(summary_lines)

    markdown_parts: List[str] = [
        "### Why this configuration works",
        "\n".join(summary_lines),
    ]
    if drivers:
        markdown_parts.append("**Main supporting reasons**")
        markdown_parts.extend([f"- {x}" for x in drivers])
    if tradeoffs:
        markdown_parts.append("**Main trade-offs / caveats**")
        markdown_parts.extend([f"- {x}" for x in tradeoffs])
    if ann_vol is not None or turnover is not None:
        stats_bits = []
        if ann_vol is not None:
            stats_bits.append(f"annual vol ≈ {ann_vol:.2f}")
        if turnover is not None:
            stats_bits.append(f"mean turnover ≈ {turnover:.2f}")
        if stats_bits:
            markdown_parts.append("**Context**")
            markdown_parts.append("- " + " · ".join(stats_bits))

    markdown_text = "\n\n".join(markdown_parts)
    return {
        "headline": headline,
        "summary": summary_lines,
        "drivers": drivers,
        "tradeoffs": tradeoffs,
        "coherence_sentence": coherence_sentence,
        "signal_mode": signal_mode,
        "overlay_mode": overlay_mode,
        "philosophy": philosophy_name,
        "markdown": markdown_text,
    }


def summarize_governed_parameter_tradeoffs(parameter_rows: List[Dict[str, Any]], *, max_lines: int = 3) -> str:
    rows = [dict(x or {}) for x in list(parameter_rows or [])]
    interesting = [r for r in rows if bool(r.get("changed")) and str(r.get("status", "coherent")) in {"stretched", "discouraged"}]
    if not interesting:
        interesting = [r for r in rows if bool(r.get("changed"))]
    if not interesting:
        return "Manual parameters remain close to the coherent base, so the current advanced state does not materially change the structural posture."

    lines: List[str] = []
    for row in interesting[:max(1, int(max_lines))]:
        title = str(row.get("title", row.get("param_key", "parameter")) or "parameter")
        current_value = str(row.get("current_value_display", row.get("current_value", "—")) or "—")
        status = str(row.get("status", "coherent") or "coherent")
        reading = str(row.get("reading", "") or "").strip()
        structural_shift = str(row.get("structural_shift", "") or "").strip()
        parts = [f"**{title} = {current_value}**"]
        if structural_shift and structural_shift != "aligned":
            parts.append(structural_shift)
        if reading:
            parts.append(reading)
        sentence = " — ".join(parts)
        if status == "discouraged":
            sentence = f"{sentence}. This now sits in a discouraged region for the current philosophy."
        elif status == "stretched":
            sentence = f"{sentence}. This stretches the current philosophy and should be intentional."
        lines.append(f"- {sentence}")

    return "\n".join(lines)


# ============================================================
# Governed traceability explanation (6H)
# ============================================================

def _pretty_key_name(key: Any) -> str:
    raw = str(key or '').strip()
    if not raw:
        return 'field'
    return raw.replace('_', ' ')


def _coerce_trace_rows(trace: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(trace or []):
        mapped = _coerce_mapping(item)
        if mapped:
            rows.append(mapped)
    return rows


def _summarize_patch_keys(patch: Dict[str, Any], *, max_items: int = 6) -> str:
    keys = [_pretty_key_name(k) for k in list(_coerce_mapping(patch).keys())]
    keys = _dedupe_str_list(keys, max_items=max_items)
    if not keys:
        return 'none'
    return ', '.join(keys)


def _infer_traceability_tradeoff(
    *,
    results_map: Dict[str, Any],
    coherence_map: Dict[str, Any],
    philosophy_name: str,
    final_cfg_map: Dict[str, Any],
) -> str:
    perf = _coerce_mapping(results_map.get('performance_summary', results_map))
    max_dd = _safe_num(perf.get('max_drawdown', perf.get('max_dd')))
    cagr = _safe_num(perf.get('cagr'))
    turnover = _safe_num(perf.get('mean_turnover'))
    top_k = _safe_num(final_cfg_map.get('top_k'))
    corr_penalty = _safe_num(final_cfg_map.get('correlation_penalty_strength'))

    if philosophy_name == 'Defensive':
        return 'The main trade-off is accepting less raw upside and a more conservative posture in exchange for stronger drawdown control and structural robustness.'
    if philosophy_name == 'Growth':
        return 'The main trade-off is accepting a bumpier path and looser risk containment in exchange for stronger upside expression.'
    if max_dd is not None and abs(max_dd) > 0.18:
        return 'The main trade-off is that the final configuration still accepts a meaningfully rougher path in order to preserve upside.'
    if turnover is not None and turnover > 0.50:
        return 'The main trade-off is that the configuration remains relatively active, so robustness after costs matters more.'
    if top_k is not None and top_k <= 8:
        return 'The main trade-off is accepting a more concentrated book in exchange for stronger expression of the highest-ranked ideas.'
    if corr_penalty is not None and corr_penalty >= 0.75:
        return 'The main trade-off is accepting tighter risk-aware constraints, which can soften raw signal expression in favourable markets.'
    if cagr is not None and cagr > 0:
        return 'The main trade-off is a balanced one: some upside is intentionally sacrificed so the final posture stays cleaner and more governable.'
    return 'The main trade-off is that the system favours structural coherence and explainability over forcing the most aggressive possible setup.'


def build_governed_traceability_explanation(
    cfg: Any,
    base_cfg: Any,
    coherence: Any,
    philosophy: Any,
    governance_payload: Any = None,
    results: Any = None,
) -> Dict[str, Any]:
    """Build a defendable explanation of how the final governed config came to exist."""
    final_cfg_map = _coerce_mapping(cfg)
    base_cfg_map = _coerce_mapping(base_cfg)
    governance_map = _coerce_mapping(governance_payload)
    coherence_context = _coerce_governance_payload(coherence)
    coherence_map = _coerce_mapping(coherence_context.get('coherence', coherence_context))
    repairs_map = _coerce_mapping(coherence_context.get('repairs', coherence_map.get('repair_plan')))
    results_map = _coerce_mapping(results)

    philosophy_name = _coerce_philosophy_label(
        philosophy
        or governance_map.get('philosophy_effective')
        or governance_map.get('philosophy')
        or coherence_map.get('philosophy')
    )

    base_intention = _coerce_mapping(governance_map.get('base_intention'))
    overrides_applied = _coerce_mapping(governance_map.get('overrides_applied'))
    final_effective_cfg = _coerce_mapping(governance_map.get('final_effective_cfg')) or final_cfg_map
    coherent_base_cfg = _coerce_mapping(governance_map.get('coherent_base_cfg')) or base_cfg_map
    suggested_repairs = _coerce_mapping(governance_map.get('suggested_repairs')) or _coerce_mapping(repairs_map.get('suggested_patch'))
    trace_rows = _coerce_trace_rows(governance_map.get('trace'))

    if not base_intention and trace_rows:
        first_resolution = next((row for row in trace_rows if str(row.get('stage', '')).strip().lower() == 'base_resolution'), None)
        if first_resolution is not None:
            base_intention = _coerce_mapping(first_resolution.get('summary'))

    diff_vs_base = _build_structural_diff(final_effective_cfg, coherent_base_cfg)
    manual_override_count = int(len(overrides_applied))
    effective_change_count = int(len(diff_vs_base))
    coherence_label = str(coherence_map.get('label') or governance_map.get('coherence_status') or 'unavailable')
    coherence_score = _safe_num(coherence_map.get('score_continuous'))
    if coherence_score is None:
        coherence_score = _safe_num(governance_map.get('coherence_score'))

    requested_source = str(base_intention.get('source') or 'unknown')
    requested_strategy = str(base_intention.get('strategy_template') or base_intention.get('strategy') or '—')
    requested_style = str(base_intention.get('style_preset') or '—')
    requested_universe = base_intention.get('universe', '—')
    simple_spec = _coerce_mapping(base_intention.get('simple_spec'))

    recommendation_lines: List[str] = []
    if manual_override_count > 0:
        recommendation_lines.append(f"Manual overrides applied: **{manual_override_count}** field(s) ({_summarize_patch_keys(overrides_applied)}).")
    else:
        recommendation_lines.append('Manual overrides applied: **none**.')

    if suggested_repairs:
        recommendation_lines.append(f"Repairs / governed recommendation path available on: **{_summarize_patch_keys(suggested_repairs)}**.")
    else:
        recommendation_lines.append('Repairs / governed recommendation path: **none currently suggested**.')

    why_still_coherent_parts: List[str] = []
    reasons = [str(x) for x in list(coherence_map.get('reasons') or []) if str(x).strip()]
    warnings = [str(x) for x in list(coherence_map.get('warnings') or []) if str(x).strip()]
    if reasons:
        why_still_coherent_parts.append(reasons[0])
    if not why_still_coherent_parts and coherence_label not in {'unavailable', ''}:
        why_still_coherent_parts.append(f'the final configuration remains {coherence_label} relative to the {philosophy_name} philosophy')
    if warnings:
        why_still_coherent_parts.append(f'with the main structural caution being: {warnings[0]}')

    principal_tradeoff = _infer_traceability_tradeoff(
        results_map=results_map,
        coherence_map=coherence_map,
        philosophy_name=philosophy_name,
        final_cfg_map=final_effective_cfg,
    )

    markdown_parts: List[str] = ['### How this final configuration came to exist']
    markdown_parts.append(
        f"**Chosen philosophy**: **{philosophy_name}**. "
        f"This is the governing lens used to judge whether the final posture still makes structural sense."
    )

    intention_bits = [f"source = **{requested_source}**"]
    if requested_strategy and requested_strategy != '—':
        intention_bits.append(f"strategy template = **{requested_strategy}**")
    if requested_style and requested_style != '—':
        intention_bits.append(f"style preset = **{requested_style}**")
    if requested_universe != '—':
        intention_bits.append(f"universe = **{requested_universe}**")
    markdown_parts.append('**Original user / system intention**: ' + '; '.join(intention_bits) + '.')

    if simple_spec:
        simple_fields = []
        for key in ['risk_appetite', 'diversification', 'stability', 'turnover_pref', 'drawdown_protection', 'overlay_intensity', 'signal_confidence', 'simplicity']:
            value = _safe_num(simple_spec.get(key))
            if value is not None:
                simple_fields.append(f"{_pretty_key_name(key)}={value:.2f}")
        if simple_fields:
            markdown_parts.append('**Resolved simple-mode intent**: ' + ' · '.join(simple_fields[:8]) + '.')

    markdown_parts.append(
        f"**System-proposed coherent base config**: this is the governed baseline produced before manual drift, recommendation patches or repair patches. "
        f"The final config differs from that base on **{effective_change_count}** field(s)."
    )

    markdown_parts.append('**Manual / applied changes**')
    markdown_parts.extend([f"- {line}" for line in recommendation_lines])

    if trace_rows:
        trace_bullets = []
        for row in trace_rows[:8]:
            stage = str(row.get('stage') or row.get('kind') or 'stage')
            detail_parts = []
            for key in ['status', 'label', 'philosophy', 'repair_patch_available']:
                if key in row and row.get(key) not in (None, '', []):
                    detail_parts.append(f"{_pretty_key_name(key)}={row.get(key)}")
            trace_bullets.append(f"- **{stage}**" + (f": {' · '.join(detail_parts)}" if detail_parts else ''))
        markdown_parts.append('**Trace path**')
        markdown_parts.extend(trace_bullets)

    markdown_parts.append(
        f"**Final effective config**: this is the configuration actually left standing after governance resolution. "
        f"Coherence is currently **{coherence_label}**" + (f" with score **{coherence_score:.2f}**." if coherence_score is not None else '.')
    )

    if why_still_coherent_parts:
        markdown_parts.append('**Why it is still coherent**')
        markdown_parts.extend([f"- {part}" for part in why_still_coherent_parts])

    markdown_parts.append(f"**Principal trade-off assumed**: {principal_tradeoff}")

    return {
        'philosophy': philosophy_name,
        'base_intention': base_intention,
        'coherent_base_cfg': coherent_base_cfg,
        'final_effective_cfg': final_effective_cfg,
        'overrides_applied': overrides_applied,
        'suggested_repairs': suggested_repairs,
        'trace': trace_rows,
        'coherence_label': coherence_label,
        'coherence_score': coherence_score,
        'why_still_coherent': why_still_coherent_parts,
        'principal_tradeoff': principal_tradeoff,
        'effective_change_count': effective_change_count,
        'markdown': '\n\n'.join(markdown_parts),
    }

# ============================================================
# Governed actionable recommendations (6F)
# ============================================================

def _safe_int(value: Any) -> Optional[int]:
    try:
        v = int(value)
    except Exception:
        return None
    return v


def _coerce_philosophy_label(value: Any) -> str:
    raw = str(value or "Balanced").strip().capitalize()
    if raw in {"Growth", "Balanced", "Defensive"}:
        return raw
    return "Balanced"


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "1", "yes", "on"}:
            return True
        if raw in {"false", "0", "no", "off"}:
            return False
    try:
        return bool(value)
    except Exception:
        return default


def _dedupe_str_list(items: List[str], *, max_items: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in list(items or []):
        key = str(item or "").strip()
        if not key:
            continue
        lowered = key.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(key)
        if max_items is not None and len(out) >= int(max_items):
            break
    return out


def _coerce_governance_payload(coherence: Any) -> Dict[str, Any]:
    payload = _coerce_mapping(coherence)
    if not payload:
        return {}
    if "coherence" in payload or "repairs" in payload or "constraints" in payload or "blocks" in payload:
        return payload
    return {"coherence": payload}


def _extract_governed_context(
    cfg_map: Dict[str, Any],
    coherence: Any,
    philosophy: Any,
) -> Dict[str, Any]:
    governance = _coerce_governance_payload(coherence)
    coherence_map = _coerce_mapping(governance.get("coherence", governance))
    repairs_map = _coerce_mapping(governance.get("repairs", coherence_map.get("repair_plan")))
    constraints = _coerce_mapping(governance.get("constraints"))
    blocks = _coerce_mapping(governance.get("blocks", coherence_map.get("blocks")))
    philosophy_name = _coerce_philosophy_label(
        philosophy
        or governance.get("philosophy")
        or coherence_map.get("philosophy")
        or repairs_map.get("philosophy")
    )

    if not (constraints and blocks and repairs_map):
        try:
            from src.coherence import build_coherence_governance_payload
            rebuilt = _coerce_mapping(build_coherence_governance_payload(cfg_map, philosophy_name))
        except Exception:
            rebuilt = {}
        if rebuilt:
            governance = {**rebuilt, **governance}
            coherence_map = _coerce_mapping(governance.get("coherence", coherence_map))
            repairs_map = _coerce_mapping(governance.get("repairs", repairs_map))
            constraints = _coerce_mapping(governance.get("constraints", constraints))
            blocks = _coerce_mapping(governance.get("blocks", blocks))

    if not blocks:
        try:
            from src.coherence import extract_config_blocks
            blocks = _coerce_mapping(extract_config_blocks(cfg_map))
        except Exception:
            blocks = {}

    if not constraints:
        try:
            from src.coherence import resolve_philosophy_to_constraints
            constraints = _coerce_mapping(resolve_philosophy_to_constraints(philosophy_name, universe="medium"))
        except Exception:
            constraints = {}

    return {
        "governance": governance,
        "coherence": coherence_map,
        "repairs": repairs_map,
        "constraints": constraints,
        "blocks": blocks,
        "philosophy": philosophy_name,
    }


def _build_structural_diff(cfg_map: Dict[str, Any], base_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diff: Dict[str, Dict[str, Any]] = {}
    keys = sorted(set(cfg_map.keys()) | set(base_map.keys()))
    for key in keys:
        current = cfg_map.get(key)
        base = base_map.get(key)
        try:
            equal = current == base
        except Exception:
            equal = False
        if equal:
            continue
        diff[key] = {"current": current, "base": base}
    return diff


def _has_structural_change(diff: Dict[str, Dict[str, Any]], *keys: str) -> bool:
    return any(k in diff for k in keys)


def _merge_patch(*patches: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        for key, value in patch.items():
            merged[key] = value
    return merged


def _clean_patch(cfg_map: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in dict(patch or {}).items():
        if value is None:
            continue
        try:
            same = cfg_map.get(key) == value
        except Exception:
            same = False
        if same:
            continue
        out[key] = value
    return out


def _append_rec(
    out: List[Dict[str, Any]],
    *,
    title: str,
    why: str,
    expected_benefit: str,
    tradeoff: str,
    patch: Dict[str, Any],
    kind: Optional[str] = None,
) -> None:
    clean = {
        "title": str(title or "").strip(),
        "why": str(why or "").strip(),
        "expected_benefit": str(expected_benefit or "").strip(),
        "tradeoff": str(tradeoff or "").strip(),
        "patch": dict(patch or {}),
    }
    if kind:
        clean["kind"] = str(kind)
    if not clean["title"] or not clean["patch"]:
        return
    out.append(clean)


def build_governed_recommendations(
    cfg: Any,
    base_cfg: Any,
    coherence: Any,
    philosophy: Any,
) -> List[Dict[str, Any]]:
    """
    Build actionable governed recommendations for the current configuration.

    Contract:
    - never executes the engine
    - relies on coherence / structural posture / diffs vs coherent base
    - returns a UI-ready list of dicts with:
        title, why, expected_benefit, tradeoff, patch
    """
    cfg_map = _coerce_mapping(cfg)
    base_map = _coerce_mapping(base_cfg)
    context = _extract_governed_context(cfg_map, coherence, philosophy)

    coherence_map = _coerce_mapping(context.get("coherence"))
    repairs_map = _coerce_mapping(context.get("repairs"))
    constraints = _coerce_mapping(context.get("constraints"))
    blocks = _coerce_mapping(context.get("blocks"))
    philosophy_name = _coerce_philosophy_label(context.get("philosophy"))

    diff = _build_structural_diff(cfg_map, base_map)
    top_k_range = constraints.get("top_k_range", (0, 999))
    if isinstance(top_k_range, (list, tuple)) and len(top_k_range) >= 2:
        low_k_raw, high_k_raw = top_k_range[0], top_k_range[1]
    else:
        low_k_raw, high_k_raw = 0, 999
    try:
        low_k = int(low_k_raw)
    except Exception:
        low_k = 0
    try:
        high_k = int(high_k_raw)
    except Exception:
        high_k = 999

    top_k = _safe_int(cfg_map.get("top_k"))
    score = _safe_num(coherence_map.get("score_continuous"))
    label = str(coherence_map.get("label") or "unavailable").strip().lower()
    status = str(repairs_map.get("status") or coherence_map.get("status") or "").strip().lower()
    n_high = _safe_int(repairs_map.get("n_high_severity"))
    actionable_warnings = _dedupe_str_list([str(x) for x in list(repairs_map.get("actionable_warnings", []) or [])], max_items=3)
    suggested_patch = _clean_patch(cfg_map, _coerce_mapping(repairs_map.get("suggested_patch")))
    preferred_signals = [str(x) for x in list(constraints.get("preferred_signal_modes", []) or []) if str(x).strip()]
    preferred_overlays = [str(x) for x in list(constraints.get("preferred_overlay_modes", constraints.get("allowed_overlay_modes", [])) or []) if str(x).strip()]
    caps_strength = str(blocks.get("caps_strength") or "")
    covariance_model = str(blocks.get("covariance_model") or "")
    overlay_mode = str(blocks.get("overlay_mode") or "")
    signal_mode = str(blocks.get("signal_mode") or "")
    concentration_allowed = _safe_bool(constraints.get("concentration_allowed"), default=False)

    recommendations: List[Dict[str, Any]] = []

    if suggested_patch:
        why = actionable_warnings[0] if actionable_warnings else f"The current config is structurally stretched relative to {philosophy_name}."
        _append_rec(
            recommendations,
            kind="coherence_repair_only",
            title="Apply coherence repair only",
            why=why,
            expected_benefit="Recover cleaner structural alignment without changing the whole intent of the configuration.",
            tradeoff="Some manually stretched settings may be pulled back toward the governed baseline.",
            patch=suggested_patch,
        )

    safer_patch: Dict[str, Any] = {}
    if covariance_model == "none":
        safer_patch["ewma_sigma"] = True
    if philosophy_name in {"Balanced", "Defensive"}:
        corr_strength = _safe_num(cfg_map.get("correlation_penalty_strength"))
        base_corr_strength = _safe_num(base_map.get("correlation_penalty_strength"))
        safer_patch["correlation_penalty_strength"] = max(x for x in [corr_strength, base_corr_strength, 0.75] if x is not None)
        if _safe_num(cfg_map.get("asset_weight_cap")) is None or (_safe_num(cfg_map.get("asset_weight_cap")) or 1.0) > (0.10 if philosophy_name == "Defensive" else 0.20):
            safer_patch["asset_weight_cap"] = 0.10 if philosophy_name == "Defensive" else 0.20
    if philosophy_name == "Defensive":
        turnover_limit = _safe_num(cfg_map.get("turnover_constraint_max_turnover"))
        safer_patch["turnover_penalty_strength"] = max((_safe_num(cfg_map.get("turnover_penalty_strength")) or 0.0), 1.0)
        safer_patch["turnover_constraint_max_turnover"] = min(turnover_limit, 0.25) if turnover_limit is not None else 0.25
        if preferred_overlays and overlay_mode not in preferred_overlays:
            safer_patch["probabilistic_mode"] = preferred_overlays[0]
    if top_k is not None and low_k > 0 and top_k < low_k:
        safer_patch["top_k"] = low_k
    safer_patch = _clean_patch(cfg_map, _merge_patch(suggested_patch, safer_patch))
    if safer_patch and (
        bool(suggested_patch)
        or label in {"mixed", "incoherent", "unavailable"}
        or status in {"repairable", "incompatible"}
        or (score is not None and score < 0.70)
        or (n_high is not None and n_high > 0)
    ):
        _append_rec(
            recommendations,
            kind="safer_current",
            title="Apply safer version of current config",
            why=(
                actionable_warnings[0]
                if actionable_warnings
                else f"The current posture can be made safer while preserving the broad {philosophy_name} intent."
            ),
            expected_benefit="Stronger guardrails and lower structural fragility, with minimal change to the current setup.",
            tradeoff="Risk control becomes more dominant, which can reduce upside expression or flexibility.",
            patch=safer_patch,
        )

    if philosophy_name == "Balanced":
        concentration_patch: Dict[str, Any] = {}
        too_concentrated = (
            (top_k is not None and low_k > 0 and top_k < low_k)
            or ((not concentration_allowed) and top_k is not None and top_k <= 8)
            or caps_strength in {"none", "soft"}
        )
        if too_concentrated:
            if low_k > 0:
                concentration_patch["top_k"] = max(low_k, top_k + 2 if top_k is not None else low_k)
            if _safe_num(cfg_map.get("asset_weight_cap")) is None or (_safe_num(cfg_map.get("asset_weight_cap")) or 1.0) > 0.20:
                concentration_patch["asset_weight_cap"] = 0.20
            current_shrink = _safe_num(cfg_map.get("weight_shrink"))
            base_shrink = _safe_num(base_map.get("weight_shrink"))
            target_shrink_candidates = [x for x in [current_shrink, base_shrink, 0.12] if x is not None]
            concentration_patch["weight_shrink"] = max(target_shrink_candidates) if target_shrink_candidates else 0.12
            concentration_patch = _clean_patch(cfg_map, concentration_patch)
        if concentration_patch:
            why_bits = []
            if top_k is not None and low_k > 0 and top_k < low_k:
                why_bits.append(f"top_k={top_k} sits below the Balanced compatibility band [{low_k}, {high_k}]")
            if caps_strength in {"none", "soft"}:
                why_bits.append(f"caps_strength='{caps_strength}' leaves concentration relatively loose for Balanced")
            why = "; ".join(why_bits) if why_bits else "The current Balanced posture leans too concentrated relative to its own structural target."
            _append_rec(
                recommendations,
                kind="reduce_concentration_balanced",
                title="Reduce concentration for Balanced",
                why=why,
                expected_benefit="Cleaner diversification posture and lower dependence on a narrow subset of bets.",
                tradeoff="A broader book can dilute some upside concentration when the strongest signals are right.",
                patch=concentration_patch,
            )

    covariance_patch: Dict[str, Any] = {}
    covariance_required = _safe_bool(constraints.get("covariance_required"), default=philosophy_name in {"Balanced", "Defensive"})
    if covariance_required and covariance_model == "none":
        covariance_patch["ewma_sigma"] = True
    if philosophy_name in {"Balanced", "Defensive"}:
        curr_corr = _safe_num(cfg_map.get("correlation_penalty_strength"))
        base_corr = _safe_num(base_map.get("correlation_penalty_strength"))
        floor = 0.75 if philosophy_name == "Balanced" else 1.0
        target_corr = max(x for x in [curr_corr, base_corr, floor] if x is not None)
        if curr_corr is None or curr_corr < target_corr:
            covariance_patch["correlation_penalty_strength"] = target_corr
    if philosophy_name != "Growth":
        current_target_vol = _safe_num(cfg_map.get("target_portfolio_vol_monthly"))
        base_target_vol = _safe_num(base_map.get("target_portfolio_vol_monthly"))
        if current_target_vol is None and base_target_vol is not None:
            covariance_patch["target_portfolio_vol_monthly"] = base_target_vol
    covariance_patch = _clean_patch(cfg_map, covariance_patch)
    if covariance_patch and (
        covariance_model == "none"
        or _has_structural_change(diff, "ewma_sigma", "correlation_penalty_strength", "target_portfolio_vol_monthly")
    ):
        _append_rec(
            recommendations,
            kind="strengthen_covariance_guardrails",
            title="Strengthen covariance guardrails",
            why=(
                f"{philosophy_name} expects an active covariance / risk-control layer."
                if covariance_required
                else "The current guardrail block looks lighter than the governed base."
            ),
            expected_benefit="Better path control and cleaner structural defence under the same philosophy.",
            tradeoff="Tighter risk-aware allocation can soften raw signal expression when markets are favourable.",
            patch=covariance_patch,
        )

    if philosophy_name == "Defensive":
        defensive_patch: Dict[str, Any] = {}
        if preferred_signals and signal_mode not in preferred_signals:
            defensive_patch["signal_mode"] = preferred_signals[0]
        if preferred_overlays and overlay_mode not in preferred_overlays:
            defensive_patch["probabilistic_mode"] = preferred_overlays[0]
        if top_k is not None and low_k > 0 and top_k < low_k:
            defensive_patch["top_k"] = low_k
        if _safe_num(cfg_map.get("asset_weight_cap")) is None or (_safe_num(cfg_map.get("asset_weight_cap")) or 1.0) > 0.10:
            defensive_patch["asset_weight_cap"] = 0.10
        defensive_patch["ewma_sigma"] = True
        defensive_patch["correlation_penalty_strength"] = max((_safe_num(cfg_map.get("correlation_penalty_strength")) or 0.0), 1.0)
        defensive_patch["turnover_penalty_strength"] = max((_safe_num(cfg_map.get("turnover_penalty_strength")) or 0.0), 1.0)
        defensive_patch = _clean_patch(cfg_map, defensive_patch)
        if defensive_patch:
            why_parts: List[str] = []
            if signal_mode and preferred_signals and signal_mode not in preferred_signals:
                why_parts.append(f"signal_mode='{signal_mode}' is outside the preferred Defensive set")
            if overlay_mode and preferred_overlays and overlay_mode not in preferred_overlays:
                why_parts.append(f"overlay_mode='{overlay_mode}' is outside the preferred Defensive set")
            if top_k is not None and low_k > 0 and top_k < low_k:
                why_parts.append(f"top_k={top_k} is tighter than the Defensive diversification band [{low_k}, {high_k}]")
            why = "; ".join(why_parts) if why_parts else "The current Defensive setup can be made more coherent without abandoning its risk-first posture."
            _append_rec(
                recommendations,
                kind="improve_sharpe_defensive",
                title="Improve Sharpe without breaking Defensive",
                why=why,
                expected_benefit="A cleaner Defensive structure can improve risk-adjusted quality without turning the profile aggressive.",
                tradeoff="The portfolio may feel more conservative and less upside-seeking in benign regimes.",
                patch=defensive_patch,
            )

    rollback_patch: Dict[str, Any] = {}
    rollback_keys = [
        "signal_mode",
        "probabilistic_mode",
        "top_k",
        "asset_weight_cap",
        "weight_shrink",
        "turnover_penalty_strength",
        "turnover_constraint_max_turnover",
        "correlation_penalty_strength",
        "ewma_sigma",
    ]
    for key in rollback_keys:
        if key in diff and key in base_map:
            rollback_patch[key] = base_map.get(key)
    rollback_patch = _clean_patch(cfg_map, rollback_patch)
    if rollback_patch and not suggested_patch and (
        _has_structural_change(diff, *rollback_keys)
        and ((score is not None and score < 0.75) or label in {"mixed", "unavailable"})
    ):
        _append_rec(
            recommendations,
            kind="return_toward_governed_base",
            title="Move back toward governed base",
            why="The current structure has drifted away from the coherent base even though the repair plan is not a single obvious fix.",
            expected_benefit="Restores the intended governed posture with a small, understandable patch set.",
            tradeoff="This gives up some manual experimentation in exchange for a cleaner default posture.",
            patch=rollback_patch,
        )

    preferred_order = {
        "coherence_repair_only": 0,
        "safer_current": 1,
        "reduce_concentration_balanced": 2,
        "strengthen_covariance_guardrails": 3,
        "improve_sharpe_defensive": 4,
        "return_toward_governed_base": 5,
    }
    deduped: List[Dict[str, Any]] = []
    seen_titles = set()
    for rec in sorted(recommendations, key=lambda x: (preferred_order.get(str(x.get("kind")), 99), str(x.get("title", "")).lower())):
        title_key = str(rec.get("title", "")).strip().lower()
        if not title_key or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        rec["patch"] = _clean_patch(cfg_map, _coerce_mapping(rec.get("patch")))
        if not rec["patch"]:
            continue
        deduped.append(rec)

    return deduped[:5]
