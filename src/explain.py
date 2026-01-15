# src/explain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExplanationInputs:
    income: float
    fixed_expenses: float
    variable_expenses: float
    months: int
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
    Human-readable explanation:
    - Baseline vs Scenario A vs Scenario B
    - Driver reasoning (behavioural lever vs variability)
    - Uncertainty behaviour (band width)
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
    lines.append(f"Over {inputs.months} months, the baseline ends at **£{base_final:,.2f}**.")

    # A/B assumptions (explicit parameter traceability)
    lines.append(
        f"Scenario assumptions: **A saves £{inputs.delta_a:,.0f}/month**, "
        f"**B saves £{inputs.delta_b:,.0f}/month** "
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
                f"Both scenarios apply the same behavioural change (**£{inputs.delta_a:,.0f}/month** extra savings), "
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
            f"(**£{stronger_delta:,.0f}/month**), which increases monthly surplus and compounds over time."
        )

    # Uncertainty reasoning (band widths)
    lines.append(
        f"Uncertainty comes from **variable spending fluctuations** (±{inputs.variability_pct:.0f}%). "
        f"At month {inputs.months}, the uncertainty range width is about **£{width_a:,.2f}** for Scenario A "
        f"and **£{width_b:,.2f}** for Scenario B (10–90% band)."
    )

    # Bonus micro: which band is wider?
    wider = "Scenario A" if width_a >= width_b else "Scenario B"
    lines.append(
        f"The uncertainty band is slightly wider for **{wider}** at the final month, "
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
