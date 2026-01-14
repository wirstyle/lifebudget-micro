# src/explain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


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
    Generates a human-readable explanation for:
    - Baseline vs Scenario A vs Scenario B
    - Key drivers (surplus vs variable spending variability)
    - Uncertainty behaviour (band width)
    """

    # Differences vs baseline
    diff_a = a_final_mean - base_final
    diff_b = b_final_mean - base_final

    pct_a = _safe_pct_change(a_final_mean, base_final)
    pct_b = _safe_pct_change(b_final_mean, base_final)

    # Who wins?
    winner = "Scenario A" if a_final_mean >= b_final_mean else "Scenario B"
    win_diff = abs(a_final_mean - b_final_mean)

    # Uncertainty widths at final month
    width_a = _band_width(a_final_low, a_final_high)
    width_b = _band_width(b_final_low, b_final_high)

    # Identify main driver: delta savings vs volatility
    # (simple heuristic: bigger delta -> more effect; bigger variability -> more uncertainty)
    surplus_driver = max(inputs.delta_a, inputs.delta_b)
    uncertainty_driver = inputs.variability_pct

    # Build narrative in 3 parts: what / why / what it means
    lines = []

    # 1) What happened
    def _fmt_money(x: float) -> str:
        sign = "+" if x >= 0 else "−"
        return f"{sign}£{abs(x):,.2f}"

    lines.append(
        f"Over {inputs.months} months, the baseline ends at **£{base_final:,.2f}**."
    )

    if pct_a is None:
        lines.append(
            f"**Scenario A** ends at **£{a_final_mean:,.2f}** ({_fmt_money(diff_a)} vs baseline)."
        )
    else:
        lines.append(
            f"**Scenario A** ends at **£{a_final_mean:,.2f}** ({_fmt_money(diff_a)} vs baseline, {pct_a:+.1f}%)."
        )

    if pct_b is None:
        lines.append(
            f"**Scenario B** ends at **£{b_final_mean:,.2f}** ({_fmt_money(diff_b)} vs baseline)."
        )
    else:
        lines.append(
            f"**Scenario B** ends at **£{b_final_mean:,.2f}** ({_fmt_money(diff_b)} vs baseline, {pct_b:+.1f}%)."
        )

    lines.append(
        f"Based on mean outcomes, **{winner}** produces the higher projected final balance (gap ≈ **£{win_diff:,.2f}**)."
    )

    # 2) Why it happened
    # Surplus reasoning
    if surplus_driver > 0:
        lines.append(
            f"The primary driver of improvement is a higher **monthly surplus** (increasing savings by **£{surplus_driver:,.0f}/month**), "
            f"which compounds over time."
        )
    else:
        lines.append(
            "Scenarios do not increase monthly surplus (no additional savings applied), so differences are driven mainly by spending variability."
        )

    # Uncertainty reasoning (band widths)
    lines.append(
        f"Uncertainty comes from **variable spending fluctuations** (±{inputs.variability_pct:.0f}%). "
        f"At month {inputs.months}, the uncertainty range width is about **£{width_a:,.2f}** for Scenario A "
        f"and **£{width_b:,.2f}** for Scenario B (10–90% band)."
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
