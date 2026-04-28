from __future__ import annotations

import streamlit as st

from ui.common.global_sidebar import render_global_sidebar
from ui.state.bootstrap import bootstrap_session_state
from ui.state.keys import CURRENT_STEP
from ui.state.updates import apply_queued_session_updates
from ui.steps.step0_intent import render_step_0
from ui.steps.step1_inputs import render_step_1
from ui.steps.step2_goal import render_step_2
from ui.steps.step3_results import render_step_3
from ui.steps.step4_universe import render_step_4
from ui.steps.step5_workspace import render_step_5
from ui.steps.step6_projection import render_step_6
from ui.steps.step7_insights import render_step_7


ROUTES = {
    0: render_step_0,
    1: render_step_1,
    2: render_step_2,
    3: render_step_3,
    4: render_step_4,
    5: render_step_5,
    6: render_step_6,
    7: render_step_7,
}


def _current_step() -> int:
    """Return the active wizard step with a safe fallback to Step 0."""
    try:
        return int(st.session_state.get(CURRENT_STEP, 0) or 0)
    except Exception:
        return 0


def main() -> None:
    st.set_page_config(page_title="LifeBudget Micro", layout="centered")

    bootstrap_session_state()
    apply_queued_session_updates()

    if CURRENT_STEP not in st.session_state:
        st.session_state[CURRENT_STEP] = 0

    step = _current_step()

    # Shared contextual sidebar. This is intentionally not step navigation;
    # the main wizard still owns Back/Continue routing inside each screen.
    render_global_sidebar()

    # Step 0 has its own landing title, so avoid showing the app title twice.
    if step != 0:
        st.title("LifeBudget Micro")

    ROUTES.get(step, render_step_0)()


if __name__ == "__main__":
    main()
