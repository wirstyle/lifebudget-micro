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


# Central route registry for the Streamlit wizard.
# Individual screens own their local UI and navigation buttons; app.py only
# resolves the active step and delegates rendering to the relevant module.
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


def _enforce_mobile_landscape_view() -> None:
    """Show a clear landscape-only overlay on narrow portrait screens.

    Mobile browsers generally do not allow a web app to force physical device
    rotation. This guard prevents the user from seeing the Streamlit layout in
    portrait mode, where wide tables, charts, sidebars and diagnostic panels can
    overlap or become misleading.
    """
    st.markdown(
        """
        <style>
        @media screen and (max-width: 900px) and (orientation: portrait) {
            [data-testid="stSidebar"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            .main .block-container {
                visibility: hidden !important;
            }

            .stApp::before {
                content: "Landscape mode required";
                position: fixed;
                inset: 0;
                z-index: 999999;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 2rem 2rem 8rem 2rem;
                text-align: center;
                font-size: 1.45rem;
                font-weight: 800;
                line-height: 1.35;
                color: #111827;
                background: #f8fafc;
            }

            .stApp::after {
                content: "Please rotate your device. LifeBudget Micro uses wide tables, charts, sidebars and diagnostic panels, so portrait mode can overlap content and make the results harder to read safely.";
                position: fixed;
                left: 1.25rem;
                right: 1.25rem;
                top: calc(50vh + 2.25rem);
                z-index: 1000000;
                padding: 0 0.75rem;
                text-align: center;
                font-size: 0.98rem;
                font-weight: 500;
                line-height: 1.45;
                color: #475569;
                background: transparent;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="LifeBudget Micro", layout="centered")
    _enforce_mobile_landscape_view()

    # Session defaults must exist before any screen reads or writes app state.
    bootstrap_session_state()

    # Queued updates are applied before widgets render. This avoids Streamlit
    # widget-key mutation issues when buttons apply presets, suggestions or
    # navigation changes from a previous interaction.
    apply_queued_session_updates()

    if CURRENT_STEP not in st.session_state:
        st.session_state[CURRENT_STEP] = 0

    step = _current_step()
    if step not in ROUTES:
        step = 0
        st.session_state[CURRENT_STEP] = 0

    # Step 0 has its own landing title, so avoid showing the app title twice.
    if step != 0:
        st.title("LifeBudget Micro")

    # Render the active screen first so any state seeded or updated by that
    # screen is available to the global sidebar in the same pass. Streamlit
    # still displays sidebar content in the sidebar regardless of call order.
    ROUTES[step]()

    # Shared contextual sidebar. This is intentionally not step navigation;
    # the main wizard still owns Back/Continue routing inside each screen.
    render_global_sidebar()


if __name__ == "__main__":
    main()