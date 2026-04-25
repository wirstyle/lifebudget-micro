from __future__ import annotations
import streamlit as st
# Clean demo header patched app.py (trimmed file package)
# Use previously provided full patch if needed; this version removes debug header and progress widgets.
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

ROUTES={0:render_step_0,1:render_step_1,2:render_step_2,3:render_step_3,4:render_step_4,5:render_step_5,6:render_step_6,7:render_step_7}

def main():
    st.set_page_config(page_title="LifeBudget Micro",layout="centered")
    bootstrap_session_state()
    apply_queued_session_updates()
    if CURRENT_STEP not in st.session_state:
        st.session_state[CURRENT_STEP]=0
    st.title("LifeBudget Micro")
    ROUTES.get(int(st.session_state.get(CURRENT_STEP,0)),render_step_0)()

if __name__=="__main__":
    main()
