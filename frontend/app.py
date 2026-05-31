import json
import os
import sys

import streamlit as st
from google import genai
from google.genai import types

# Make the repository root importable when Streamlit launches from frontend/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.cognitive_router import run_cognitive_router
from agents.department_agents import (
    generate_civil_defense_plan,
    generate_emergency_response_plan,
    generate_public_works_plan,
)
from memory.insight_manager import load_insights, save_insight
from ml_layer.disaster_models import (
    DisasterClassifier,
    forecast_weather_metrics,
    simulate_weather_ingestion,
)
from tools.context_tools import fetch_news_context, get_live_weather_summary


st.set_page_config(
    page_title="CrisisOps Control Desk",
    page_icon="!",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fafc 0%, #eef3f8 45%, #e8edf3 100%);
        color: #17202a;
    }
    section[data-testid="stSidebar"] {
        background: #101820;
        border-right: 1px solid #253242;
    }
    section[data-testid="stSidebar"] * {
        color: #edf4fb;
    }
    .control-strip {
        padding: 1rem 1.1rem;
        border: 1px solid #d6dee8;
        border-radius: 8px;
        background: #ffffff;
        margin-bottom: 1.2rem;
    }
    .status-tile {
        background: #ffffff;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #d8e0ea;
        box-shadow: 0 10px 26px rgba(29, 48, 67, 0.08);
        text-align: left;
    }
    .status-tile h4 {
        color: #5c6b7a;
        font-size: 0.85rem;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
    }
    .status-tile h2 {
        color: #0f3d5f;
        margin: 0;
    }
    .dispatch-note {
        background: #fff6e5;
        border-left: 5px solid #e59f24;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        color: #513709;
        font-weight: 600;
    }
    .handoff-panel {
        background: #eef8f2;
        border: 1px solid #8cc99d;
        padding: 1.1rem 1.2rem;
        border-radius: 8px;
        color: #173f24;
    }
    .soft-gap {
        height: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def execute_pipeline(location: str, weather_profile: str):
    """Run the same ingestion, prediction, routing, and planning chain used by the app."""
    profile_mapping = {
        "NORMAL": "normal",
        "HEAVY_RAIN": "heavy_rain",
        "HIGH_WINDS": "high_winds",
        "EXTREME_HEAT": "extreme_heat",
    }
    scenario = profile_mapping.get(weather_profile, "normal")

    with st.spinner("Running ingestion, forecast, routing, and planning layers..."):
        history_df = simulate_weather_ingestion(hours=168, scenario=scenario, seed=42)
        forecast_df = forecast_weather_metrics(history_df, forecast_hours=48)

        classifier = DisasterClassifier()
        ml_prediction = classifier.predict_disaster(forecast_df)

        news = fetch_news_context(location)
        weather = get_live_weather_summary(location)
        past_insights = load_insights()

        routing_result = run_cognitive_router(
            ml_prediction=ml_prediction,
            news_context=news,
            weather_context=weather,
            past_insights=past_insights,
        )

        target_dept = routing_result.get("target_department")
        justification = routing_result.get("justification")

        agent_mapping = {
            "Emergency Response": generate_emergency_response_plan,
            "Civil Defense": generate_civil_defense_plan,
            "Public Works": generate_public_works_plan,
        }
        planner = agent_mapping.get(target_dept)
        if not planner:
            planner = generate_emergency_response_plan

        payload = {
            "location": location,
            "disaster_type": ml_prediction["disaster_type"],
            "probability": ml_prediction["probability"],
            "severity": ml_prediction["severity"],
            "metrics": ml_prediction["metrics"],
            "news_context": news,
            "weather_context": weather,
        }
        drafted_plan = planner(payload, past_insights=past_insights)

        st.session_state.state = {
            "location": location,
            "weather_profile": weather_profile,
            "ml_prediction": ml_prediction,
            "news_context": news,
            "weather_context": weather,
            "target_department": target_dept,
            "routing_justification": justification,
            "drafted_plan": drafted_plan,
            "past_insights": past_insights,
        }
        st.session_state.show_hil_panel = True
        st.session_state.approved = False
        st.session_state.rejection_active = False


def abstract_feedback_to_rule(feedback: str, state: dict) -> str:
    """Convert operator feedback into the long-lived rule format consumed by memory."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return f"Rule: Always ensure that: {feedback}"

    prompt = (
        f"A human disaster controller has REJECTED the following drafted action plan.\n\n"
        f"Target Department: {state.get('target_department')}\n"
        f"Drafted Alert Message: {state.get('drafted_plan', {}).get('alert_message')}\n"
        f"Drafted Action Plan:\n"
        f"{json.dumps(state.get('drafted_plan', {}).get('action_plan'), indent=2)}\n\n"
        f"Human Critique/Feedback:\n"
        f"\"{feedback}\"\n\n"
        f"Identify the human's core concern. Formulate a single, concise, and generalized "
        f"engineering rule or guideline that starts with 'Rule: ' (e.g., 'Rule: Always specify shelter coordinates near central locations'). "
        f"Return ONLY the rule string."
    )

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text.strip()


if "state" not in st.session_state:
    st.session_state.state = None
if "show_hil_panel" not in st.session_state:
    st.session_state.show_hil_panel = False
if "approved" not in st.session_state:
    st.session_state.approved = False
if "rejection_active" not in st.session_state:
    st.session_state.rejection_active = False
if "feedback_input" not in st.session_state:
    st.session_state.feedback_input = ""


st.sidebar.title("Response Memory")
st.sidebar.caption("Permanent review rules loaded before every routing and planning cycle.")

insights = load_insights()
if not insights:
    st.sidebar.info("No review rules saved yet. Rejected plans will add new memory here.")
else:
    for idx, rule in enumerate(insights, 1):
        st.sidebar.markdown(f"**{idx}.** `{rule}`")

st.sidebar.markdown("---")
if st.sidebar.button("Clear Memory Rules", use_container_width=True):
    insights_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "memory", "insights.json"
    )
    if os.path.exists(insights_path):
        try:
            with open(insights_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            st.sidebar.success("Memory cleared.")
            st.rerun()
        except IOError as e:
            st.sidebar.error(f"Could not clear memory: {e}")


st.title("CrisisOps: Disaster Response Command Desk")
st.caption("Forecast simulation, cognitive routing, department planning, and review in one control surface.")

st.markdown("### Simulation Inputs")
st.markdown("<div class='control-strip'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    loc_input = st.text_input("Operational Area", value="Jaipur")
with col2:
    profile_input = st.selectbox(
        "Weather Pattern",
        options=["NORMAL", "HEAVY_RAIN", "HIGH_WINDS", "EXTREME_HEAT"],
    )
with col3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    trigger = st.button("Run Scenario", use_container_width=True, type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if trigger:
    execute_pipeline(loc_input, profile_input)
    st.rerun()


if st.session_state.state:
    state_data = st.session_state.state
    ml = state_data["ml_prediction"]

    st.markdown("### Forecast Classification")

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown(
            f"<div class='status-tile'><h4>Predicted Event</h4><h2>{ml['disaster_type']}</h2></div>",
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            f"<div class='status-tile'><h4>Confidence</h4><h2>{ml['probability']:.1%}</h2></div>",
            unsafe_allow_html=True,
        )
    with mcol3:
        st.markdown(
            f"<div class='status-tile'><h4>Severity Band</h4><h2>{ml['severity']}</h2></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='soft-gap'></div>", unsafe_allow_html=True)

    det_col1, det_col2 = st.columns(2)
    with det_col1:
        with st.expander("Weather Station Packet", expanded=False):
            st.text(state_data["weather_context"])
        with st.expander("Local Situation Feed", expanded=False):
            st.text(state_data["news_context"])

    with det_col2:
        routing_html = (
            f"<div class='handoff-panel'>"
            f"<h4>Department Handoff: {state_data['target_department']}</h4>"
            f"<p style='font-size:0.95rem; margin-top:0.5rem;'><strong>Reason:</strong> {state_data['routing_justification']}</p>"
            f"</div>"
        )
        st.markdown(routing_html, unsafe_allow_html=True)

    if st.session_state.show_hil_panel:
        st.markdown("### Operator Review")
        plan = state_data["drafted_plan"]

        with st.container(border=True):
            st.subheader(f"Draft Protocol: {state_data['target_department']}")

            st.markdown(
                f"<div class='dispatch-note'>BROADCAST MESSAGE<br/>{plan.get('alert_message')}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("**Action Sequence**")
            for idx, task in enumerate(plan.get("action_plan", []), 1):
                st.markdown(f"**{idx}.** {task}")

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(f"**Severity Check:** {plan.get('severity_verification')}")

            st.markdown("---")

            btn_col1, btn_col2, _ = st.columns([1, 1, 3])
            with btn_col1:
                approve_clicked = st.button("Approve Plan", use_container_width=True)
            with btn_col2:
                reject_clicked = st.button("Request Revision", use_container_width=True)

            if approve_clicked:
                st.session_state.approved = True
                st.session_state.show_hil_panel = False
                st.session_state.rejection_active = False
                st.rerun()

            if reject_clicked:
                st.session_state.rejection_active = True
                st.rerun()

            if st.session_state.rejection_active:
                st.markdown("#### Create Corrective Rule")
                critique_text = st.text_input("Operator feedback")

                if st.button("Save Rule And Rebuild Plan"):
                    if not critique_text.strip():
                        st.warning("Feedback cannot be empty.")
                    else:
                        with st.spinner("Saving review rule and regenerating plan..."):
                            new_rule = abstract_feedback_to_rule(critique_text, state_data)
                            save_insight(new_rule)
                            execute_pipeline(state_data["location"], state_data["weather_profile"])
                            st.success(f"Rule added: \"{new_rule}\". Plan regenerated.")
                            st.rerun()


if st.session_state.approved:
    st.success("Plan approved. Broadcast workflow completed.")
    st.balloons()
