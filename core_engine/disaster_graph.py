import os
import json
from typing import Dict, Any, List
from google import genai
from google.genai import types

# Weather simulation, forecast projection, and local classification layer.
from ml_layer.disaster_models import (
    simulate_weather_ingestion,
    forecast_weather_metrics,
    DisasterClassifier
)

# Context adapters for external or offline news/weather packets.
from tools.context_tools import fetch_news_context, get_live_weather_summary

# Durable review-rule memory.
from memory.insight_manager import load_insights, save_insight

# Department-specific planning agents.
from agents.department_agents import (
    generate_emergency_response_plan,
    generate_civil_defense_plan,
    generate_public_works_plan
)

# Ownership router for department handoff.
from agents.cognitive_router import run_cognitive_router

# Pipeline state container.
class SystemState:
    """
    Holds the evolving state for one disaster-response pipeline run.
    """
    def __init__(self, location: str):
        self.location: str = location
        self.ml_prediction: dict = {}
        self.news_context: str = ""
        self.weather_context: str = ""
        self.target_department: str = ""
        self.routing_justification: str = ""
        self.drafted_plan: dict = {}
        self.human_decision: str = "Pending"  # Pending, Approve, or Reject
        self.human_feedback: str = ""
        self.loop_count: int = 0
        self.past_insights: List[str] = []

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "ml_prediction": self.ml_prediction,
            "news_context": self.news_context,
            "weather_context": self.weather_context,
            "target_department": self.target_department,
            "routing_justification": self.routing_justification,
            "drafted_plan": self.drafted_plan,
            "human_decision": self.human_decision,
            "human_feedback": self.human_feedback,
            "loop_count": self.loop_count,
            "past_insights": self.past_insights
        }

# Pipeline node implementations.
def ingest_and_predict_node(state: SystemState) -> SystemState:
    """
    Generate weather inputs, classify the forecast, and attach context packets.
    The console harness keeps heavy_rain as its severe-event default.
    """
    print(f"\n--- [1] Executing Ingestion & ML Prediction Node (Pass {state.loop_count + 1}) ---")
    
    # Produce synthetic recent observations for the console harness.
    history_df = simulate_weather_ingestion(hours=168, scenario="heavy_rain", seed=42)
    forecast_df = forecast_weather_metrics(history_df, forecast_hours=48)
    
    # Classify the forecast horizon.
    classifier = DisasterClassifier()
    state.ml_prediction = classifier.predict_disaster(forecast_df)
    
    # Attach local context narratives.
    state.news_context = fetch_news_context(state.location)
    state.weather_context = get_live_weather_summary(state.location)
    
    print(f"ML Classifier predicted disaster type: {state.ml_prediction['disaster_type']} "
          f"with {state.ml_prediction['probability']:.1%} probability.")
    print("News and weather context blocks successfully fetched.")
    return state

def cognitive_routing_node(state: SystemState) -> SystemState:
    """
    Load review memory and route the incident to the owning department.
    """
    print(f"\n--- [2] Executing Cognitive Routing Node ---")
    
    # Pull long-lived review rules before routing.
    state.past_insights = load_insights()
    print(f"Loaded {len(state.past_insights)} permanent rules from reflection memory.")
    
    # Resolve departmental ownership.
    routing_result = run_cognitive_router(
        ml_prediction=state.ml_prediction,
        news_context=state.news_context,
        weather_context=state.weather_context,
        past_insights=state.past_insights
    )
    
    state.target_department = routing_result.get("target_department")
    state.routing_justification = routing_result.get("justification")
    
    print(f"Cognitive Router resolved ownership: [{state.target_department}]")
    print(f"Justification: {state.routing_justification}")
    return state

def department_planning_node(state: SystemState) -> SystemState:
    """
    Invoke the selected department planner with the current incident payload.
    """
    print(f"\n--- [3] Executing Department Planning Node ---")
    
    # Compose the planner payload from the current state.
    payload = {
        "location": state.location,
        "disaster_type": state.ml_prediction["disaster_type"],
        "probability": state.ml_prediction["probability"],
        "severity": state.ml_prediction["severity"],
        "metrics": state.ml_prediction["metrics"],
        "news_context": state.news_context,
        "weather_context": state.weather_context
    }
    
    # Map department ownership to its planner function.
    agent_mapping = {
        "Emergency Response": generate_emergency_response_plan,
        "Civil Defense": generate_civil_defense_plan,
        "Public Works": generate_public_works_plan
    }
    
    planner = agent_mapping.get(state.target_department)
    if not planner:
        raise ValueError(f"Unknown target department resolved: {state.target_department}")
        
    # Generate the structured plan consumed by review surfaces.
    state.drafted_plan = planner(payload, past_insights=state.past_insights)
    
    print("Action plan and citizen alerts successfully generated.")
    return state

def human_gatekeeper_node(state: SystemState) -> SystemState:
    """
    Console review checkpoint for approve/reject feedback.
    """
    print("\n" + "=" * 60)
    print(f"HUMAN-IN-THE-LOOP GATEKEEPER: REVIEW PLAN FOR {state.location.upper()}")
    print("=" * 60)
    print(f"Target Department: {state.target_department}")
    print(f"Routing Reason:    {state.routing_justification}")
    print("-" * 60)
    print(f"BROADCAST ALERT MESSAGE:\n  \"{state.drafted_plan.get('alert_message')}\"")
    print("\nACTION PLAN:")
    for idx, step in enumerate(state.drafted_plan.get("action_plan", []), 1):
        print(f"  {idx}. {step}")
    print(f"\nSeverity Verification:\n  {state.drafted_plan.get('severity_verification')}")
    print("=" * 60)
    
    # Keep the console loop active until the operator gives a valid decision.
    while True:
        choice = input("\nApprove Plan (A) or Reject & Provide Feedback (R)? ").strip().upper()
        if choice in ["A", "APPROVE"]:
            state.human_decision = "Approve"
            state.human_feedback = ""
            print("[INFO] Plan approved by human operator.")
            break
        elif choice in ["R", "REJECT"]:
            state.human_decision = "Reject"
            feedback = input("Enter your feedback/critique: ").strip()
            state.human_feedback = feedback
            print("[INFO] Plan rejected. Entering self-correction phase.")
            break
        else:
            print("[WARNING] Invalid choice. Please enter 'A' or 'R'.")
            
    return state

def self_improvement_node(state: SystemState) -> SystemState:
    """
    Convert rejected-plan feedback into a saved rule for future planning runs.
    """
    print(f"\n--- [5] Executing Self-Improvement Memory Node ---")
    
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    if not api_key:
        print("[INFO] GEMINI_API_KEY missing. Generating deterministic reflection rule locally...")
        # Offline mode stores the feedback in the same rule format.
        new_rule = f"Rule: Always ensure that: {state.human_feedback}"
    else:
        # Ask Gemini to abstract the critique into a reusable planning rule.
        print("Consulting Reflection Model to abstract critique into a rule...")
        prompt = (
            f"A human disaster controller has REJECTED the following drafted action plan.\n\n"
            f"Target Department: {state.target_department}\n"
            f"Drafted Alert Message: {state.drafted_plan.get('alert_message')}\n"
            f"Drafted Action Plan:\n"
            f"{json.dumps(state.drafted_plan.get('action_plan'), indent=2)}\n\n"
            f"Human Critique/Feedback:\n"
            f"\"{state.human_feedback}\"\n\n"
            f"Identify the human's core concern. Formulate a single, concise, and generalized "
            f"engineering rule or guideline that starts with 'Rule: ' (e.g., 'Rule: Always specify shelter coordinates near central locations'). "
            f"This rule will be fed back into the prompt in subsequent runs so the model never repeats this mistake. "
            f"Return ONLY the rule string."
        )
        
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )
        new_rule = response.text.strip()
        
    # Persist the new memory rule.
    save_insight(new_rule)
    print(f"\n>>> REFLECTION MEMORY UPDATED:")
    print(f"  Saved New Rule: \"{new_rule}\"")
    
    # Reset the gate so the pipeline can regenerate and return to review.
    state.human_decision = "Pending"
    state.loop_count += 1
    
    return state

# Core orchestration loop.
def run_disaster_system(location: str):
    """
    Run the pipeline repeatedly until the operator approves a generated plan.
    """
    # Start a fresh state object for the selected location.
    state = SystemState(location)
    
    while state.human_decision != "Approve":
        # Node 1: weather simulation and forecast classification.
        state = ingest_and_predict_node(state)
        
        # Node 2: rule-aware routing.
        state = cognitive_routing_node(state)
        
        # Node 3: department planning.
        state = department_planning_node(state)
        
        # Node 4: operator review.
        state = human_gatekeeper_node(state)
        
        # Node 5: self-correction, only when rejected.
        if state.human_decision == "Reject":
            state = self_improvement_node(state)
            
    print("\n" + "=" * 60)
    print(f"SUCCESS: System successfully exited after {state.loop_count} correction loop(s).")
    print(f"Final approved payload saved.")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 70)
    print("CRISISOPS DISASTER SYSTEM - RUNTIME ENGINE")
    print("=" * 70)
    
    # Reset memory for a clean console harness run.
    insights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory", "insights.json")
    if os.path.exists(insights_path):
        try:
            os.remove(insights_path)
            print("[INFO] Wiped memory/insights.json for a clean test run.")
        except IOError:
            pass
            
    test_location = "Jaipur"
    run_disaster_system(test_location)
