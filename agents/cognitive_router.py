import os
import json
from typing import Literal, List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Local context and department-planning modules used by the standalone harness.
from tools.context_tools import fetch_news_context, get_live_weather_summary
from agents.department_agents import (
    generate_emergency_response_plan,
    generate_civil_defense_plan,
    generate_public_works_plan
)

class RoutingDecision(BaseModel):
    target_department: Literal["Emergency Response", "Civil Defense", "Public Works"] = Field(
        ..., 
        description="The primary department that must handle this phase of the disaster management operations."
    )
    justification: str = Field(
        ..., 
        description="A technical justification based on numerical forecast thresholds, news, and weather details."
    )

def run_cognitive_router(
    ml_prediction: dict, 
    news_context: str, 
    weather_context: str, 
    past_insights: List[str] = None
) -> dict:
    """
    Resolve which department should own the next response step.
    Gemini is used when configured; otherwise deterministic routing keeps demos runnable.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    # Reuse the same extracted values for the prompt and offline routing branch.
    disaster_type = ml_prediction.get('disaster_type', 'None')
    probability = ml_prediction.get('probability', 0.0)
    severity = ml_prediction.get('severity', 'LOW')
    metrics = ml_prediction.get('metrics', {})
    max_rain = metrics.get('max_rainfall_mm', 0.0)
    max_wind = metrics.get('max_wind_speed_kmh', 0.0)
    max_temp = metrics.get('max_temperature_c', 0.0)

    if not api_key:
        print("[INFO] GEMINI_API_KEY missing. Activating deterministic mock cognitive routing...")
        
        # Deterministic ownership rules for offline execution.
        if disaster_type == "Flood" and severity == "HIGH":
            target_department = "Emergency Response"
            justification = f"Mock Routing: High-severity Flood event with maximum rainfall {max_rain} mm poses an immediate danger to human life. Immediate search, rescue, and triage are critical."
        elif disaster_type == "Hurricane" or max_wind > 75.0:
            target_department = "Emergency Response"
            justification = f"Mock Routing: Extreme wind speeds ({max_wind} km/h) and severe rain will cause structural damages and debris, making life safety and rescue the immediate priority."
        elif disaster_type == "Heatwave" and max_temp > 40.0:
            target_department = "Civil Defense"
            justification = f"Mock Routing: Sustained extreme heat ({max_temp} °C) requires Civil Defense to set up cooling shelters, warnings, and water distribution grids."
        else:
            target_department = "Public Works"
            justification = f"Mock Routing: Weather indicators represent moderate structural or flooding risk. Directing to Public Works for drainage and road clearing."
            
        return {
            "target_department": target_department,
            "justification": justification
        }

    # Consolidated information packet for model-based routing.
    prompt = (
        f"Consolidated Disaster Information Packet:\n"
        f"1. ML Predictive Target: {disaster_type}\n"
        f"   - Predicted Probability: {probability:.2%}\n"
        f"   - Predicted Severity: {severity}\n"
        f"   - Forecast Rainfall (Max): {max_rain} mm\n"
        f"   - Forecast Wind (Max): {max_wind} km/h\n"
        f"   - Forecast Temp (Max): {max_temp} °C\n\n"
        f"2. Local News Context:\n"
        f"{news_context}\n\n"
        f"3. Weather Station Context:\n"
        f"{weather_context}\n"
    )
    
    if past_insights:
        prompt += f"\n4. Historical Reflection Memory Insights:\n"
        for insight in past_insights:
            prompt += f"  - {insight}\n"

    system_instruction = (
        "You are the Cognitive Routing Engine of the Autonomous Disaster Management System. "
        "Your objective is to analyze the consolidated disaster packet and route the operation to "
        "the most suitable specialized department.\n\n"
        "Routing Criteria:\n"
        "- 'Emergency Response': Select when there is an immediate, high-severity threat to human life requiring search and rescue, triage, and rapid deployment of emergency units.\n"
        "- 'Civil Defense': Select when the scenario requires large-scale public evacuation, sheltering logistics, food/water distribution, and mass population warning/safety coordination.\n"
        "- 'Public Works': Select when the threat is primarily infrastructural (drainage failure, dam overflow, road blockages, power grid collapse) and needs engineer coordination, debris clearing, and utility stabilization."
    )

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=RoutingDecision,
            temperature=0.1,
        )
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        return {
            "target_department": "Emergency Response",
            "justification": f"Fallback routing activated due to JSON parsing exception: {str(e)}"
        }

if __name__ == "__main__":
    print("=" * 70)
    print("CRISISOPS DISASTER SYSTEM - ROUTING HARNESS")
    print("=" * 70)
    
    # Step 1: prepare a representative ML output for a heavy-rain flood scenario.
    mock_ml_prediction = {
        "disaster_type": "Flood",
        "probability": 1.0000,
        "severity": "HIGH",
        "metrics": {
            "max_rainfall_mm": 137.07,
            "max_wind_speed_kmh": 41.63,
            "max_temperature_c": 29.62
        }
    }
    
    location = "Jaipur"
    print(f"Scenario Location: {location}")
    print(f"Mock ML Prediction: {mock_ml_prediction['disaster_type']} "
          f"(Prob: {mock_ml_prediction['probability']:.1%}, Severity: {mock_ml_prediction['severity']})")
    
    # Step 2: collect surrounding context from the tool layer.
    print("\nFetching real-time news and weather contexts...")
    news = fetch_news_context(location)
    weather = get_live_weather_summary(location)
    print("Contexts successfully retrieved.")
    
    # Step 3: resolve response ownership.
    print("\nExecuting Cognitive Router Decision...")
    routing_result = run_cognitive_router(
        ml_prediction=mock_ml_prediction,
        news_context=news,
        weather_context=weather
    )
    
    print("\n>>> ROUTING DECISION RECEIVED:")
    print(json.dumps(routing_result, indent=2))
    
    target_dept = routing_result.get("target_department")
    justification = routing_result.get("justification")
    
    # Step 4: hand the packet to the matching department planner.
    agent_mapping = {
        "Emergency Response": generate_emergency_response_plan,
        "Civil Defense": generate_civil_defense_plan,
        "Public Works": generate_public_works_plan
    }
    
    selected_agent_function = agent_mapping.get(target_dept)
    
    if selected_agent_function:
        print(f"\nHanding off scenario payload to: [{target_dept}] Agent...")
        
        # Compose the planner payload without altering classifier results.
        payload = {
            "location": location,
            "disaster_type": mock_ml_prediction["disaster_type"],
            "probability": mock_ml_prediction["probability"],
            "severity": mock_ml_prediction["severity"],
            "metrics": mock_ml_prediction["metrics"],
            "news_context": news,
            "weather_context": weather
        }
        
        plan_result = selected_agent_function(payload)
        
        print(f"\n>>> [{target_dept.upper()}] ACTION PLAN GENERATED:")
        print(json.dumps(plan_result, indent=2))
    else:
        print(f"\n[ERROR] Unknown target department returned: {target_dept}")
        
    print("=" * 70)
    print("Harness execution complete.")
