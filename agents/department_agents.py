import os
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class DepartmentPlan(BaseModel):
    action_plan: list[str] = Field(
        ..., 
        description="Chronological, actionable steps for the teams in the field to execute immediately."
    )
    alert_message: str = Field(
        ..., 
        description="Official alert broadcast message to be sent to citizens via sirens, SMS, or radio."
    )
    severity_verification: str = Field(
        ..., 
        description="A validation of the predicted disaster severity (HIGH/MEDIUM/LOW) with context-based reasoning."
    )

def _call_gemini_agent(system_instruction: str, data: dict, past_insights: list[str] = None) -> dict:
    """
    Prepare the department payload, call Gemini when configured, and normalize the
    response into the DepartmentPlan schema. Offline mode returns a structured demo plan.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    # Pull shared values used by both the prompt and offline fallback branch.
    location = data.get('location', 'Unknown')
    disaster_type = data.get('disaster_type', 'N/A')
    probability = data.get('probability', 0.0)
    severity = data.get('severity', 'N/A')
    metrics = data.get('metrics', {})
    max_rain = metrics.get('max_rainfall_mm', 0.0)
    max_wind = metrics.get('max_wind_speed_kmh', 0.0)
    max_temp = metrics.get('max_temperature_c', 0.0)
    news_context = data.get('news_context', 'No news context available')
    weather_context = data.get('weather_context', 'No weather context available')

    if not api_key:
        # Offline structured response used for local demos and CI-style runs.
        print(f"[INFO] GEMINI_API_KEY missing. Generating mock plan for {location} [{disaster_type}]...")
        
        if "Emergency Response" in system_instruction or "Chief Officer of Emergency Response" in system_instruction:
            res = {
                "action_plan": [
                    f"Deploy search and rescue teams (boats, water rescue crafts) to flooded sectors of {location}.",
                    f"Establish emergency medical triage stations at designated high points.",
                    f"Mobilize rescue helicopters and transport teams for stranded individuals."
                ],
                "alert_message": f"EMERGENCY BROADCAST: Flash flooding has reached critical thresholds in {location}. Search & rescue operations are active. Stay indoors or move to high ground immediately.",
                "severity_verification": f"HIGH - Rainfall of {max_rain} mm has overloaded primary drainage infrastructure in {location}, confirming immediate life safety threat."
            }
        elif "Civil Defense" in system_instruction or "Director of Civil Defense" in system_instruction:
            res = {
                "action_plan": [
                    f"Establish evacuation logistics and open municipal shelters across {location}.",
                    "Activate regional warning sirens and dispatch public transport buses to low-lying zones.",
                    "Distribute emergency food rations, blankets, and clean drinking water to active shelters."
                ],
                "alert_message": f"CIVIL WARNING: Low-lying neighborhoods in {location} must evacuate immediately. Emergency shelters are now open at designated city halls and schools.",
                "severity_verification": f"HIGH - Severe weather conditions ({max_rain} mm rainfall, {max_wind} km/h winds) require regional evacuation and population coordination."
            }
        else:  # Public Works
            res = {
                "action_plan": [
                    f"Deploy municipal debris clearing machinery to restore primary roadway corridors in {location}.",
                    f"Activate emergency storm-water pumps at maximum output.",
                    "Coordinate with electricity operators to safely isolate flooded power distribution blocks."
                ],
                "alert_message": f"PUBLIC SAFETY NOTICE: Municipal teams in {location} are working to clear road blockages. Avoid travel due to debris and high-voltage grid risks.",
                "severity_verification": f"HIGH - Storm accumulation of {max_rain} mm rain and {max_wind} km/h wind exceeds standard drainage capacities. Restoring transport access is critical."
            }

        # Reflect saved review rules in the offline plan so the feedback loop is visible.
        if past_insights:
            for insight in past_insights:
                res["action_plan"].append(f"[Reflective Adaptation] {insight}")
                res["alert_message"] += f" (Adhering to: {insight})"
                res["severity_verification"] += f" [Verified rule: {insight}]"
                
        return res

    # Build the model prompt, including saved review rules when present.
    prompt = (
        f"Analyze the following disaster scenario data:\n"
        f"Location: {location}\n"
        f"Disaster Classification: {disaster_type}\n"
        f"ML Prediction Probability: {probability:.2%}\n"
        f"Assigned Severity: {severity}\n"
        f"Aggregated Forecast Metrics (Next 48 Hours):\n"
        f"  - Max Rainfall: {max_rain} mm\n"
        f"  - Max Wind Speed: {max_wind} km/h\n"
        f"  - Max Temperature: {max_temp} °C\n\n"
        f"Local News Context:\n"
        f"{news_context}\n\n"
        f"Current Weather Station Context:\n"
        f"{weather_context}\n"
    )

    if past_insights:
        prompt += f"\nCritical Instruction: You MUST strictly incorporate the following reflection rules into your planning and alerts:\n"
        for rule in past_insights:
            prompt += f"- {rule}\n"

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=DepartmentPlan,
            temperature=0.2,
        )
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        return {
            "action_plan": [
                f"Deploy local emergency response scouting teams to {location}.",
                "Establish core command communications.",
                f"Respond to {disaster_type} emergency."
            ],
            "alert_message": f"ALERT: A weather emergency is unfolding in {location}. Monitor local updates.",
            "severity_verification": f"Fallback activated due to parsing error: {str(e)}"
        }

def generate_emergency_response_plan(data: dict, past_insights: list[str] = None) -> dict:
    """
    Draft the first-responder plan for immediate rescue, triage, and medical dispatch.
    """
    system_instruction = (
        "You are the Chief Officer of Emergency Response. Your persona is a decisive, "
        "first-responder tactical lead. You prioritize immediate life-saving search and rescue operations, "
        "casualty mitigation, emergency medical coordination, triage setups, and deploying active responder "
        "teams (boats, helicopters, crews) directly to hotspots. Keep your action plan highly actionable, "
        "tactical, and focused on the first 0-24 hours."
    )
    return _call_gemini_agent(system_instruction, data, past_insights)

def generate_civil_defense_plan(data: dict, past_insights: list[str] = None) -> dict:
    """
    Draft the civil-defense plan for evacuation, sheltering, and population support.
    """
    system_instruction = (
        "You are the Director of Civil Defense. Your persona is a regional civil strategist. "
        "You focus on evacuation zones, coordinating emergency shelters (schools, community halls), "
        "managing resources (food, blankets, clean drinking water, medicines), and organizing public transport "
        "or evacuation routes. Your alerts should be clear, calming, and detailed about where citizens must go."
    )
    return _call_gemini_agent(system_instruction, data, past_insights)

def generate_public_works_plan(data: dict, past_insights: list[str] = None) -> dict:
    """
    Draft the municipal works plan for transport, drainage, utilities, and asset safety.
    """
    system_instruction = (
        "You are the Commissioner of Public Works. Your persona is an experienced municipal infrastructure engineer. "
        "You prioritize road debris removal to reopen transit corridors, tracking bridge and dam stability under high stress, "
        "working with utility companies to isolate or restore power grids and water supplies, and managing critical "
        "municipal assets (like drainage gates or pumping stations). Your plan is highly engineering-focused."
    )
    return _call_gemini_agent(system_instruction, data, past_insights)
