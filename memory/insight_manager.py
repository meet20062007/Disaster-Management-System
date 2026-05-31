import os
import json

# Keep the JSON store anchored to this package regardless of launch directory.
INSIGHTS_FILE = os.path.join(os.path.dirname(__file__), "insights.json")

def load_insights() -> list[str]:
    """
    Load saved operator review rules from memory/insights.json.
    Missing, empty, or invalid files are treated as no saved rules.
    """
    if not os.path.exists(INSIGHTS_FILE):
        return []
    
    try:
        with open(INSIGHTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            return []
    except (json.JSONDecodeError, IOError):
        return []

def save_insight(new_rule: str):
    """
    Persist a new review rule, creating the JSON store when needed.
    """
    # Start from the current rule set.
    rules = load_insights()
    
    # Preserve insertion order while avoiding duplicate entries.
    if new_rule not in rules:
        rules.append(new_rule)
        
    # Write the refreshed list back to disk.
    os.makedirs(os.path.dirname(INSIGHTS_FILE), exist_ok=True)
    try:
        with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[ERROR] Failed to save rule to {INSIGHTS_FILE}: {e}")
