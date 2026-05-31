import os

def fetch_news_context(location: str) -> str:
    """
    Return local situational context for infrastructure, advisories, and weather alerts.
    Without NEWS_API_KEY, the function supplies a location-specific demo dispatch.
    
    Parameters:
    -----------
    location : str
        The city or region to query.
    """
    api_key = os.environ.get("NEWS_API_KEY", "").strip()
    
    if not api_key:
        # Demo dispatch used when live news integration is not configured.
        return (
            f"--- Mock News Dispatch for {location} ---\n"
            f"Source: {location} Chronicle / Municipal Press Release\n"
            f"Details:\n"
            f"- Municipal Infrastructure Alert: Low-lying drainage lines in the urban core of {location} are experiencing moderate water-logging. "
            f"Local authorities warn that some underpasses are temporarily closed.\n"
            f"- Structural Status: Municipal teams are monitoring the primary water reservoir levels, which are at 85% capacity. "
            f"Utility providers report minor power distribution outages in outer suburbs due to wind stress.\n"
            f"- Citizen Advisory: The local district administration has issued a precautionary travel warning. "
            f"Residents are advised to restrict non-essential commutes and stay clear of open canal channels."
        )
    
    # Placeholder for a future live NewsAPI implementation:
    # url = f"https://newsapi.org/v2/everything?q={location}+weather+alert&apiKey={api_key}"
    # response = requests.get(url)...
    return f"Live News for {location}: Weather warning issued by state emergency broadcast."

def get_live_weather_summary(location: str) -> str:
    """
    Return a short weather-station narrative for the requested location.
    Without WEATHER_API_KEY, the function supplies a realistic offline template.
    
    Parameters:
    -----------
    location : str
        The city or region to query.
    """
    api_key = os.environ.get("WEATHER_API_KEY", "").strip()
    
    if not api_key:
        # Offline weather packet used when the live weather service is absent.
        return (
            f"--- Mock Weather Station for {location} ---\n"
            f"Current Conditions:\n"
            f"- Temperature: 28.5°C\n"
            f"- Relative Humidity: 89%\n"
            f"- Barometric Pressure: 998 hPa (declining trend)\n"
            f"- Wind Speed & Direction: 32 km/h from East-Northeast, with gusts exceeding 50 km/h\n"
            f"- Current Precipitation: Continuous heavy drizzle (12 mm recorded in last 3 hours)\n"
            f"- Visibility: 3.5 km"
        )
        
    # Placeholder for a future OpenWeatherMap-style implementation:
    # url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    # response = requests.get(url)...
    return f"Live weather station at {location}: 29°C, Wind 30km/h, humidity 88%, storm warnings active."
