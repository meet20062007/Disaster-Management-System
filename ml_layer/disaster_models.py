import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def simulate_weather_ingestion(hours: int = 168, scenario: str = "normal", seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic block of recent weather telemetry for the selected scenario.
    
    Generates a Pandas DataFrame containing:
    - Timestamp: Hourly timestamps ending at the current time.
    - Rainfall: Precipitation in mm.
    - Wind_Speed: Wind speed in km/h.
    - Temperature: Temperature in °C.
    
    Parameters:
    -----------
    hours : int
        Number of historical hours to generate. Default is 168 (7 days).
    scenario : str
        The type of weather conditions to simulate. Options: "normal", "heavy_rain", "high_winds", "extreme_heat".
    seed : int
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    
    # Build an hourly timeline that ends at the current clock time.
    end_time = pd.Timestamp.now()
    timestamps = pd.date_range(end=end_time, periods=hours, freq="h")
    
    # Use a daily sinusoidal temperature wave, then shift the baseline per scenario.
    t_indices = np.arange(hours)
    diurnal_cycle = 8.0 * np.sin(2 * np.pi * t_indices / 24.0)
    
    if scenario == "extreme_heat":
        base_temp = 38.0
        temp_noise = np.random.normal(0, 1.0, hours)
        temperature = base_temp + diurnal_cycle + temp_noise
        
        # Heatwave profiles stay dry with comparatively low wind movement.
        rainfall = np.zeros(hours)
        wind_speed = np.random.uniform(5, 15, hours)
        
    elif scenario == "heavy_rain":
        base_temp = 20.0
        temp_noise = np.random.normal(0, 1.0, hours)
        temperature = base_temp + diurnal_cycle + temp_noise
        
        # Rain events use random showers with heavier bursts near the simulated crisis window.
        rain_prob = np.random.binomial(1, 0.4, hours)
        rain_base = np.random.exponential(15.0, hours)
        rainfall = rain_prob * rain_base
        # Push the last day higher to imitate an intensifying incoming event.
        rainfall[-24:] = rainfall[-24:] * 2.5
        
        wind_speed = np.random.uniform(10, 25, hours)
        
    elif scenario == "high_winds":
        base_temp = 24.0
        temp_noise = np.random.normal(0, 1.5, hours)
        temperature = base_temp + diurnal_cycle + temp_noise
        
        # Wind-driven storm profile: sustained strong gusts with meaningful rainfall.
        wind_base = np.random.normal(40.0, 10.0, hours)
        wind_base[-36:] = wind_base[-36:] + np.random.uniform(40, 80, 36)  # Escalation
        wind_speed = np.clip(wind_base, 10, 160)
        
        rain_prob = np.random.binomial(1, 0.5, hours)
        rainfall = rain_prob * np.random.exponential(25.0, hours)
        
    else:  # normal operating weather
        base_temp = 25.0
        temp_noise = np.random.normal(0, 1.2, hours)
        temperature = base_temp + diurnal_cycle + temp_noise
        
        # Ordinary weather keeps rain sparse and wind moderate.
        rain_prob = np.random.binomial(1, 0.08, hours)
        rainfall = rain_prob * np.random.exponential(5.0, hours)
        wind_speed = np.random.uniform(8, 28, hours)
        
    # Return the telemetry table expected by the forecast layer.
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Rainfall": np.round(np.clip(rainfall, 0, None), 2),
        "Wind_Speed": np.round(np.clip(wind_speed, 0, None), 2),
        "Temperature": np.round(temperature, 2)
    })
    
    return df

def forecast_weather_metrics(historical_df: pd.DataFrame, forecast_hours: int = 48) -> pd.DataFrame:
    """
    Project weather metrics for the upcoming forecast window.
    
    Uses rolling averages and trends extracted from the tail of the historical data
    combined with diurnal patterns and random perturbations.
    """
    last_row = historical_df.iloc[-1]
    last_timestamp = last_row["Timestamp"]
    
    # Create the future timestamp index directly after the last observed hour.
    forecast_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=forecast_hours, freq="h")
    
    # Anchor the projection to the most recent day of observations.
    history_tail_24 = historical_df.tail(24)
    mean_temp_trend = history_tail_24["Temperature"].mean()
    mean_wind_trend = history_tail_24["Wind_Speed"].mean()
    mean_rain_trend = history_tail_24["Rainfall"].mean()
    
    # Temperature keeps the recent baseline while preserving the day/night wave.
    t_indices = np.arange(forecast_hours)
    diurnal_cycle = 8.0 * np.sin(2 * np.pi * (t_indices + last_timestamp.hour) / 24.0)
    temp_noise = np.random.normal(0, 1.0, forecast_hours)
    forecasted_temp = mean_temp_trend + diurnal_cycle + temp_noise
    
    # Wind uses persistence toward the recent mean plus volatility.
    wind_std = max(history_tail_24["Wind_Speed"].std(), 2.0)
    wind_noise = np.random.normal(0, wind_std, forecast_hours)
    forecasted_wind = np.zeros(forecast_hours)
    current_wind = last_row["Wind_Speed"]
    for i in range(forecast_hours):
        # Blend the last state with the recent trend to imitate gradual drift.
        current_wind = 0.85 * current_wind + 0.15 * mean_wind_trend + wind_noise[i]
        forecasted_wind[i] = current_wind
    forecasted_wind = np.clip(forecasted_wind, 0, None)
    
    # Rainfall carries recent wetness forward with stochastic bursts.
    rain_scale = max(mean_rain_trend, 1.5)
    rain_noise = np.random.exponential(rain_scale, forecast_hours)
    forecasted_rain = np.zeros(forecast_hours)
    current_rain = last_row["Rainfall"]
    for i in range(forecast_hours):
        # Recent rain keeps stronger persistence; otherwise the signal decays quickly.
        decay_factor = 0.85 if current_rain > 0.0 else 0.3
        # Event probability follows the recent rainfall trend.
        rain_prob = 0.7 if mean_rain_trend > 2.0 else 0.1
        rain_event = np.random.binomial(1, rain_prob)
        
        current_rain = decay_factor * current_rain + rain_event * rain_noise[i]
        forecasted_rain[i] = current_rain
    forecasted_rain = np.clip(forecasted_rain, 0, None)
    
    forecast_df = pd.DataFrame({
        "Timestamp": forecast_timestamps,
        "Rainfall": np.round(forecasted_rain, 2),
        "Wind_Speed": np.round(forecasted_wind, 2),
        "Temperature": np.round(forecasted_temp, 2)
    })
    
    return forecast_df

class DisasterClassifier:
    """
    Lightweight RandomForest wrapper that classifies forecast aggregates into
    None, Flood, Hurricane, or Heatwave categories.
    """
    def __init__(self):
        self.classes = ["None", "Flood", "Hurricane", "Heatwave"]
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._train_model()
        
    def _train_model(self):
        """
        Build deterministic synthetic training rows and fit the local classifier.
        """
        # Keep the generated training set stable across runs.
        rng = np.random.RandomState(42)
        
        # Feature order: [max_rainfall, max_wind_speed, max_temp].
        # Labels: 0=None, 1=Flood, 2=Hurricane, 3=Heatwave.
        samples_per_class = 200
        
        # Class 0: normal weather bounds.
        rain_none = rng.uniform(0.0, 15.0, samples_per_class)
        wind_none = rng.uniform(5.0, 30.0, samples_per_class)
        temp_none = rng.uniform(15.0, 35.0, samples_per_class)
        X_none = np.column_stack((rain_none, wind_none, temp_none))
        y_none = np.zeros(samples_per_class)
        
        # Class 1: extreme rainfall with manageable wind and temperature.
        rain_flood = rng.uniform(50.0, 130.0, samples_per_class)
        wind_flood = rng.uniform(10.0, 45.0, samples_per_class)
        temp_flood = rng.uniform(15.0, 30.0, samples_per_class)
        X_flood = np.column_stack((rain_flood, wind_flood, temp_flood))
        y_flood = np.ones(samples_per_class)
        
        # Class 2: dangerous wind plus notable rainfall.
        rain_hurr = rng.uniform(30.0, 100.0, samples_per_class)
        wind_hurr = rng.uniform(70.0, 160.0, samples_per_class)
        temp_hurr = rng.uniform(18.0, 32.0, samples_per_class)
        X_hurr = np.column_stack((rain_hurr, wind_hurr, temp_hurr))
        y_hurr = np.ones(samples_per_class) * 2
        
        # Class 3: extreme heat with dry, low-wind conditions.
        rain_heat = rng.uniform(0.0, 5.0, samples_per_class)
        wind_heat = rng.uniform(5.0, 20.0, samples_per_class)
        temp_heat = rng.uniform(40.0, 55.0, samples_per_class)
        X_heat = np.column_stack((rain_heat, wind_heat, temp_heat))
        y_heat = np.ones(samples_per_class) * 3
        
        # Merge all class blocks into one training matrix.
        X = np.vstack((X_none, X_flood, X_hurr, X_heat))
        y = np.concatenate((y_none, y_flood, y_hurr, y_heat))
        
        # Shuffle without losing reproducibility.
        indices = np.arange(len(y))
        rng.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Fit the in-memory model used during every prediction.
        self.clf.fit(X, y)
        
    def predict_disaster(self, forecast_df: pd.DataFrame) -> dict:
        """
        Convert forecast rows into one disaster prediction with severity metadata.
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            48-hour hourly weather forecast with columns 'Rainfall', 'Wind_Speed', 'Temperature'.
            
        Returns:
        --------
        dict
            {"disaster_type": str, "probability": float, "severity": str}
        """
        # Use peak values over the forecast horizon as model features.
        max_rainfall = float(forecast_df["Rainfall"].max())
        max_wind = float(forecast_df["Wind_Speed"].max())
        max_temp = float(forecast_df["Temperature"].max())
        
        features = np.array([[max_rainfall, max_wind, max_temp]])
        
        # Ask the model for class probabilities.
        probabilities = self.clf.predict_proba(features)[0]
        
        # Select the class with the strongest model confidence.
        pred_class_idx = int(np.argmax(probabilities))
        disaster_type = self.classes[pred_class_idx]
        probability = float(probabilities[pred_class_idx])
        
        # Translate probability into the existing LOW/MEDIUM/HIGH severity band.
        if disaster_type == "None":
            severity = "LOW"
        else:
            if probability >= 0.75:
                severity = "HIGH"
            elif probability >= 0.40:
                severity = "MEDIUM"
            else:
                severity = "LOW"
                
        return {
            "disaster_type": disaster_type,
            "probability": round(probability, 4),
            "severity": severity,
            "metrics": {
                "max_rainfall_mm": round(max_rainfall, 2),
                "max_wind_speed_kmh": round(max_wind, 2),
                "max_temperature_c": round(max_temp, 2)
            }
        }

if __name__ == "__main__":
    print("=" * 60)
    print("AUTONOMOUS DISASTER MANAGEMENT SYSTEM - DATA & ML HARNESS")
    print("=" * 60)
    
    # Bring up the classifier for the standalone verification harness.
    print("Initializing and fitting DisasterClassifier...")
    classifier = DisasterClassifier()
    print("Classifier successfully fitted.\n")
    
    # Exercise each supported synthetic weather profile.
    scenarios = ["normal", "heavy_rain", "high_winds", "extreme_heat"]
    
    for scenario in scenarios:
        print(f"--- Simulating Weather Scenario: {scenario.upper()} ---")
        
        # Step 1: create recent synthetic observations.
        history_df = simulate_weather_ingestion(hours=168, scenario=scenario, seed=42)
        print(f"Ingested {len(history_df)} hours of historical data.")
        print(f"Historical stats:")
        print(f"  - Rainfall: max={history_df['Rainfall'].max()}mm, mean={history_df['Rainfall'].mean():.2f}mm")
        print(f"  - Wind Speed: max={history_df['Wind_Speed'].max()}km/h, mean={history_df['Wind_Speed'].mean():.2f}km/h")
        print(f"  - Temperature: max={history_df['Temperature'].max()}C, min={history_df['Temperature'].min()}C")
        
        # Step 2: project the next forecast window.
        forecast_df = forecast_weather_metrics(history_df, forecast_hours=48)
        print(f"Generated {len(forecast_df)} hours of forecasted weather.")
        
        # Step 3: classify the forecast aggregate.
        prediction = classifier.predict_disaster(forecast_df)
        print("\nFinal Classifier Prediction:")
        print(f"  Disaster Type: {prediction['disaster_type']}")
        print(f"  Probability:   {prediction['probability']:.4%}")
        print(f"  Severity:      {prediction['severity']}")
        print(f"  Aggregated Forecast Metrics:")
        print(f"    - Max Rainfall:    {prediction['metrics']['max_rainfall_mm']} mm")
        print(f"    - Max Wind Speed:  {prediction['metrics']['max_wind_speed_kmh']} km/h")
        print(f"    - Max Temperature: {prediction['metrics']['max_temperature_c']} C")
        print("-" * 60)
    
    print("Verification execution complete.")
