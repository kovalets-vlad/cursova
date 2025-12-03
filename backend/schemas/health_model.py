from pydantic import BaseModel
from typing import List

class HealthDataRequest(BaseModel):
    user_stats: dict
    prediction_delta: int
    predicted_bpm: int

class DailyStats(BaseModel):
    steps: int
    very_active_minutes: int
    minutesAsleep: int
    sleep_efficiency: int
    nremhr: float
    stress_score: int
    nightly_temperature: float
    resting_hr: float
    age: int
    bmi: float
    is_weekend: int

class PredictionRequest(BaseModel):
    history: List[DailyStats]