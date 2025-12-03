from pydantic import BaseModel
from typing import List, Optional
from sqlmodel import Field
from datetime import date as dt_date

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

class DailyStatsRead(BaseModel):
    date: dt_date
    steps: Optional[int] = Field(None, schema_extra={"example": 8000})
    very_active_minutes: Optional[int] = Field(None, schema_extra={"example": 30})
    minutesAsleep: Optional[int] = Field(None, schema_extra={"example": 420})
    sleep_efficiency: Optional[int] = Field(None, ge=0, le=100, schema_extra={"example": 90})
    nremhr: Optional[float] = Field(None, schema_extra={"example": 60.5})
    stress_score: Optional[int] = Field(None, schema_extra={"example": 25})
    nightly_temperature: Optional[float] = Field(None, schema_extra={"example": 36.6})
    resting_hr: Optional[float] = Field(None, schema_extra={"example": 60.0})

class DailyStatsCreate(BaseModel):
    date: dt_date
    steps: Optional[int] = Field(None, schema_extra={"example": 8000})
    very_active_minutes: Optional[int] = Field(None, schema_extra={"example": 30})
    minutesAsleep: Optional[int] = Field(None, schema_extra={"example": 420})
    sleep_efficiency: Optional[int] = Field(None, ge=0, le=100, schema_extra={"example": 90})
    nremhr: Optional[float] = Field(None, schema_extra={"example": 60.5})
    stress_score: Optional[int] = Field(None, schema_extra={"example": 25})
    nightly_temperature: Optional[float] = Field(None, schema_extra={"example": 36.6})
    resting_hr: Optional[float] = Field(None, schema_extra={"example": 60.0})

class DailyStatsUpdate(BaseModel):
    steps: Optional[int] = None
    very_active_minutes: Optional[int] = None
    minutesAsleep: Optional[int] = None
    sleep_efficiency: Optional[int] = None
    nremhr: Optional[float] = None
    stress_score: Optional[int] = None
    nightly_temperature: Optional[float] = None
    resting_hr: Optional[float] = None

class PredictionRequest(BaseModel):
    history: List[DailyStats]

class HealthDataRequest(BaseModel):
    user_stats: dict 
    prediction_delta: float  
    predicted_bpm: float     