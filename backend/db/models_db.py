from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from datetime import date as dt_date

class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str
    
    nickname: Optional[str] = "User"
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    bmi: Optional[float] = None

    stats: List["DailyStats"] = Relationship(back_populates="user")


class DailyStats(SQLModel, table=True):
    __tablename__ = "daily_stats"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    date: dt_date = Field(index=True)
    
    steps: int
    very_active_minutes: int
    minutesAsleep: int
    sleep_efficiency: int
    nremhr: float
    stress_score: int
    nightly_temperature: float
    resting_hr: float
    
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    
    user: Optional[User] = Relationship(back_populates="stats")