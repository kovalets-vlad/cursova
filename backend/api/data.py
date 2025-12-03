from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from sqlalchemy import select
from datetime import date as dt_date
from db.session import AsyncSessionDep
from db.models_db import DailyStats, User  
from db.dependencies import get_current_user
from schemas.health_model import DailyStatsCreate, DailyStatsUpdate

router = APIRouter()

@router.post("/manual_entry")
async def create_manual_stats(
    stats_in: DailyStatsCreate,
    session: AsyncSessionDep,
    current_user: User = Depends(get_current_user)
):

    query = select(DailyStats).where(
        DailyStats.user_id == current_user.id,
        DailyStats.date == stats_in.date
    )
    existing_stat = await session.execute(query)
    if existing_stat.scalars().first():
        raise HTTPException(status_code=400, detail="Stats for this date already exist")

    is_weekend_val = 1 if stats_in.date.weekday() >= 5 else 0

    user_bmi = 0.0
    if current_user.weight and current_user.height:
        height_m = current_user.height / 100 
        user_bmi = round(current_user.weight / (height_m ** 2), 2)

    user_age = 0
    if current_user.age:
        user_age = current_user.age
    new_stats = DailyStats(
        user_id=current_user.id,
        date=stats_in.date,
        
        steps=stats_in.steps if stats_in.steps is not None else 5000,
        very_active_minutes=stats_in.very_active_minutes if stats_in.very_active_minutes is not None else 0,
        minutesAsleep=stats_in.minutesAsleep if stats_in.minutesAsleep is not None else 420, 
        sleep_efficiency=stats_in.sleep_efficiency if stats_in.sleep_efficiency is not None else 85,
        nremhr=stats_in.nremhr if stats_in.nremhr is not None else 65.0,
        stress_score=stats_in.stress_score if stats_in.stress_score is not None else 50,
        nightly_temperature=stats_in.nightly_temperature if stats_in.nightly_temperature is not None else 36.6,
        resting_hr=stats_in.resting_hr if stats_in.resting_hr is not None else 65.0,

        age=user_age,
        bmi=user_bmi,
        is_weekend=is_weekend_val
    )

    session.add(new_stats)
    await session.commit()
    await session.refresh(new_stats)
    
    return {"message": "Stats created successfully", "data": new_stats}

@router.get("/", response_model=List[DailyStatsCreate]) # Або створи окрему схему Response
async def get_all_stats(
    session: AsyncSessionDep,
    current_user: User = Depends(get_current_user),
    limit: int = 30, # Обмежимо, наприклад, останніми 30 записами
    offset: int = 0
):
    query = select(DailyStats).where(
        DailyStats.user_id == current_user.id
    ).order_by(DailyStats.date.desc()).limit(limit).offset(offset)
    
    result = await session.execute(query)
    return result.scalars().all()


# --- READ ONE (Отримати запис за конкретну дату) ---
@router.get("/{stat_date}")
async def get_stat_by_date(
    stat_date: dt_date,
    session: AsyncSessionDep,
    current_user: User = Depends(get_current_user)
):
    query = select(DailyStats).where(
        DailyStats.user_id == current_user.id,
        DailyStats.date == stat_date
    )
    result = await session.execute(query)
    stat = result.scalars().first()
    
    if not stat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Stats for this date not found"
        )
    return stat


# --- UPDATE (Оновити дані за конкретну дату) ---
@router.patch("/{stat_date}")
async def update_stat(
    stat_date: dt_date,
    stats_update: DailyStatsUpdate,
    session: AsyncSessionDep,
    current_user: User = Depends(get_current_user)
):
    # 1. Знаходимо запис
    query = select(DailyStats).where(
        DailyStats.user_id == current_user.id,
        DailyStats.date == stat_date
    )
    result = await session.execute(query)
    stat = result.scalars().first()

    if not stat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Stats for this date not found"
        )

    # 2. Оновлюємо тільки ті поля, які прийшли (exclude_unset=True)
    update_data = stats_update.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(stat, key, value)

    await session.commit()
    await session.refresh(stat)
    return stat


# --- DELETE (Видалити запис за конкретну дату) ---
@router.delete("/{stat_date}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stat(
    stat_date: dt_date,
    session: AsyncSessionDep,
    current_user: User = Depends(get_current_user)
):
    # 1. Перевіряємо, чи існує запис (можна одразу delete, але краще перевірити для 404)
    query = select(DailyStats).where(
        DailyStats.user_id == current_user.id,
        DailyStats.date == stat_date
    )
    result = await session.execute(query)
    stat = result.scalars().first()

    if not stat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Stats for this date not found"
        )

    await session.delete(stat)
    await session.commit()
    
    return None