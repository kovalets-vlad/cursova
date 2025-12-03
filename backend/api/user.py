from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated
from sqlalchemy import select  # <--- Не забудь імпортувати select
from schemas.user_schemas import UserProfileUpdate
from db.dependencies import get_current_user
from db.models_db import User
from db.session import AsyncSessionDep

router = APIRouter()

@router.patch("/user/update")
async def update_user_profile(
    profile_data: UserProfileUpdate, 
    current_user: Annotated[User, Depends(get_current_user)], 
    session: AsyncSessionDep
):
    print("Updating profile for user:", current_user)
    query = select(User).where(User.id == current_user.id)
    result = await session.execute(query)
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if profile_data.age is not None:
        user.age = profile_data.age

    current_weight = profile_data.weight if profile_data.weight is not None else user.weight
    current_height = profile_data.height if profile_data.height is not None else user.height

    if profile_data.weight is not None:
        user.weight = profile_data.weight
        
        if current_height and current_height > 0:
            height_m = current_height / 100.0
            new_bmi = current_weight / (height_m ** 2)
            user.bmi = round(new_bmi, 2)

    if profile_data.height is not None:
        user.height = profile_data.height
        if current_weight and current_weight > 0:
            height_m = current_height / 100.0
            new_bmi = current_weight / (height_m ** 2)
            user.bmi = round(new_bmi, 2)

    await session.commit()
    await session.refresh(user)

    return {
        "message": "Profile updated successfully",
        "user": {
            "id": user.id,
            "age": user.age,
            "weight": user.weight,
            "height": user.height,
            "bmi": user.bmi
        }
    }