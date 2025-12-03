from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from db.models_db import User
from security.security import verify_password, get_password_hash 

async def get_user_by_username(db: AsyncSession, username: str):
    result = await db.execute(select(User).filter(User.username == username))
    return result.scalars().first() 


async def authenticate_user(db: AsyncSession, username: str, password: str):
    user = await get_user_by_username(db, username) 
    if not user:
        return False 
    if not verify_password(password, user.hashed_password):
        return False 
    return user


async def create_user(db: AsyncSession, username: str, password: str):
    hashed_password = get_password_hash(password)
    
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    
    await db.commit()
    await db.refresh(new_user)
    return new_user