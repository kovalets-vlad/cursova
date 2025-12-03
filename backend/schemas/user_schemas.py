from pydantic import BaseModel
from typing import Optional

class UserProfileUpdate(BaseModel):
    age: Optional[int] = None     
    weight: Optional[float] = None 
    height: Optional[float] = None 