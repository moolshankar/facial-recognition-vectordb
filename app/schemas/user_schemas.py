from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    phone_number: str

class UserResponse(BaseModel):
    user_id: str
    name: str
    phone_number: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class FaceDetectionResponse(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None
    phone_number: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[list] = None