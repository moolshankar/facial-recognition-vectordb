from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:password123@localhost:5432/facial_recognition"
    secret_key: str = "your-secret-key-here"
    debug: bool = True
    
    # Face recognition settings
    face_recognition_tolerance: float = 0.6
    max_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    class Config:
        env_file = ".env"

settings = Settings()