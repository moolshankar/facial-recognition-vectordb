from sqlalchemy import Column, Integer, String, JSON, DateTime, Text
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database.connection import Base
import uuid

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    embedding = Column(Vector(128), nullable=False)  # face_recognition produces 128-dim vectors
    face_metadata = Column(JSON, nullable=True)  # Changed from 'metadata' to 'face_metadata'
    created_at = Column(DateTime(timezone=True), server_default=func.now())