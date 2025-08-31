from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from app.database.models import User, FaceEmbedding
from app.schemas.user_schemas import UserCreate, UserResponse
from typing import List, Optional, Tuple
import numpy as np

class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_user(self, user_data: UserCreate) -> User:
        user = User(**user_data.model_dump())  # Changed from dict() to model_dump()
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.user_id == user_id)
        )
        return result.scalar_one_or_none()

class FaceEmbeddingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_embedding(self, user_id: str, embedding: np.ndarray, face_metadata: dict = None) -> FaceEmbedding:
        face_embedding = FaceEmbedding(
            user_id=user_id,
            embedding=embedding.tolist(),
            face_metadata=face_metadata  # Changed from metadata to face_metadata
        )
        self.session.add(face_embedding)
        await self.session.commit()
        await self.session.refresh(face_embedding)
        return face_embedding
    
    async def find_similar_faces(self, query_embedding: np.ndarray, threshold: float = 0.6, limit: int = 5) -> List[Tuple[str, float]]:
        # query_vector = query_embedding.tolist()
        query_embedding_str = self.numpy_to_pgvector(query_embedding)
        
        # Use cosine distance for similarity search
        query = text("""
            SELECT user_id, 1 - (embedding <=> :query_embedding) as similarity
            FROM face_embeddings
            WHERE 1 - (embedding <=> :query_embedding) > :threshold
            ORDER BY embedding <=> :query_embedding
            LIMIT :limit
        """)
        
        result = await self.session.execute(
            query,
            {
                "query_embedding": query_embedding_str,
                "threshold": threshold,
                "limit": limit
            }
        )
        
        return [(row.user_id, row.similarity) for row in result.fetchall()]
    
    async def get_all_embeddings_with_users(self) -> List[dict]:
        query = text("""
            SELECT fe.user_id, fe.embedding, u.name, u.phone_number
            FROM face_embeddings fe
            JOIN users u ON fe.user_id = u.user_id
        """)
        
        result = await self.session.execute(query)
        return [
            {
                "user_id": row.user_id,
                "embedding": row.embedding,
                "name": row.name,
                "phone_number": row.phone_number
            }
            for row in result.fetchall()
        ]

    def numpy_to_pgvector(self, embedding: np.ndarray) -> str:
        """Convert a NumPy array to a pgvector-compatible string."""
        return f"[{','.join(map(str, embedding))}]"