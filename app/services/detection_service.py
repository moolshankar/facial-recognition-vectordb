import cv2
import numpy as np
import asyncio
import face_recognition  # Make sure this import is present
from typing import Optional, List, Dict, Any
from app.services.face_service import FaceService
from app.services.cache_service import CacheService
from app.database.repositories import FaceEmbeddingRepository, UserRepository
from app.database.connection import async_session
from app.config import settings
import hashlib

class DetectionService:
    def __init__(self):
        self.face_service = FaceService()
        self.cache_service = CacheService(
            max_size=settings.max_cache_size,
            ttl_seconds=settings.cache_ttl_seconds
        )
        self._processing_lock = asyncio.Lock()
        self.detection_method = "hog"
    
    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return annotated frame"""
        # Create a hash of the frame for caching
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        
        # Check cache first
        cached_result = self.cache_service.get(frame_hash)
        if cached_result:
            return self._annotate_frame(frame, cached_result)
        
        # If not in cache, process asynchronously without blocking
        asyncio.create_task(self._process_and_cache_frame(frame, frame_hash))
        
        # Return original frame
        return frame
    
    async def _process_and_cache_frame(self, frame: np.ndarray, frame_hash: str) -> None:
        """Process frame and update cache asynchronously"""
        async with self._processing_lock:
            # Double-check cache to avoid duplicate processing
            if self.cache_service.get(frame_hash):
                return
            
            try:
                # Detect faces
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model=self.detection_method)
                # Extract encodings
                # encodings = await self.face_service.extract_face_encodings(rgb_frame, face_locations)
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Find matches in database
                matches = []
                async with async_session() as session:
                    embedding_repo = FaceEmbeddingRepository(session)
                    user_repo = UserRepository(session)
                    
                    for i, encoding in enumerate(encodings):
                        similar_faces = await embedding_repo.find_similar_faces(
                            encoding, 
                            threshold=settings.face_recognition_tolerance
                        )
                        
                        if similar_faces:
                            user_id, confidence = similar_faces[0]
                            print(f"Similar face found: {user_id}, confidence: {confidence}")
                            user = await user_repo.get_user_by_id(user_id)
                            if user:
                                matches.append({
                                    'bbox': face_locations[i],
                                    'name': user.name,
                                    'phone': user.phone_number,
                                    'confidence': confidence
                                })
                                print(f"Match added: {matches}")
                        else:
                            matches.append({
                                'bbox': face_locations[i],
                                'name': 'Unknown',
                                'phone': '',
                                'confidence': 0.0
                            })
                            print(f"Unknown Match added: {matches}")
                
                # Cache the results
                self.cache_service.set(frame_hash, matches)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                self.cache_service.set(frame_hash, [])
    
    def _annotate_frame(self, frame: np.ndarray, matches: List[Dict[str, Any]]) -> np.ndarray:
        """Annotate frame with face detection results"""
        annotated_frame = frame.copy()
        
        for match in matches:
            bbox = match['bbox']
            name = match['name']
            phone = match['phone']
            confidence = match['confidence']
            
            # Convert face_recognition bbox format (top, right, bottom, left) to OpenCV format
            top, right, bottom, left = bbox
            
            # Draw rectangle
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name}"
            if phone:
                label += f" | {phone}"
            if name != 'Unknown':
                label += f" | {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (left, top - label_height - 10), (left + label_width, top), color, -1)
            cv2.putText(annotated_frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame

# Global instance
detection_service = DetectionService()