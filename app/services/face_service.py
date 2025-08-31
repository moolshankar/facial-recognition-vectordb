import face_recognition
import numpy as np
import cv2
from typing import List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class FaceService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def detect_faces_opencv(self, image: np.ndarray) -> List[dict]:
        """Detect faces using OpenCV's built-in cascade classifier"""
        def _detect():
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return [
                {
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 1.0  # OpenCV doesn't provide confidence scores
                }
                for (x, y, w, h) in faces
            ]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _detect)
    
    async def detect_faces_face_recognition(self, image: np.ndarray) -> List[dict]:
        """Detect faces using face_recognition library"""
        def _detect():
            face_locations = face_recognition.face_locations(image, model="hog")  # Use HOG model for speed
            
            return [
                {
                    'bbox': [left, top, right, bottom],
                    'confidence': 1.0
                }
                for (top, right, bottom, left) in face_locations
            ]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _detect)
    
    async def extract_face_encodings(self, image: np.ndarray, face_locations: List[tuple]) -> List[np.ndarray]:
        """Extract face encodings using face_recognition"""
        def _extract():
            return face_recognition.face_encodings(image, face_locations, model="small")  # Use small model for speed
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _extract)
    
    async def process_uploaded_image(self, image_bytes: bytes) -> Tuple[Optional[np.ndarray], List[dict]]:
        """Process uploaded image and return encoding and face info"""
        def _process():
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces using face_recognition
            face_locations = face_recognition.face_locations(image_rgb, model="hog")
            if not face_locations:
                return None, []
            
            # Get encodings
            encodings = face_recognition.face_encodings(image_rgb, face_locations, model="small")
            if not encodings:
                return None, []
            
            # Return first face encoding and location info
            return encodings[0], [{'bbox': face_locations[0]}]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _process)
    
    def compare_faces(self, known_encoding: np.ndarray, face_encoding: np.ndarray, tolerance: float = 0.6) -> bool:
        """Compare two face encodings"""
        return face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)[0]
    
    def face_distance(self, known_encoding: np.ndarray, face_encoding: np.ndarray) -> float:
        """Calculate distance between face encodings"""
        return face_recognition.face_distance([known_encoding], face_encoding)[0]