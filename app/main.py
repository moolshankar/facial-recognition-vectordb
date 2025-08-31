from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
import cv2
import asyncio
from typing import Generator
import numpy as np

from app.database.connection import get_session, create_tables
from app.database.repositories import UserRepository, FaceEmbeddingRepository
from app.services.face_service import FaceService
from app.services.detection_service import detection_service
from app.schemas.user_schemas import UserCreate, UserResponse
from app.config import settings

app = FastAPI(title="Facial Recognition App", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/static")

# Initialize services
face_service = FaceService()

@app.on_event("startup")
async def startup_event():
    await create_tables()
    print("Application started successfully!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)  
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/detection", response_class=HTMLResponse)
async def detection_page(request: Request):
    return templates.TemplateResponse("detection.html", {"request": request})

@app.post("/api/register", response_model=UserResponse)
async def register_user(
    name: str = Form(...),
    phone_number: str = Form(...),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session)
):
    try:
        # Process uploaded image
        image_bytes = await image.read()
        encoding, face_info = await face_service.process_uploaded_image(image_bytes)
        
        if encoding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Create user
        user_repo = UserRepository(session)
        user_data = UserCreate(name=name, phone_number=phone_number)
        user = await user_repo.create_user(user_data)
        
        # Store face embedding
        embedding_repo = FaceEmbeddingRepository(session)
        await embedding_repo.create_embedding(
            user_id=user.user_id,
            embedding=encoding,
            face_metadata={"face_info": face_info}
        )
        
        return UserResponse.model_validate(user)
        
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video_feed")
async def video_feed():
    async def generate_frames() -> Generator[bytes, None, None]:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open camera")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame through detection service
                processed_frame = await detection_service.process_frame(frame)
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.03)  # ~30 FPS
                
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)