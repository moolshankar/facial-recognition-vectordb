# Facial Recognition Application using VectorDB and Caching

A FastAPI-based facial recognition system that allows user registration with face images and real-time face detection with identification.

## Features

- User registration with face image capture
- Real-time face detection and recognition via webcam
- PostgreSQL with pgvector for efficient similarity search
- Async processing with caching for optimal performance
- RetinaFace for face detection and face_recognition for encodings

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- macOS with camera access
- Sufficient disk space for ML model downloads
- Xcode installation from MacOS app store for dlib support

## Setup Instructions for Mac Silicon

### 1. Clone Repository
```bash
git clone https://github.com/moolshankar/facial-recognition-vectordb.git
cd facial_recognition_app
sh setup.sh
sh install.sh
sh db-start.sh
sh start.sh


http://localhost:8000/


