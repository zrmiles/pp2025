# License Plate Detection Service

This service processes video streams, detects Russian license plates, and stores the data in PostgreSQL.

## Features

- Video stream processing (RTSP, HTTP, WebRTC)
- License plate detection using OpenCV and EasyOCR
- PostgreSQL database integration
- REST API interface
- Logging and monitoring

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following variables:
```
DATABASE_URL=postgresql://user:password@localhost:5432/license_plates
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- POST /detect - Upload video file or stream
- GET /plates - Get list of detected license plates

## Database Schema

```sql
CREATE TABLE detected_plates (
    id SERIAL PRIMARY KEY,
    plate_number VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    image_url TEXT,
    video_source VARCHAR(255)
);
``` 