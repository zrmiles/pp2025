from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
import numpy as np
from typing import List
import os
from datetime import datetime
import logging
from .services.plate_detector import PlateDetector
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.services.plate_detector import process_image_with_ocr
import re
from sqlalchemy.orm import Session
from app.core.database import get_db, Base, engine
from app.models.database import DetectedPlate
import itertools
from itertools import permutations

# Инициализация базы данных при запуске
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="License Plate Detection Service",
    description="Service for detecting and recognizing Russian license plates from video streams",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="app/static"), name="static")


logger.add("logs/app.log", rotation="500 MB")

plate_detector = PlateDetector()


UPLOAD_DIR = "uploads"
PLATES_DIR = "plates"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLATES_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLATE_PATTERN = re.compile(r'^[АВЕКМНОРСТУХ]{1}\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$')

@app.post("/detect")
async def detect_plates(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Обработка загруженного видеофайла с сохранением в БД"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        detected_plates = []
        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 5 == 0:
                result = plate_detector.process_video_frame(frame)
                if result:
                    plate_text, confidence, processed_frame = result
                    
                    frame_path = os.path.join(
                        PLATES_DIR, 
                        f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    )
                    cv2.imwrite(frame_path, processed_frame)
                    
                    # Сохраняем в базу данных
                    plate_record = DetectedPlate(
                        plate_number=plate_text,
                        confidence=confidence,
                        image_url=frame_path,
                        video_source=file.filename,
                        timestamp=datetime.utcnow()
                    )
                    db.add(plate_record)
                    db.commit()
                    
                    detected_plates.append({
                        "plate_number": plate_text,
                        "confidence": confidence,
                        "frame_url": frame_path,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    processed_frames += 1

            frame_count += 1

        cap.release()
        logger.info(f"Processed {frame_count} frames, found {processed_frames} plates")
        return JSONResponse(content={
            "message": "Video processed successfully",
            "total_frames": frame_count,
            "detected_plates": detected_plates
        })

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plates")
def get_plates(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Получение списка обнаруженных номеров из базы данных"""
    plates = db.query(DetectedPlate).order_by(DetectedPlate.timestamp.desc()).offset(skip).limit(limit).all()
    return plates

@app.get("/")
async def root():
    """Возвращаем HTML-страницу для загрузки фотографий"""
    return FileResponse("app/static/index.html")

def try_combine_blocks(blocks):
    n = len(blocks)
    for i in range(n):
        for j in range(i+1, n+1):
            candidate = ''.join(blocks[i:j])
            candidate = ''.join([c for c in candidate if c in 'АВЕКМНОРСТУХ0123456789'])
            if PLATE_PATTERN.match(candidate):
                return candidate
    return None

def find_russian_plate(blocks):
    for i, b1 in enumerate(blocks):
        if len(b1) == 1 and b1 in 'АВЕКМНОРСТУХ':
            for j, b2 in enumerate(blocks):
                if j == i: continue
                if len(b2) == 3 and b2.isdigit():
                    for k, b3 in enumerate(blocks):
                        if k in [i, j]: continue
                        if len(b3) == 2 and all(c in 'АВЕКМНОРСТУХ' for c in b3):
                            for m, b4 in enumerate(blocks):
                                if m in [i, j, k]: continue
                                if len(b4) in [2, 3] and b4.isdigit():
                                    candidate = b1 + b2 + b3 + b4
                                    if PLATE_PATTERN.match(candidate):
                                        return candidate
    return None

def find_russian_plate_any_order(blocks):
    blocks = [b for b in blocks if b and b not in ['RUS', 'RUS.']]
    for combo in itertools.permutations(blocks, 4):
        candidate = ''.join(combo)
        candidate = ''.join([c for c in candidate if c in 'АВЕКМНОРСТУХ0123456789'])
        if PLATE_PATTERN.match(candidate):
            return candidate
    return None

def to_cyrillic(text):
    mapping = {
        'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М', 'H': 'Н',
        'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т', 'Y': 'У', 'X': 'Х'
    }
    return ''.join([mapping.get(c, c) for c in text])

@app.post("/detect/image")
async def detect_plate_image(
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Распознавание номера на фотографии с сохранением в БД"""
    try:
        if not image_file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Поддерживаются только форматы JPG, JPEG и PNG")

        content = await image_file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

        plates = process_image_with_ocr(image)
        detected_plates = []
        all_texts = []
        
        # Обработка результатов OCR
        for text, conf in plates:
            cleaned = text.replace(' ', '').upper()
            cleaned = to_cyrillic(cleaned)
            all_texts.append({"raw": text, "cleaned": cleaned, "confidence": conf})
            logger.info(f"Распознанный текст: {text}, уверенность: {conf}")

        blocks = [t['cleaned'] for t in all_texts if t['cleaned']]
        has_rus = any(b in ['RUS', 'RUS.'] for b in blocks)

        # Улучшенное распознавание номеров
        plate_candidates = []
        
        # Пробуем найти номер в правильном порядке
        for i in range(len(blocks)):
            for j in range(i+1, len(blocks)):
                for k in range(j+1, len(blocks)):
                    for m in range(k+1, len(blocks)):
                        candidate = blocks[i] + blocks[j] + blocks[k] + blocks[m]
                        if PLATE_PATTERN.match(candidate):
                            plate_candidates.append(candidate)
        
        # Если не нашли в правильном порядке, пробуем все перестановки
        if not plate_candidates:
            for combo in itertools.permutations(blocks, 4):
                candidate = ''.join(combo)
                if PLATE_PATTERN.match(candidate):
                    plate_candidates.append(candidate)

        # Сохраняем найденные номера
        for candidate in plate_candidates:
            if has_rus:
                candidate = f"{candidate} RUS"
            
            # Сохраняем изображение
            frame_path = os.path.join(
                PLATES_DIR,
                f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(frame_path, image)
            
            # Сохраняем в базу данных
            try:
                plate_record = DetectedPlate(
                    plate_number=candidate,
                    confidence=min([t['confidence'] for t in all_texts]) if all_texts else 0,
                    image_url=frame_path,
                    timestamp=datetime.utcnow()
                )
                db.add(plate_record)
                db.commit()
                logger.info(f"Номер {candidate} успешно сохранен в базу данных")
            except Exception as db_error:
                logger.error(f"Ошибка при сохранении в базу данных: {str(db_error)}")
                db.rollback()
            
            detected_plates.append({
                "plate_number": candidate,
                "confidence": min([t['confidence'] for t in all_texts]) if all_texts else 0
            })

        if not detected_plates:
            logger.warning("Номера не обнаружены на изображении")
            return {
                "detected_plates": [],
                "all_ocr_results": all_texts,
                "message": "Номера не обнаружены"
            }

        return {
            "detected_plates": detected_plates,
            "all_ocr_results": all_texts,
            "message": f"Найдено номеров: {len(detected_plates)}"
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 