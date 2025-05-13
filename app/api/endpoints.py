from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import cv2
import numpy as np
from ..core.database import get_db
from ..models.database import DetectedPlate
from ..services.plate_detector import PlateDetector
from loguru import logger
import os
from pydantic import BaseModel

router = APIRouter()
plate_detector = PlateDetector()

class PlateResponse(BaseModel):
    plate_number: str
    confidence: float
    timestamp: datetime
    image_url: Optional[str] = None
    video_source: Optional[str] = None

@router.post("/detect", response_model=List[PlateResponse])
async def detect_plates(
    video_file: UploadFile = File(...),
    process_every_n_frames: int = Query(5, ge=1, le=30),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """
    Обработка загруженного видеофайла для обнаружения номерных знаков
    
    - **video_file**: Видеофайл для обработки
    - **process_every_n_frames**: Обрабатывать каждый N-й кадр (для оптимизации)
    - **min_confidence**: Минимальный порог уверенности для распознавания
    """
    if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только форматы MP4, AVI и MKV"
        )

    try:
        # Создаем временную директорию, если её нет
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Сохраняем временный файл
        temp_path = os.path.join(temp_dir, f"temp_{datetime.now().timestamp()}.mp4")
        with open(temp_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Обрабатываем видео
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Не удалось открыть видеофайл")

        detected_plates = []
        frame_count = 0
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_every_n_frames == 0:
                result = plate_detector.process_video_frame(frame)
                if result:
                    plate_text, confidence, processed_frame = result
                    
                    if confidence >= min_confidence:
                        # Сохраняем кадр с обнаруженным номером
                        frame_path = os.path.join(
                            "plates",
                            f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        )
                        os.makedirs("plates", exist_ok=True)
                        cv2.imwrite(frame_path, processed_frame)
                        
                        # Сохраняем в базу данных
                        plate_record = DetectedPlate(
                            plate_number=plate_text,
                            confidence=confidence,
                            image_url=frame_path,
                            video_source=video_file.filename,
                            timestamp=datetime.utcnow()
                        )
                        db.add(plate_record)
                        db.commit()
                        
                        detected_plates.append(PlateResponse(
                            plate_number=plate_text,
                            confidence=confidence,
                            timestamp=datetime.utcnow(),
                            image_url=frame_path,
                            video_source=video_file.filename
                        ))
                        processed_frames += 1

            frame_count += 1

        cap.release()
        os.remove(temp_path)
        
        logger.info(f"Обработано {frame_count} кадров, найдено {processed_frames} номеров")
        
        return detected_plates
        
    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plates", response_model=List[PlateResponse])
async def get_plates(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    plate_number: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Получение списка обнаруженных номеров с возможностью фильтрации
    
    - **skip**: Количество записей для пропуска
    - **limit**: Максимальное количество записей
    - **plate_number**: Фильтр по номеру
    - **start_date**: Начальная дата для фильтрации
    - **end_date**: Конечная дата для фильтрации
    """
    query = db.query(DetectedPlate)
    
    if plate_number:
        query = query.filter(DetectedPlate.plate_number.ilike(f"%{plate_number}%"))
    if start_date:
        query = query.filter(DetectedPlate.timestamp >= start_date)
    if end_date:
        query = query.filter(DetectedPlate.timestamp <= end_date)
        
    plates = query.order_by(DetectedPlate.timestamp.desc()).offset(skip).limit(limit).all()
    return plates

@router.post("/detect/image", response_model=List[PlateResponse])
async def detect_plates_from_image(
    image_file: UploadFile = File(...),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """
    Обработка изображения для обнаружения номерных знаков
    
    - **image_file**: Изображение для обработки
    - **min_confidence**: Минимальный порог уверенности для распознавания
    """
    if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только форматы JPG, JPEG и PNG"
        )

    try:
        # Создаем директорию для сохранения изображений, если её нет
        os.makedirs("plates", exist_ok=True)
        
        # Читаем изображение
        content = await image_file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

        # Обрабатываем изображение
        result = plate_detector.process_video_frame(image)
        detected_plates = []
        
        if result:
            plate_text, confidence, processed_image = result
            
            if confidence >= min_confidence:
                # Сохраняем изображение с обнаруженным номером
                frame_path = os.path.join(
                    "plates",
                    f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(frame_path, processed_image)
                
                # Сохраняем в базу данных
                plate_record = DetectedPlate(
                    plate_number=plate_text,
                    confidence=confidence,
                    image_url=frame_path,
                    timestamp=datetime.utcnow()
                )
                db.add(plate_record)
                db.commit()
                
                detected_plates.append(PlateResponse(
                    plate_number=plate_text,
                    confidence=confidence,
                    timestamp=datetime.utcnow(),
                    image_url=frame_path
                ))
                
                logger.info(f"Обнаружен номер: {plate_text} с уверенностью {confidence:.2f}")
        
        return detected_plates
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 