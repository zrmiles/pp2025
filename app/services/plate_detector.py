import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from loguru import logger
import re
from typing import Optional, Tuple, List
import os
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='ru', use_angle_cls=True)

class PlateDetector:
    def __init__(self):
        
        self.car_model = YOLO('yolov8n.pt')
        
        self.plate_model = YOLO('yolov8n.pt')  
        self.plate_pattern = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\s?\d{2,3}$')
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для улучшения распознавания"""
        
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh

    def detect_plate(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Обнаружение номерных знаков на кадре"""
        results = self.car_model(frame)
        detected_plates = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 2:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    
                    car_region = frame[y1:y2, x1:x2]
                    if car_region.size == 0:
                        continue
                    
                    
                    plate_results = self.plate_model(car_region)
                    for plate_result in plate_results:
                        plate_boxes = plate_result.boxes
                        for plate_box in plate_boxes:
                            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                            plate_confidence = float(plate_box.conf[0])
                            
                            
                            plate_region = car_region[py1:py2, px1:px2]
                            if plate_region.size == 0:
                                continue
                            
                            
                            processed_plate = self.preprocess_image(plate_region)
                            
                            
                            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789'
                            plate_text = pytesseract.image_to_string(
                                processed_plate, 
                                lang='rus',
                                config=custom_config
                            )
                            
                            
                            plate_text = self.clean_plate_number(plate_text)
                            if self.validate_plate_number(plate_text):
                                
                                abs_coords = (
                                    x1 + px1,
                                    y1 + py1,
                                    x1 + px2,
                                    y1 + py2
                                )
                                detected_plates.append((plate_text, plate_confidence, abs_coords))
        
        return detected_plates

    def clean_plate_number(self, text: str) -> str:
        """Очистка распознанного номера"""
        
        cleaned = re.sub(r'[^АВЕКМНОРСТУХ0-9\s]', '', text.upper())
        
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def validate_plate_number(self, plate_number: str) -> bool:
        """Проверка соответствия номера российскому формату"""
        return bool(self.plate_pattern.match(plate_number))

    def process_video_frame(self, frame: np.ndarray) -> Optional[Tuple[str, float, np.ndarray]]:
        """Обработка одного кадра видео или изображения"""
        detected_plates = self.detect_plate(frame)
        
        if not detected_plates:
            return None
            
    
        best_plate = max(detected_plates, key=lambda x: x[1])
        plate_text, confidence, coords = best_plate
        
        
        x1, y1, x2, y2 = coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return plate_text, confidence, frame

    def process_video_stream(self, video_source: str):
        """Обработка видеопотока"""
        cap = cv2.VideoCapture(video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            detected_plates = self.detect_plate(frame)
            
            for plate_number, confidence, coords in detected_plates:
                logger.info(f"Detected plate: {plate_number} with confidence {confidence:.2f}")
                
        
                
        cap.release()

def process_image_with_ocr(image: np.ndarray):
    temp_path = 'temp_input.jpg'
    cv2.imwrite(temp_path, image)
    result = ocr.ocr(temp_path, cls=True)
    plates = []
    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        plates.append((text, conf))
    return plates 