from sqlalchemy import Column, Integer, String, DateTime, Text, Float, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.sql import func

Base = declarative_base()

class DetectedPlate(Base):
    __tablename__ = "detected_plates"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    image_url = Column(Text)
    video_source = Column(String(255), index=True)
    confidence = Column(Float, nullable=False)
    camera_location = Column(String(255), nullable=True)
    plate_metadata = Column(JSON, nullable=True)
    
    # Создаем составной индекс для часто используемых запросов
    __table_args__ = (
        Index('idx_plate_timestamp', 'plate_number', 'timestamp'),
        Index('idx_source_timestamp', 'video_source', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<DetectedPlate(plate_number='{self.plate_number}', timestamp='{self.timestamp}')>" 