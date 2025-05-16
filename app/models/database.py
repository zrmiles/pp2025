from sqlalchemy import Column, Integer, String, DateTime, Text, Float, JSON, Index
from datetime import datetime
from sqlalchemy.sql import func
from app.core.database import Base

class DetectedPlate(Base):
    __tablename__ = "detected_plates"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    image_url = Column(Text)
    video_source = Column(String(255), index=True)
    confidence = Column(Float, nullable=False)
    camera_location = Column(String(255), nullable=True)
    plate_metadata = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_plate_timestamp', 'plate_number', 'timestamp'),
        Index('idx_source_timestamp', 'video_source', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<DetectedPlate(plate_number='{self.plate_number}', timestamp='{self.timestamp}')>" 