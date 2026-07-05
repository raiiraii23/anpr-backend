from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String(20), nullable=True, index=True)
    decibel_level = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    image_path = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    location = Column(String(100), nullable=True, default="Checkpoint A")
    status = Column(String(20), default="pending")  # pending, cited, dismissed
    notes = Column(Text, nullable=True)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    class_name = Column(String(30), nullable=False)
    confidence = Column(Float, nullable=False)
    plate_number = Column(String(20), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    image_path = Column(String(255), nullable=True)
