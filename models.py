from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String, unique=True, index=True)
    email      = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())