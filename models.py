from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
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


class Bacteria(Base):
    __tablename__ = "bacteria"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, unique=True, index=True)
    gram_type  = Column(String)
    morphology = Column(String)
    created_at = Column(DateTime, default=func.now())


class WaterTreatment(Base):
    __tablename__ = "water_treatments"
    id             = Column(Integer, primary_key=True, index=True)
    bacteria_id    = Column(Integer, ForeignKey("bacteria.id"))
    method_name    = Column(String)
    mechanism      = Column(String)
    specifics      = Column(String)
    priority       = Column(Integer, default=1)
    treatment_type = Column(String)


class Contraindication(Base):
    __tablename__ = "contraindications"
    id          = Column(Integer, primary_key=True, index=True)
    treatment_id = Column(Integer, ForeignKey("water_treatments.id"))
    description = Column(String)


class Antibiotic(Base):
    __tablename__ = "antibiotics"
    id              = Column(Integer, primary_key=True, index=True)
    bacteria_id     = Column(Integer, ForeignKey("bacteria.id"))
    drug_name       = Column(String)
    use_case        = Column(String)
    resistance_risk = Column(String)
    notes           = Column(String)


class TreatmentPipeline(Base):
    __tablename__ = "treatment_pipeline"
    id                 = Column(Integer, primary_key=True, index=True)
    bacteria_id        = Column(Integer, ForeignKey("bacteria.id"))
    stage_order        = Column(Integer)
    stage_name         = Column(String)
    mechanism          = Column(String)
    parameters         = Column(String)
    kill_rate_percent  = Column(Integer)
    treatment_category = Column(String)


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"))
    created_at          = Column(DateTime, default=func.now())
    gram_result         = Column(String)
    gram_confidence     = Column(String)
    final_bacteria_name = Column(String)
    final_bacteria_id   = Column(Integer, ForeignKey("bacteria.id"), nullable=True)
    sample_image_url    = Column(String)
    path_image_url      = Column(String)
    biochemical_tags    = Column(String)
    overridden          = Column(Boolean, default=False)
    status              = Column(String, default="completed")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, index=True)
    role       = Column(String)      # "user" or "assistant"
    content    = Column(String)
    created_at = Column(DateTime, server_default=func.now())    