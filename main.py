from database import engine, get_db
from email_service import generate_otp, send_otp_email
from models import Base, User, Bacteria, WaterTreatment, Contraindication, Antibiotic, TreatmentPipeline, AnalysisSession, ChatMessage, ChatSession
from schemas import UserCreate, UserResponse, Token, LoginRequest
from crud import get_user_by_username, get_user_by_email, create_user
from auth import hash_password, verify_password, create_access_token, verify_token, create_refresh_token, verify_refresh_token
from sqlalchemy.orm import Session
from fastapi import Depends
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from predict import load_model, predict as predict_image
import shutil
import os
import uuid
import tempfile
import time
from fastapi import BackgroundTasks
from chatbot import get_answer, reset_conversation, train_word2vec, build_search_index, search, load_conversation_history, generate_chat_title_smart
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Chatbot Setup ──
import sys
DOCS_FOLDER = os.path.join(BASE_DIR, "documents_original", "documents")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

print("Building search index...")
build_search_index(DOCS_FOLDER)
from chatbot import doc_index
print(f"DEBUG: doc_index has {len(doc_index)} chunks")

# Load document contents
doc_contents = {}
for fname in os.listdir(DOCS_FOLDER):
    if fname.endswith('.txt'):
        path = os.path.join(DOCS_FOLDER, fname)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            doc_contents[fname] = f.read()

# Adapter function
def chatbot_search(query):
    return search(query, top_k=5)

# Train Word2Vec once
print("Training Word2Vec...")
w2v_model = train_word2vec(DOCS_FOLDER)
print("✅ Chatbot ready!")



otp_store = {}
#loading the model
ml_models = {}
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "resnet50_FINAL.pth")
    ml_models["model"] = load_model(model_path)
    ml_models["class_names"] = ['gram_negative', 'gram_positive', 'not_gram_stain']
    print(f"Model loaded from: {model_path}")
    yield
    

    ml_models.clear()



#App Setup
app = FastAPI(
    title="Gram Stain Classifier API",
    description="AI-powered Gram stain classification using EfficientNet-B0 (98% accuracy)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow frontend to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}





#Health Check
@app.get("/", tags=["Health"])
async def root():
   return { "status": "running", "model": "ResNet50", "accuracy": "98%", "endpoint": "POST /predict-gram"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model_loaded": "model" in ml_models}


# Register
@app.post("/register", response_model=UserResponse, tags=["Auth"])
async def register(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_username(db, user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    if get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    return create_user(db, user.username, user.email, user.password)

# Login
@app.post("/login", response_model=Token, tags=["Auth"])
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, request.email)
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    access_token = create_access_token({"sub": user.username})
    refresh_token = create_refresh_token({"sub": user.username})
    return {
    "access_token": access_token,
    "refresh_token": refresh_token,
    "token_type": "bearer",
    "username": user.username
}

@app.post("/predict-gram", tags=["Prediction"])
async def predict(file: UploadFile = File(...), current_user: str = Depends(verify_token)):
    """
    Upload a microscopy image and get Gram stain classification.

    - **file**: Image file (JPG, PNG, BMP, TIFF, WEBP, GIF)
    """

    #validte format the image user uploaded
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    # ── Save temp file with unique name to avoid conflicts ──
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = predict_image(temp_path, ml_models["model"])
        
        upload_result = cloudinary.uploader.upload(
            temp_path,
            resource_type = "image",
            folder        = "hydroscope/samples"
        )
        result["sample_image_url"] = upload_result["secure_url"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result

@app.post("/refresh-token", response_model=Token, tags=["Auth"])
async def refresh_token_endpoint(token: str, db: Session = Depends(get_db)):
    username = verify_refresh_token(token)
    user = get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    new_access_token = create_access_token({"sub": user.username})
    new_refresh_token = create_refresh_token({"sub": user.username})
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "username": user.username
    }

# Forgot Password - Step 1: Request OTP
@app.post("/forgot-password", tags=["Auth"])
async def forgot_password(email: str, db: Session = Depends(get_db)):
    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")
    
    otp = generate_otp()
    otp_store[email] = {
        "otp": otp,
        "expires_at": time.time() + 600  
    }
    
    send_otp_email(email, otp, user.username)
    return {"message": "OTP sent to your email"}
# Verify OTP
@app.post("/verify-otp", tags=["Auth"])
async def verify_otp(email: str, otp: str):
    if email not in otp_store:
        raise HTTPException(status_code=400, detail="No OTP found for this email")
    
    stored = otp_store[email]
    
    if time.time() > stored["expires_at"]:
        del otp_store[email]
        raise HTTPException(status_code=400, detail="OTP expired")
    
    if stored["otp"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    return {"message": "OTP verified successfully"}


# Reset Password
@app.post("/reset-password", tags=["Auth"])
async def reset_password(email: str, otp: str, new_password: str, db: Session = Depends(get_db)):
    
    # Password validation
    import re
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not re.search(r'[A-Z]', new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter")
    if not re.search(r'[a-z]', new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one lowercase letter")
    if not re.search(r'\d', new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one special character")

    if email not in otp_store:
        raise HTTPException(status_code=400, detail="No OTP found for this email")
    
    stored = otp_store[email]
    
    if time.time() > stored["expires_at"]:
        del otp_store[email]
        raise HTTPException(status_code=400, detail="OTP expired")
    
    if stored["otp"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    user = get_user_by_email(db, email)
    if verify_password(new_password, user.hashed_password):
      raise HTTPException(
        status_code=400, 
        detail="New password cannot be the same as your old password"
    )
    user.hashed_password = hash_password(new_password)
    db.commit()
    
    del otp_store[email]
    return {"message": "Password reset successfully"}


# ── Chatbot Endpoints ──
from pydantic import BaseModel as PydanticBase

class ChatRequest(PydanticBase):
    question: str
    session_id: int | None = None   # None = start a new conversation

class ChatResponse(PydanticBase):
    answer: str
    sources: list


@app.post("/chat", tags=["Chatbot"])
async def chat(request: ChatRequest, db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    user = get_user_by_username(db, current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # ── Get or create the session ──
    if request.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == request.session_id,
            ChatSession.user_id == user.id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        is_new_session = False
    else:
        session = ChatSession(user_id=user.id, title="New Chat")
        db.add(session)
        db.commit()
        db.refresh(session)
        is_new_session = True

    # ── Load this session's messages into chatbot memory ──
    history_rows = db.query(ChatMessage).filter(
        ChatMessage.session_id == session.id
    ).order_by(ChatMessage.created_at.asc()).limit(10).all()

    history_messages = [{"role": m.role, "content": m.content} for m in history_rows]
    load_conversation_history(history_messages)

    result = get_answer(
        user_question   = request.question,
        search_function = chatbot_search,
        docs_folder     = DOCS_FOLDER,
        groq_api_key    = GROQ_API_KEY,
        w2v_model       = w2v_model
    )

    # ── Generate title if this is the first message in the session ──
    if is_new_session:
        try:
            session.title = generate_chat_title_smart(request.question, GROQ_API_KEY)
        except Exception:
            session.title = request.question[:40]
        db.commit()

    # ── Save the new messages to DB ──
    db.add(ChatMessage(session_id=session.id, user_id=user.id, role="user", content=request.question))
    db.add(ChatMessage(session_id=session.id, user_id=user.id, role="assistant", content=result["answer"]))
    db.commit()

    return {
        "session_id": session.id,
        "title":      session.title,
        "answer":     result["answer"],
        "sources":    result["top_documents"],
        "confidence": result["confidence"]
    }


@app.post("/chat/reset", tags=["Chatbot"])
async def chat_reset(current_user: str = Depends(verify_token)):
    reset_conversation()
    return {"status": "ok"}


@app.get("/chat/sessions", tags=["Chatbot"])
async def get_chat_sessions(db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    user = get_user_by_username(db, current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user.id
    ).order_by(ChatSession.created_at.desc()).all()

    return {
        "sessions": [
            {"id": s.id, "title": s.title, "created_at": s.created_at}
            for s in sessions
        ]
    }


@app.get("/chat/sessions/{session_id}", tags=["Chatbot"])
async def get_chat_session_messages(session_id: int, db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    user = get_user_by_username(db, current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session.id
    ).order_by(ChatMessage.created_at.asc()).all()

    return {
        "session_id": session.id,
        "title":      session.title,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at}
            for m in messages
        ]
    }


@app.delete("/chat/sessions/{session_id}", tags=["Chatbot"])
async def delete_chat_session(session_id: int, db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    user = get_user_by_username(db, current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    db.query(ChatMessage).filter(ChatMessage.session_id == session.id).delete()
    db.delete(session)
    db.commit()
    return {"message": "Chat session deleted successfully"}

# ── Treatment Endpoints ──

@app.get("/api/bacteria", tags=["Treatment"])
def get_all_bacteria(db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    return db.query(Bacteria).all()


@app.get("/api/bacteria/{bacteria_id}/full", tags=["Treatment"])
def get_bacteria_full(bacteria_id: int, db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    bacteria = db.query(Bacteria).filter(Bacteria.id == bacteria_id).first()
    if not bacteria:
        raise HTTPException(status_code=404, detail="Bacteria not found")

    treatments = db.query(WaterTreatment).filter(
        WaterTreatment.bacteria_id == bacteria_id
    ).order_by(WaterTreatment.priority).all()

    antibiotics = db.query(Antibiotic).filter(
        Antibiotic.bacteria_id == bacteria_id
    ).all()

    pipeline = db.query(TreatmentPipeline).filter(
        TreatmentPipeline.bacteria_id == bacteria_id
    ).order_by(TreatmentPipeline.stage_order).all()

    return {
        "bacteria": bacteria,
        "treatments": treatments,
        "antibiotics": antibiotics,
        "pipeline": pipeline
    }

# ── History Endpoints ──

import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name  = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key     = os.getenv("CLOUDINARY_API_KEY"),
    api_secret  = os.getenv("CLOUDINARY_API_SECRET")
)

class SessionCreate(PydanticBase):
    gram_result:         str
    gram_confidence:     str
    final_bacteria_name: str
    final_bacteria_id:   int | None = None
    biochemical_tags:    str
    overridden:          bool = False
    sample_image_url:    str
    svg_content:         str

@app.post("/api/sessions", tags=["History"])
async def create_session(
    payload: SessionCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    user = get_user_by_username(db, current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Upload SVG to Cloudinary
    svg_bytes = payload.svg_content.encode("utf-8")
    upload_result = cloudinary.uploader.upload(
        svg_bytes,
        resource_type = "raw",
        format        = "svg",
        folder        = "hydroscope/paths"
    )
    path_image_url = upload_result["secure_url"]

    session = AnalysisSession(
        user_id             = user.id,
        gram_result         = payload.gram_result,
        gram_confidence     = payload.gram_confidence,
        final_bacteria_name = payload.final_bacteria_name,
        final_bacteria_id   = payload.final_bacteria_id,
        sample_image_url    = payload.sample_image_url,
        path_image_url      = path_image_url,
        biochemical_tags    = payload.biochemical_tags,
        overridden          = payload.overridden,
        status              = "completed"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/api/history", tags=["History"])
def get_user_history(
    page: int = 1,
    limit: int = 8,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    user = get_user_by_username(db, current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    total = db.query(AnalysisSession).filter(
        AnalysisSession.user_id == user.id
    ).count()

    sessions = db.query(AnalysisSession).filter(
        AnalysisSession.user_id == user.id
    ).order_by(AnalysisSession.created_at.desc()).offset((page - 1) * limit).limit(limit).all()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": -(-total // limit),
        "sessions": sessions
    }


@app.get("/api/sessions/{session_id}", tags=["History"])
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    user = get_user_by_username(db, current_user)
    session = db.query(AnalysisSession).filter(
        AnalysisSession.id == session_id,
        AnalysisSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
    


@app.delete("/api/sessions/{session_id}", tags=["History"])
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    user = get_user_by_username(db, current_user)
    session = db.query(AnalysisSession).filter(
        AnalysisSession.id == session_id,
        AnalysisSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"message": "Session deleted successfully"}

