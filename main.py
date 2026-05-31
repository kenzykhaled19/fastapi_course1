from database import engine, get_db
from email_service import generate_otp, send_otp_email
from models import Base, User
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
from chatbot import get_answer, reset_conversation, train_word2vec, build_search_index, search
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

class ChatResponse(PydanticBase):
    answer: str
    sources: list

@app.post("/chat", tags=["Chatbot"])
async def chat(request: ChatRequest, current_user: str = Depends(verify_token)):
    result = get_answer(
        user_question   = request.question,
        search_function = chatbot_search,
        docs_folder     = DOCS_FOLDER,
        groq_api_key    = GROQ_API_KEY,
        w2v_model       = w2v_model
    )
    return {
        "answer":  result["answer"],
        "sources": result["top_documents"]
    }

@app.post("/chat/reset", tags=["Chatbot"])
async def chat_reset(current_user: str = Depends(verify_token)):
    reset_conversation()
    return {"status": "ok"}
