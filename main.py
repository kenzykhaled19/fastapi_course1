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
    "name": user.username
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
        "name": user.name
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
        "expires_at": time.time() + 600  # 10 دقايق
    }
    
    send_otp_email(email, otp, user.name)
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

















































































































































































# from fastapi import FastAPI, HTTPException,status
# from enum import Enum

# from pydantic import BaseModel


# app = FastAPI()

# class Post(BaseModel):
#     id: int
#     title: str
#     content: str


# my_posts=[
#     {"id": 1, "title": "post1" , "content": "this is the content of post1" },
#     {"id": 2, "title": "post2" , "content": "this is the content of post2" },
#     {"id": 3, "title": "post3" , "content": "this is the content of post3" },]




# @app.get("/posts" , status_code=status.HTTP_201_CREATED)
# def get_posts():
#     return {"message":my_posts}


# def delete_post(id):
#     for i,p in enumerate(my_posts):
#         if p["id"]==id:
#             return i

# @app.delete("/posts/{id}")
# def delete_posts(id: int):
#     item=delete_post(id)
#     if item is None:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"post with id {id} not found")
#     else:
#      my_posts.pop(item)
#      return {"message":f"post with id {id} has been deleted"}





# @app.post("/posts")
# def create_posts(post :Post):
#     my_posts.append(post)
#     return {"message":f"post with id {post.id} has been created"}


# @app.get("/posts/latestpost")
# def get_latest_post():
#     item=my_posts[len(my_posts)-1]
#     print(item)
#     return {"latest post:": item}


# @app.get("/posts/{id}")
# def get_posts(id:int):
#     for p in my_posts:
#         if p["id"]==id:
#             return {"post:": p}








































































# items=[
#     {"id": 1, "name": "item1" , "price": "$10" , "stock" : True },
#     {"id": 2, "name": "item2" , "price": "$20" , "stock" : False },
#     {"id": 3, "name": "item3" , "price": "$30" , "stock" : True },
#     {"id": 4, "name": "item4" , "price": "$40" , "stock" : False },
#     {"id": 5, "name": "item5" , "price": "$50" , "stock" : True },
# ]

# @app.get("/items")
# async def get_items(start: int =0, end: int =10, id: int=None , name:str=None , in_stock: bool=None):
#     if id:
#         item=[item for item in items if item["id"]==id]
#         if item:
#             return item
#         else:
#              return {"message":" Enter a valid id"}
          
#     if name:
#         item=[item for item in items if item["name"]==name]
#         if item:
#             return item
#         else:
#              return {"message":" Enter a valid name"}


#     if in_stock:
#             item=[item for item in items if item["stock"]==True]
#             return item
#     elif in_stock == False:
#             item=[item for item in items if item["stock"]==False]
#             return item
#     else:
#         return {"message":" No items in stock"}


#     return items[start : start + end]
















# class ListUser(str, Enum):
#     Admin = "kenzy"
#     manager = "john"
#     user = "mary"

# @app.get("/{users}/{ListyUser}" , description="Get users")
# async def get_users(users: str , ListyUser: ListUser):
#     return {"message": f"this is a get request for {users} added by {ListyUser.name}:"}

# @app.get("/")
# async def root():
#     return {"message": "Hello Kenzy"}

# @app.post("/")
# async def post():
#     return {"message": "this is a post request"}

