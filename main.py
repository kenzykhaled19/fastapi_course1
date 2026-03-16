from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from predict import load_model, predict_gram
import shutil
import os
import uuid
import tempfile

#loading the model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "gram_classifier.pth")
    ml_models["model"], ml_models["class_names"] = load_model(model_path)
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}





#Health Check
@app.get("/", tags=["Health"])
async def root():
    return { "status": "running", "model":  "EfficientNet-B0","accuracy": "98%", "endpoint": "POST /predict-gram"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model_loaded": "model" in ml_models}

@app.post("/predict-gram", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
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

        result = predict_gram(
            temp_path,
            ml_models["model"],
            ml_models["class_names"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result



















































































































































































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

