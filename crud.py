from sqlalchemy.orm import Session
from models import User
from auth import hash_password

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, name: str, username: str, email: str, password: str):
    hashed = hash_password(password)
    user = User(
        name=name,
        username=username,
        email=email,
        hashed_password=hashed
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
