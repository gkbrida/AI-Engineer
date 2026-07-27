from app.models.users import userSchema
from pydantic import BaseModel, EmailStr
from fastapi import Depends, HTTPException, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pwdlib import PasswordHash 
from app.database import SessionLocal, Base
from sqlalchemy.orm import Session
from typing import Annotated
from app.auth.jwt_handle import creat_jwt
from starlette import status
from decouple import config   

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

default_pwd = config("default_pwd")

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/token")
password_hash = PasswordHash.recommended()
defaul_password = password_hash.hash(default_pwd) 

class Users(BaseModel):
    username : str 
    password : str

class UserInDB (Users):
    email : EmailStr 
    full_name : str 
    is_actived : bool 
    
class Token(BaseModel):
    access_token : str
    type_token : str

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally :
        db.close()

db = Annotated[Session, Depends(get_db)]


def authentificate_user(username:str, password:str, db:db):
    user_db = db.query(userSchema).filter(userSchema.username == username).first()
    if not user_db :
        password_hash.verify(password, defaul_password)
        return False
    if not password_hash.verify(password, user_db.hashed_password):
        return False
    return user_db
    

# Function to create new user
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: Users, db: db):
    hashed_password = password_hash.hash(user.password)
    new_user = userSchema(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully"}

@router.post("token", response_model=Token)
async def login_access_token(form_data : Annotated[OAuth2PasswordRequestForm, Depends()], db:db):
    user = authentificate_user(form_data.username,form_data.password, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User invalid")
    token = creat_jwt({"sub":form_data.username})
    return Token(access_token=token, type_token="Bearer")
    
@router.post("/login")
async def login_user(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db:db, token : Annotated[str, Depends(oauth2_bearer)]):
    user = authentificate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    return{"Token": token, "user":user}
