from sqlalchemy import Column, Integer, String, Boolean
from ..database import Base


class Users(Base):
    __tablename__ = "Users"
    
    id = Column(Integer,primary_key=True, index=True)
    username = Column(String,unique=True, index=True)
    email = Column(String)
    full_name = Column(String)
    hashed_password = Column(String)
    is_actived = Column(Boolean)