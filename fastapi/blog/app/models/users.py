from sqlalchemy import Column, Integer, String, Boolean
from ..database import Base


class userSchema(Base):
    __tablename__ = "userSchema"
    
    id = Column(Integer,primary_key=True, index=True)
    username = Column(String,unique=True, index=True)
    email = Column(String)
    full_name = Column(String)
    hashed_password = Column(String)
    is_actived = Column(Boolean)