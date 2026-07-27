from sqlalchemy import Column, Integer, String
from ..database import Base

class postSchema(Base):
    
    __tablename__ = "postSchema"
    
    id = Column(Integer, primary_key = True, index=True)
    title = Column(String, default=None)
    content = Column(String, default=None)
    class Config : 
        schema_extra = {
            "post_demo":{
                "title": "Some title about animal",
                "content": "Some content about animal"
            }
        }