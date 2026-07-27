from fastapi import FastAPI, Depends, Query, HTTPException, Path
from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker, Session
from app.database import SessionLocal, engine
from app.models.models import Base, postSchema
from typing import Annotated
from app.auth.jwt_bearer import jwtBearer
import app.auth.auth as auth




app = FastAPI(dependencies=[Depends(jwtBearer())])
app.include_router(auth.router)


Base.metadata.create_all(bind=engine)

class posts(BaseModel):
    title : str = Field(title="Title", min_length=3)
    content : str = Field(None, title="Content")
class postInDB(posts):
    id : int
    
def get_db ():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

db = Annotated[Session, Depends(get_db)]  

def length_post(db:db):
    return db.query(postSchema).count()

lenpost = Annotated[int, Depends(length_post)]

@app.get("/", tags=["test"])
def greet():
    return "Helle Kevins"

@app.post("/post/", tags=["Post"])
async def creat_post(post:posts, db:db):
    post_db = postSchema(title=post.title, content=post.content)
    db.add(post_db)
    db.commit()
    return {"message":"post added successfuly"}
    
    
@app.get("/post/", tags=["Post"])
async def get_post (db:db, 
                    lenpost : lenpost,
                    limit : Annotated[int, Query(description="Number of post you want")]=None, 
                    start : Annotated[int, Query()]=0 
                    ):
    if not limit:
        limit = lenpost
    posts = db.query(postSchema).offset(start).limit(limit).all()
    if not posts:
        return HTTPException(status_code=404, detail="Aucun post trouvé")
    return posts


@app.get("/post/{id_post}")
async def get_post_id(id_post:Annotated[int,Path(ge=0)], db:db, lenpost:lenpost)-> dict:
    if id_post >lenpost:
        return {"status_code":"401", "detail":"Post with this ID doesn't exist"}
    post = db.query(postSchema).filter(postSchema.id == id_post).first() 
    return postInDB(id=post.id, title=post.title, content=post.content).model_dump()
    
    
    