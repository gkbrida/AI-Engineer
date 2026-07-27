from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from decouple import config  


username = config("username")
password = config("password")
host = config("host")
port = config("port")
database = config("database")
domain = config("domain")

DATABASE_URL = f"{domain}://{username}:{password}@{host}:{port}/{database}"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()