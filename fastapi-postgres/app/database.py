from typing import Annotated
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 'postgresql://<username>:<password>@<ip-address/hostname><database-name>'
SQL_ALCHEMY_DATABASE_URL = "postgresql://postgres:fiifidawson@localhost:5432/fastapi"

engine = create_engine(SQL_ALCHEMY_DATABASE_URL)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()