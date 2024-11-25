from sqlalchemy import create_engine #type: ignore
from sqlalchemy.orm import declarative_base #type: ignore
from sqlalchemy.orm import sessionmaker #type: ignore

from configure import Config

config = Config()
URL_DATABASE = Config.database.URL_DATABASE
engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
