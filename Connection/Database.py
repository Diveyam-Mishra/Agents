import os
from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.environ.get("DB_HOST")    
DB_PORT = os.environ.get("DB_PORT")    
DB_NAME = os.environ.get("DB_NAME")    
DB_USER = os.environ.get("DB_USER")    
DB_PASSWORD = os.environ.get("DB_PASSWORD")

from sqlalchemy import create_engine

db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"