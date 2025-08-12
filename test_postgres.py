from sqlmodel import create_engine
from sqlalchemy import text
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DB_POSTGRES_URL"))
with engine.connect() as conn:
    print(conn.execute(text("select version()")).scalar())
