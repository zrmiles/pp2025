from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER", "uralazarev")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "12345")
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "plate_detection")


def create_database():
    try:
       
        conn = psycopg2.connect(
            dbname="postgres",  
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_SERVER,
            port=POSTGRES_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (POSTGRES_DB,))
        exists = cur.fetchone()
        
        if not exists:
            cur.execute(f'CREATE DATABASE {POSTGRES_DB}')
            logger.info(f"База данных {POSTGRES_DB} успешно создана")
        
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка при создании базы данных: {str(e)}")
        raise


create_database()

SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()

def init_db():
    """Инициализация базы данных - создание всех таблиц"""
    try:
        
        from app.models.database import DetectedPlate
        
        logger.info("Создание таблиц базы данных...")
        Base.metadata.create_all(bind=engine)
        logger.info("Таблицы успешно созданы!")
    except Exception as e:
        logger.error(f"Ошибка при создании таблиц: {str(e)}")
        raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 