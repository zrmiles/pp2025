from app.core.database import Base, engine
from app.models.database import DetectedPlate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    try:
        logger.info("Начинаем создание таблиц...")
        Base.metadata.drop_all(bind=engine)  
        Base.metadata.create_all(bind=engine)  
        logger.info("Таблицы успешно созданы!")
    except Exception as e:
        logger.error(f"Ошибка при создании таблиц: {str(e)}")
        raise

if __name__ == "__main__":
    create_tables() 