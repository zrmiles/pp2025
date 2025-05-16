from app.core.database import Base, engine
from app.models.database import DetectedPlate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recreate_tables():
    try:
        logger.info("Удаление существующих таблиц...")
        Base.metadata.drop_all(bind=engine)
        
        logger.info("Создание таблиц с новыми параметрами...")
        Base.metadata.create_all(bind=engine)
        
        logger.info("Таблицы успешно пересозданы!")
    except Exception as e:
        logger.error(f"Ошибка при пересоздании таблиц: {str(e)}")
        raise

if __name__ == "__main__":
    recreate_tables() 