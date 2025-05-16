import uvicorn
from app.core.database import init_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Инициализируем базу данных перед запуском
        logger.info("Инициализация базы данных...")
        init_db()
        
        # Запускаем приложение
        logger.info("Запуск приложения...")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {str(e)}")
        raise 