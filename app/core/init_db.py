from app.core.database import Base, engine
from app.models.database import DetectedPlate

def init_db():
    """Инициализация базы данных - создание всех таблиц"""
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("Создание таблиц базы данных...")
    init_db()
    print("Таблицы успешно созданы!") 