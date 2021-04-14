from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    echo=settings.SQLALCHEMY_ECHO,
    pool_pre_ping=True,
    future=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
