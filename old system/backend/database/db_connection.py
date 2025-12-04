"""Database connection and session management."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.db_schema import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///storage/poc_db.sqlite")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables."""
    # Ensure storage directory exists
    os.makedirs("storage", exist_ok=True)

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")

def get_db():
    """Get database session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Get database session directly."""
    return SessionLocal()


def close_db(db):
    """Close database session."""
    db.close()