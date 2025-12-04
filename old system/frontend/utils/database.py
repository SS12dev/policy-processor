"""
Database utilities for the frontend
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.database.db_connection import get_db_session, close_db
from contextlib import contextmanager


@contextmanager
def get_database_session():
    """Context manager for database sessions"""
    db = get_db_session()
    try:
        yield db
    finally:
        close_db(db)


def get_db_for_streamlit():
    """Get database session for streamlit - caller must close"""
    return get_db_session()