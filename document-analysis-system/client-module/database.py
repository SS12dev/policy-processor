"""
Database Module
SQLite database for storing documents and analysis results.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from settings import settings


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or settings.db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    file_data BLOB NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    mime_type TEXT DEFAULT 'application/pdf',
                    checksum TEXT,
                    UNIQUE(checksum)
                )
            """)

            # Requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    context_id TEXT NOT NULL,
                    task_id TEXT,
                    request_timestamp TEXT NOT NULL,
                    streaming_enabled BOOLEAN NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)

            # Responses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id INTEGER NOT NULL,
                    response_timestamp TEXT NOT NULL,
                    processing_time_seconds REAL,
                    status TEXT NOT NULL,
                    response_data TEXT,
                    error_message TEXT,
                    FOREIGN KEY (request_id) REFERENCES requests(id)
                )
            """)

            # Extracted results table (for easier querying)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_id INTEGER NOT NULL,
                    page_count INTEGER,
                    word_count INTEGER,
                    heading_count INTEGER,
                    keyword_count INTEGER,
                    headings TEXT,
                    keywords TEXT,
                    summary TEXT,
                    document_type TEXT,
                    FOREIGN KEY (response_id) REFERENCES responses(id)
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_checksum
                ON documents(checksum)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_requests_document_id
                ON requests(document_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_requests_context_id
                ON requests(context_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_responses_request_id
                ON responses(request_id)
            """)

    def store_document(
        self,
        filename: str,
        file_data: bytes,
        file_size: int,
        checksum: str
    ) -> int:
        """
        Store a document in the database.

        Args:
            filename: Original filename
            file_data: PDF file bytes
            file_size: File size in bytes
            checksum: File checksum for deduplication

        Returns:
            Document ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if document already exists
            cursor.execute(
                "SELECT id FROM documents WHERE checksum = ?",
                (checksum,)
            )
            existing = cursor.fetchone()
            if existing:
                return existing[0]

            # Insert new document
            cursor.execute("""
                INSERT INTO documents (
                    filename, file_size_bytes, file_data,
                    upload_timestamp, checksum
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                filename,
                file_size,
                file_data,
                datetime.utcnow().isoformat(),
                checksum
            ))

            return cursor.lastrowid

    def create_request(
        self,
        document_id: int,
        context_id: str,
        streaming_enabled: bool
    ) -> int:
        """
        Create a new request record.

        Args:
            document_id: ID of the document
            context_id: A2A context ID
            streaming_enabled: Whether streaming was enabled

        Returns:
            Request ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO requests (
                    document_id, context_id, request_timestamp,
                    streaming_enabled, status
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                document_id,
                context_id,
                datetime.utcnow().isoformat(),
                streaming_enabled,
                "pending"
            ))

            return cursor.lastrowid

    def update_request_status(
        self,
        request_id: int,
        status: str,
        task_id: Optional[str] = None
    ):
        """Update request status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if task_id:
                cursor.execute("""
                    UPDATE requests
                    SET status = ?, task_id = ?
                    WHERE id = ?
                """, (status, task_id, request_id))
            else:
                cursor.execute("""
                    UPDATE requests
                    SET status = ?
                    WHERE id = ?
                """, (status, request_id))

    def store_response(
        self,
        request_id: int,
        response_data: Dict[str, Any],
        processing_time: float,
        status: str,
        error_message: Optional[str] = None
    ) -> int:
        """
        Store analysis response.

        Args:
            request_id: Request ID
            response_data: Response data dictionary
            processing_time: Processing time in seconds
            status: Response status
            error_message: Optional error message

        Returns:
            Response ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Store response
            cursor.execute("""
                INSERT INTO responses (
                    request_id, response_timestamp, processing_time_seconds,
                    status, response_data, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                request_id,
                datetime.utcnow().isoformat(),
                processing_time,
                status,
                json.dumps(response_data) if response_data else None,
                error_message
            ))

            response_id = cursor.lastrowid

            # Extract and store structured results
            if response_data and status == "success":
                self._store_extracted_results(cursor, response_id, response_data)

            return response_id

    def _store_extracted_results(
        self,
        cursor,
        response_id: int,
        response_data: Dict[str, Any]
    ):
        """Store extracted results for easy querying."""
        doc_info = response_data.get("document_info", {})
        extraction = response_data.get("extraction_results", {})
        analysis = response_data.get("analysis", {})

        headings_data = extraction.get("headings", {})
        keywords_data = extraction.get("keywords", {})

        cursor.execute("""
            INSERT INTO extracted_results (
                response_id, page_count, word_count,
                heading_count, keyword_count,
                headings, keywords, summary, document_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            response_id,
            doc_info.get("page_count"),
            doc_info.get("word_count"),
            headings_data.get("count", 0),
            keywords_data.get("count", 0),
            json.dumps(headings_data.get("items", [])),
            json.dumps(keywords_data.get("items", [])),
            analysis.get("summary"),
            analysis.get("document_type")
        ))

    def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent documents with their latest analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    d.id,
                    d.filename,
                    d.file_size_bytes,
                    d.upload_timestamp,
                    d.checksum,
                    COUNT(req.id) as request_count,
                    MAX(resp.response_timestamp) as last_analyzed
                FROM documents d
                LEFT JOIN requests req ON d.id = req.document_id
                LEFT JOIN responses resp ON req.id = resp.request_id
                GROUP BY d.id
                ORDER BY d.upload_timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_document_analyses(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all analyses for a document."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    req.id as request_id,
                    req.request_timestamp,
                    req.streaming_enabled,
                    resp.status,
                    resp.processing_time_seconds,
                    resp.response_data,
                    er.heading_count,
                    er.keyword_count,
                    er.summary
                FROM requests req
                LEFT JOIN responses resp ON req.id = resp.request_id
                LEFT JOIN extracted_results er ON resp.id = er.response_id
                WHERE req.document_id = ?
                ORDER BY req.request_timestamp DESC
            """, (document_id,))

            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                if row_dict["response_data"]:
                    row_dict["response_data"] = json.loads(row_dict["response_data"])
                results.append(row_dict)

            return results

    def get_document_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID (without file data)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, filename, file_size_bytes, upload_timestamp, mime_type
                FROM documents
                WHERE id = ?
            """, (document_id,))

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_document_file_data(self, document_id: int) -> Optional[bytes]:
        """Get document file data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT file_data
                FROM documents
                WHERE id = ?
            """, (document_id,))

            row = cursor.fetchone()
            return row[0] if row else None


# Global database instance
_database: Optional[Database] = None


def get_database() -> Database:
    """Get or create the global database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database
