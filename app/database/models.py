"""
SQLAlchemy ORM models for Policy Document Processor database.

Defines the database schema for storing processing jobs, results, and history.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime,
    JSON, ForeignKey, Index, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class ProcessingJob(Base):
    """Model for processing jobs."""

    __tablename__ = "processing_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), unique=True, nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)  # submitted, processing, completed, failed
    document_type = Column(String(100))  # insurance_policy, legal_document, etc.

    # Processing options
    use_gpt4 = Column(Boolean, default=False)
    enable_streaming = Column(Boolean, default=True)
    confidence_threshold = Column(Float, default=0.7)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Processing statistics
    total_pages = Column(Integer)
    total_policies = Column(Integer)
    total_decision_trees = Column(Integer)
    processing_time_seconds = Column(Float)
    validation_confidence = Column(Float)
    validation_passed = Column(Boolean)

    # Error information (if failed)
    error_message = Column(Text)
    error_stage = Column(String(100))  # Which stage failed

    # Relationships
    document = relationship("PolicyDocument", back_populates="job", uselist=False, cascade="all, delete-orphan")
    results = relationship("ProcessingResult", back_populates="job", uselist=False, cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_status_created', 'status', 'created_at'),
        Index('idx_document_type', 'document_type'),
    )

    def __repr__(self):
        return f"<ProcessingJob(job_id='{self.job_id}', status='{self.status}')>"


class PolicyDocument(Base):
    """Model for policy documents."""

    __tablename__ = "policy_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), ForeignKey('processing_jobs.job_id'), unique=True, nullable=False)

    # Document metadata
    filename = Column(String(500))
    file_size_bytes = Column(Integer)
    mime_type = Column(String(100))

    # Document content (store the PDF as base64 - required for review)
    content_base64 = Column(Text, nullable=False)  # Store PDF for side-by-side viewing

    # Extracted metadata
    document_hash = Column(String(64), index=True)  # SHA-256 hash for deduplication
    total_pages = Column(Integer)
    has_tables = Column(Boolean, default=False)
    has_images = Column(Boolean, default=False)

    # Analysis results
    complexity_score = Column(Float)
    structure_type = Column(String(100))  # hierarchical, flat, mixed

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    job = relationship("ProcessingJob", back_populates="document")

    def __repr__(self):
        return f"<PolicyDocument(job_id='{self.job_id}', filename='{self.filename}')>"


class ProcessingResult(Base):
    """Model for processing results."""

    __tablename__ = "processing_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), ForeignKey('processing_jobs.job_id'), unique=True, nullable=False)

    # Complete results as JSON
    policy_hierarchy_json = Column(JSON)  # Complete policy hierarchy
    decision_trees_json = Column(JSON)  # All decision trees
    validation_result_json = Column(JSON)  # Validation results
    processing_stats_json = Column(JSON)  # Processing statistics
    metadata_json = Column(JSON)  # Document metadata

    # Quick access fields (denormalized for faster queries)
    total_policies = Column(Integer, index=True)
    total_root_policies = Column(Integer)
    max_hierarchy_depth = Column(Integer)
    total_decision_trees = Column(Integer)
    total_definitions = Column(Integer)

    # Validation metrics
    overall_confidence = Column(Float, index=True)
    completeness_score = Column(Float)
    consistency_score = Column(Float)
    traceability_score = Column(Float)
    is_valid = Column(Boolean, index=True)

    # Issue counts
    total_issues = Column(Integer)
    warning_count = Column(Integer)
    error_count = Column(Integer)
    info_count = Column(Integer)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    job = relationship("ProcessingJob", back_populates="results")

    # Indexes
    __table_args__ = (
        Index('idx_confidence_valid', 'overall_confidence', 'is_valid'),
    )

    def __repr__(self):
        return f"<ProcessingResult(job_id='{self.job_id}', policies={self.total_policies})>"


class JobHistory(Base):
    """Model for job history and user interactions."""

    __tablename__ = "job_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), ForeignKey('processing_jobs.job_id'), nullable=False, index=True)

    # Event information
    event_type = Column(String(50), nullable=False)  # created, started, completed, failed, viewed, exported
    event_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Event details
    event_data_json = Column(JSON)  # Additional event-specific data
    user_agent = Column(String(500))  # Browser/client info
    ip_address = Column(String(50))  # User IP (if applicable)

    # Notes
    notes = Column(Text)  # User-added notes

    __table_args__ = (
        Index('idx_job_event_time', 'job_id', 'event_timestamp'),
    )

    def __repr__(self):
        return f"<JobHistory(job_id='{self.job_id}', event='{self.event_type}')>"


# Database initialization helpers

def create_database_engine(database_url: str = "sqlite:///./data/policy_processor.db", **kwargs):
    """
    Create SQLAlchemy engine.

    Args:
        database_url: Database connection URL
        **kwargs: Additional engine arguments

    Returns:
        SQLAlchemy engine
    """
    engine = create_engine(
        database_url,
        echo=False,  # Set to True for SQL logging
        **kwargs
    )
    return engine


def create_all_tables(engine):
    """
    Create all database tables.

    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(engine)


def get_session_maker(engine):
    """
    Create session maker.

    Args:
        engine: SQLAlchemy engine

    Returns:
        Session maker
    """
    return sessionmaker(bind=engine)
