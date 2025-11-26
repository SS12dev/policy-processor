"""
Database operations for Policy Document Processor.

Provides CRUD operations for jobs, documents, results, and history.
"""

import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from app.database.models import (
    ProcessingJob, PolicyDocument, ProcessingResult, JobHistory,
    create_database_engine, create_all_tables, get_session_maker
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseOperations:
    """Database operations manager."""

    def __init__(self, database_url: str = "sqlite:///./data/policy_processor.db"):
        """
        Initialize database operations.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_database_engine(database_url)
        self.SessionMaker = get_session_maker(self.engine)

        # Create tables if they don't exist
        create_all_tables(self.engine)

        logger.info(f"Database initialized: {database_url}")

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.SessionMaker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
        finally:
            session.close()

    # Job operations

    def save_job(self, job_data: Dict[str, Any]) -> ProcessingJob:
        """
        Save a new processing job.

        Args:
            job_data: Job data dictionary

        Returns:
            Created ProcessingJob instance
        """
        with self.get_session() as session:
            job = ProcessingJob(
                job_id=job_data["job_id"],
                status=job_data.get("status", "submitted"),
                document_type=job_data.get("document_type"),
                use_gpt4=job_data.get("use_gpt4", False),
                enable_streaming=job_data.get("enable_streaming", True),
                confidence_threshold=job_data.get("confidence_threshold", 0.7),
                created_at=job_data.get("created_at", datetime.utcnow()),
            )

            session.add(job)
            session.commit()
            session.refresh(job)

            logger.info(f"Saved job {job.job_id}")

            # Add history entry
            self.add_history_event(job.job_id, "created", {"status": "submitted"})

            return job

    def update_job_status(
        self,
        job_id: str,
        status: str,
        **updates
    ) -> Optional[ProcessingJob]:
        """
        Update job status and other fields.

        Args:
            job_id: Job identifier
            status: New status
            **updates: Additional fields to update

        Returns:
            Updated ProcessingJob or None if not found
        """
        with self.get_session() as session:
            job = session.query(ProcessingJob).filter_by(job_id=job_id).first()

            if not job:
                logger.warning(f"Job {job_id} not found")
                return None

            job.status = status
            job.updated_at = datetime.utcnow()

            # Update timestamps based on status
            if status == "processing" and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in ["completed", "failed"]:
                job.completed_at = datetime.utcnow()
                if job.started_at:
                    job.processing_time_seconds = (
                        job.completed_at - job.started_at
                    ).total_seconds()

            # Apply additional updates
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)

            session.commit()
            session.refresh(job)

            logger.info(f"Updated job {job_id} status to {status}")

            # Add history entry
            self.add_history_event(job_id, status, updates)

            return job

    def update_job(self, job_id: str, job_data: Dict[str, Any]) -> Optional[ProcessingJob]:
        """
        Update job with new data.

        Args:
            job_id: Job identifier
            job_data: Dictionary of fields to update

        Returns:
            Updated ProcessingJob or None if not found
        """
        with self.get_session() as session:
            job = session.query(ProcessingJob).filter_by(job_id=job_id).first()

            if not job:
                logger.warning(f"Job {job_id} not found")
                return None

            # Update all provided fields
            for key, value in job_data.items():
                if hasattr(job, key) and key != 'job_id':  # Don't update primary key
                    setattr(job, key, value)

            job.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(job)

            logger.info(f"Updated job {job_id}")

            return job

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """
        Get a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            ProcessingJob or None
        """
        with self.get_session() as session:
            job = session.query(ProcessingJob).filter_by(job_id=job_id).first()
            if job:
                # Detach from session
                session.expunge(job)
            return job

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status (None for all)
            limit: Maximum number of jobs
            offset: Number of jobs to skip

        Returns:
            List of job dictionaries
        """
        with self.get_session() as session:
            query = session.query(ProcessingJob)

            if status:
                query = query.filter_by(status=status)

            query = query.order_by(desc(ProcessingJob.created_at))
            query = query.limit(limit).offset(offset)

            jobs = query.all()

            return [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "document_type": job.document_type,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "total_policies": job.total_policies,
                    "validation_confidence": job.validation_confidence,
                    "processing_time_seconds": job.processing_time_seconds,
                }
                for job in jobs
            ]

    def count_jobs(self, status: Optional[str] = None) -> int:
        """
        Count jobs, optionally filtered by status.

        Args:
            status: Filter by status (None for all)

        Returns:
            Number of jobs
        """
        with self.get_session() as session:
            query = session.query(ProcessingJob)

            if status:
                query = query.filter_by(status=status)

            return query.count()

    # Document operations

    def save_document(self, document_data: Dict[str, Any]) -> PolicyDocument:
        """
        Save document metadata.

        Args:
            document_data: Document data dictionary

        Returns:
            Created PolicyDocument instance
        """
        with self.get_session() as session:
            # Calculate hash if content provided
            document_hash = None
            if document_data.get("content_base64"):
                content = document_data["content_base64"]
                document_hash = hashlib.sha256(content.encode()).hexdigest()

            document = PolicyDocument(
                job_id=document_data["job_id"],
                filename=document_data.get("filename"),
                file_size_bytes=document_data.get("file_size_bytes"),
                mime_type=document_data.get("mime_type", "application/pdf"),
                content_base64=document_data.get("content_base64"),
                document_hash=document_hash,
                total_pages=document_data.get("total_pages"),
                has_tables=document_data.get("has_tables", False),
                has_images=document_data.get("has_images", False),
                complexity_score=document_data.get("complexity_score"),
                structure_type=document_data.get("structure_type"),
            )

            session.add(document)
            session.commit()
            session.refresh(document)

            logger.info(f"Saved document for job {document.job_id}")

            return document

    def get_document(self, job_id: str) -> Optional[PolicyDocument]:
        """
        Get document by job ID.

        Args:
            job_id: Job identifier

        Returns:
            PolicyDocument or None
        """
        with self.get_session() as session:
            document = session.query(PolicyDocument).filter_by(job_id=job_id).first()
            if document:
                session.expunge(document)
            return document

    # Results operations

    def save_results(self, job_id: str, results: Dict[str, Any]) -> ProcessingResult:
        """
        Save processing results.

        Args:
            job_id: Job identifier
            results: Complete results dictionary

        Returns:
            Created ProcessingResult instance
        """
        with self.get_session() as session:
            # Extract quick-access fields
            policy_hierarchy = results.get("policy_hierarchy", {})
            decision_trees = results.get("decision_trees", [])
            validation = results.get("validation_result", {})

            result = ProcessingResult(
                job_id=job_id,
                policy_hierarchy_json=policy_hierarchy,
                decision_trees_json=decision_trees,
                validation_result_json=validation,
                processing_stats_json=results.get("processing_stats", {}),
                metadata_json=results.get("metadata", {}),
                total_policies=policy_hierarchy.get("total_policies", 0),
                total_root_policies=len(policy_hierarchy.get("root_policies", [])),
                max_hierarchy_depth=policy_hierarchy.get("max_depth", 0),
                total_decision_trees=len(decision_trees),
                total_definitions=len(policy_hierarchy.get("definitions", {})),
                overall_confidence=validation.get("overall_confidence", 0.0),
                completeness_score=validation.get("completeness_score", 0.0),
                consistency_score=validation.get("consistency_score", 0.0),
                traceability_score=validation.get("traceability_score", 0.0),
                is_valid=validation.get("is_valid", False),
                total_issues=len(validation.get("issues", [])),
                warning_count=sum(1 for i in validation.get("issues", []) if i.get("severity") == "warning"),
                error_count=sum(1 for i in validation.get("issues", []) if i.get("severity") == "error"),
                info_count=sum(1 for i in validation.get("issues", []) if i.get("severity") == "info"),
            )

            session.add(result)

            # Also update the job with summary stats
            job = session.query(ProcessingJob).filter_by(job_id=job_id).first()
            if job:
                job.total_pages = results.get("metadata", {}).get("total_pages")
                job.total_policies = result.total_policies
                job.total_decision_trees = result.total_decision_trees
                job.validation_confidence = result.overall_confidence
                job.validation_passed = result.is_valid

            session.commit()
            session.refresh(result)

            logger.info(f"Saved results for job {job_id}")

            # Add history entry
            self.add_history_event(job_id, "results_saved", {
                "total_policies": result.total_policies,
                "confidence": result.overall_confidence
            })

            return result

    def get_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing results.

        Args:
            job_id: Job identifier

        Returns:
            Results dictionary or None
        """
        with self.get_session() as session:
            result = session.query(ProcessingResult).filter_by(job_id=job_id).first()

            if not result:
                return None

            return {
                "job_id": job_id,
                "status": "completed",
                "metadata": result.metadata_json,
                "policy_hierarchy": result.policy_hierarchy_json,
                "decision_trees": result.decision_trees_json,
                "validation_result": result.validation_result_json,
                "processing_stats": result.processing_stats_json,
            }

    # History operations

    def add_history_event(
        self,
        job_id: str,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> JobHistory:
        """
        Add a history event.

        Args:
            job_id: Job identifier
            event_type: Type of event
            event_data: Event-specific data
            notes: Optional notes

        Returns:
            Created JobHistory instance
        """
        with self.get_session() as session:
            history = JobHistory(
                job_id=job_id,
                event_type=event_type,
                event_data_json=event_data or {},
                notes=notes,
            )

            session.add(history)
            session.commit()
            session.refresh(history)

            logger.debug(f"Added history event for job {job_id}: {event_type}")

            return history

    def get_job_history(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get history for a job.

        Args:
            job_id: Job identifier

        Returns:
            List of history events
        """
        with self.get_session() as session:
            history_items = session.query(JobHistory).filter_by(job_id=job_id).order_by(
                JobHistory.event_timestamp
            ).all()

            return [
                {
                    "event_type": h.event_type,
                    "timestamp": h.event_timestamp.isoformat(),
                    "data": h.event_data_json,
                    "notes": h.notes,
                }
                for h in history_items
            ]

    # Statistics and analytics

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics.

        Returns:
            Statistics dictionary
        """
        with self.get_session() as session:
            total_jobs = session.query(ProcessingJob).count()
            completed_jobs = session.query(ProcessingJob).filter_by(status="completed").count()
            failed_jobs = session.query(ProcessingJob).filter_by(status="failed").count()
            processing_jobs = session.query(ProcessingJob).filter_by(status="processing").count()

            # Average confidence for completed jobs
            avg_confidence = session.query(ProcessingJob).filter(
                and_(
                    ProcessingJob.status == "completed",
                    ProcessingJob.validation_confidence.isnot(None)
                )
            ).with_entities(ProcessingJob.validation_confidence).all()

            avg_conf_value = sum(c[0] for c in avg_confidence) / len(avg_confidence) if avg_confidence else 0.0

            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "processing_jobs": processing_jobs,
                "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0.0,
                "average_confidence": avg_conf_value,
            }
