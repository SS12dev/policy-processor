"""
Backend handler for Streamlit UI.

Handles storing results from A2A agent responses into the UI database.
"""
import base64
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

from app.database.operations import DatabaseOperations
from app.utils.logger import get_logger

logger = get_logger(__name__)


class UIBackendHandler:
    """
    Backend handler for processing and storing A2A agent responses.

    Responsibilities:
    - Extract results from A2A responses
    - Save PDF documents to database
    - Save processing results to database
    - Update job status and metadata
    """

    def __init__(self, db_ops: DatabaseOperations):
        """
        Initialize the backend handler.

        Args:
            db_ops: Database operations instance
        """
        self.db_ops = db_ops
        logger.info("UIBackendHandler initialized")

    def process_a2a_response(
        self,
        response: Dict[str, Any],
        pdf_bytes: bytes,
        filename: str,
        policy_name: str,
        use_gpt4: bool = False,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process A2A response and store results in database.

        Args:
            response: Response from A2A client
            pdf_bytes: Original PDF bytes
            filename: Original filename
            policy_name: User-provided unique policy name
            use_gpt4: Whether GPT-4 was used
            confidence_threshold: Confidence threshold used

        Returns:
            Processed result with job_id and status
        """
        try:
            logger.info("[UI] Processing A2A response for storage")

            # Check response status
            status = response.get("status", "unknown")

            if status == "failed":
                logger.error(f"[UI] A2A processing failed: {response.get('message', 'Unknown error')}")
                return response

            # Extract results data
            results_data = response.get("results")
            if not results_data:
                logger.warning("[UI] No results data in response")
                return response

            # Extract job_id
            job_id = response.get("job_id") or results_data.get("job_id")
            if not job_id:
                logger.error("[UI] No job_id in response")
                return response

            logger.info(f"[UI] Saving data for job {job_id}")

            # Calculate document hash
            doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
            document_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

            # Save document to database
            document_data = {
                "job_id": job_id,
                "policy_name": policy_name,
                "content_base64": document_base64,
                "document_hash": doc_hash,
                "file_size_bytes": len(pdf_bytes),
                "filename": filename,
                "mime_type": "application/pdf",
                "uploaded_at": datetime.utcnow(),
            }
            self.db_ops.save_document(document_data)
            logger.info(f"[UI] Saved document for job {job_id} with policy name '{policy_name}'")

            # Extract job metadata
            job_data = {
                "job_id": job_id,
                "policy_name": policy_name,
                "status": results_data.get("status", "completed"),
                "created_at": datetime.utcnow(),
                "started_at": datetime.utcnow(),
                "use_gpt4": use_gpt4,
                "confidence_threshold": confidence_threshold,
            }

            # Extract document type from metadata
            if "metadata" in results_data:
                metadata = results_data["metadata"]
                job_data["document_type"] = metadata.get("document_type", "unknown")
                job_data["total_pages"] = metadata.get("total_pages")

            # Extract statistics from results
            if "policy_hierarchy" in results_data:
                job_data["total_policies"] = results_data["policy_hierarchy"].get("total_policies", 0)

            if "decision_trees" in results_data:
                job_data["total_decision_trees"] = len(results_data["decision_trees"])

            if "validation_result" in results_data:
                validation = results_data["validation_result"]
                job_data["validation_confidence"] = validation.get("overall_confidence", 0.0)
                job_data["validation_passed"] = validation.get("is_valid", False)

            if "processing_stats" in results_data:
                stats = results_data["processing_stats"]
                job_data["processing_time_seconds"] = stats.get("processing_time_seconds", 0.0)

            # Mark as completed
            job_data["completed_at"] = datetime.utcnow()

            # Save job to database
            self.db_ops.save_job(job_data)
            logger.info(f"[UI] Saved job metadata for {job_id}")

            # Save full results to database
            self.db_ops.save_results(job_id, results_data)
            logger.info(f"[UI] Saved full results for {job_id}")

            # Return enhanced response
            return {
                **response,
                "saved_to_database": True,
                "job_id": job_id,
                "document_hash": doc_hash
            }

        except Exception as e:
            logger.error(f"[UI] Error processing A2A response: {e}", exc_info=True)
            return {
                "status": "failed",
                "message": f"Error saving to database: {str(e)}",
                "original_response": response
            }

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job results from database, including PDF document.

        Args:
            job_id: Job identifier

        Returns:
            Results data with document content or None
        """
        try:
            # Get the processing results
            results = self.db_ops.get_results(job_id)
            
            if not results:
                logger.warning(f"No results found for job {job_id}")
                return None
            
            # Get the document content
            document = self.db_ops.get_document(job_id)
            
            if document:
                # Decode base64 PDF content
                import base64
                try:
                    pdf_bytes = base64.b64decode(document.content_base64)
                    results['document_content'] = pdf_bytes
                    results['document_filename'] = document.filename or 'document.pdf'
                    logger.info(f"Retrieved document for job {job_id}: {len(pdf_bytes)} bytes")
                except Exception as e:
                    logger.error(f"Error decoding document content: {e}")
            else:
                logger.warning(f"No document found for job {job_id}")
            
            # Get job metadata for additional info
            job = self.db_ops.get_job(job_id)
            if job:
                # Convert job model to dict
                job_dict = {
                    'total_pages': job.total_pages,
                    'processing_time_seconds': job.processing_time_seconds,
                    'validation_confidence': job.validation_confidence,
                    'document_type': job.document_type,
                }
                
                # Merge job metadata into results
                if 'metadata' not in results:
                    results['metadata'] = {}
                
                results['metadata']['total_pages'] = job_dict.get('total_pages')
                results['metadata']['processing_time'] = job_dict.get('processing_time_seconds')
                results['metadata']['validation_confidence'] = job_dict.get('validation_confidence')
                results['metadata']['document_type'] = job_dict.get('document_type')
                
                # Add validation result if not present
                if 'validation_result' not in results:
                    results['validation_result'] = {}
                
                results['validation_result']['overall_confidence'] = job_dict.get('validation_confidence', 0.0)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting job results: {e}", exc_info=True)
            return None

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status from database.

        Args:
            job_id: Job identifier

        Returns:
            Job data or None
        """
        try:
            jobs = self.db_ops.list_jobs(limit=1)
            for job in jobs:
                if job["job_id"] == job_id:
                    return job
            return None
        except Exception as e:
            logger.error(f"Error getting job status: {e}", exc_info=True)
            return None
