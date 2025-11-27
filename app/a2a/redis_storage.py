"""
Redis-based temporary storage for agent processing.

Used for stateless agent operation in containers with support for concurrent requests.
All data is stored in Redis with TTL for automatic cleanup.
"""
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from app.utils.redis_client import get_redis_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RedisAgentStorage:
    """
    Redis-based temporary storage for agent processing.

    Provides stateless storage for:
    - Job status tracking
    - Processing state
    - Final results (with TTL)
    - Concurrent request coordination
    """

    def __init__(self, result_ttl_hours: int = 24):
        """
        Initialize Redis storage.

        Args:
            result_ttl_hours: Hours to keep results in Redis before expiry
        """
        self.redis_client = get_redis_client()
        self.result_ttl = result_ttl_hours * 3600  # Convert to seconds
        logger.info(f"Redis agent storage initialized with {result_ttl_hours}h TTL")

    def save_job_status(self, job_id: str, status_data: Dict[str, Any], ttl_hours: int = 1) -> None:
        """
        Save job processing status.

        Args:
            job_id: Job identifier
            status_data: Status information
            ttl_hours: Hours to keep status (default 1 hour)
        """
        try:
            key = f"job:{job_id}:status"
            self.redis_client.client.setex(
                key,
                ttl_hours * 3600,
                json.dumps(status_data, default=str)
            )
            logger.debug(f"Saved status for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to save job status: {e}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job processing status.

        Args:
            job_id: Job identifier

        Returns:
            Status data or None
        """
        try:
            key = f"job:{job_id}:status"
            data = self.redis_client.client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None

    def save_job_result(self, job_id: str, result_data: Dict[str, Any]) -> None:
        """
        Save final processing result with TTL.

        Args:
            job_id: Job identifier
            result_data: Complete processing result
        """
        try:
            key = f"job:{job_id}:result"
            self.redis_client.client.setex(
                key,
                self.result_ttl,
                json.dumps(result_data, default=str)
            )
            logger.info(f"Saved result for job {job_id} (TTL: {self.result_ttl}s)")
        except Exception as e:
            logger.error(f"Failed to save job result: {e}")

    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get final processing result.

        Args:
            job_id: Job identifier

        Returns:
            Result data or None
        """
        try:
            key = f"job:{job_id}:result"
            data = self.redis_client.client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get job result: {e}")
            return None

    def delete_job_data(self, job_id: str) -> None:
        """
        Delete all data for a job.

        Args:
            job_id: Job identifier
        """
        try:
            keys = [
                f"job:{job_id}:status",
                f"job:{job_id}:result",
                f"job:{job_id}:state"
            ]
            deleted = self.redis_client.client.delete(*keys)
            logger.info(f"Deleted {deleted} keys for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to delete job data: {e}")

    def acquire_lock(self, job_id: str, timeout_seconds: int = 300) -> bool:
        """
        Acquire a lock for job processing to prevent duplicate processing.

        Args:
            job_id: Job identifier
            timeout_seconds: Lock timeout in seconds

        Returns:
            True if lock acquired, False otherwise
        """
        try:
            key = f"job:{job_id}:lock"
            acquired = self.redis_client.client.set(
                key,
                "locked",
                nx=True,
                ex=timeout_seconds
            )
            if acquired:
                logger.info(f"Acquired lock for job {job_id}")
            else:
                logger.warning(f"Failed to acquire lock for job {job_id} (already locked)")
            return bool(acquired)
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release_lock(self, job_id: str) -> None:
        """
        Release a job processing lock.

        Args:
            job_id: Job identifier
        """
        try:
            key = f"job:{job_id}:lock"
            self.redis_client.client.delete(key)
            logger.info(f"Released lock for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")

    def increment_active_jobs(self) -> int:
        """
        Increment and return active job count.

        Returns:
            Current active job count
        """
        try:
            key = "agent:active_jobs"
            count = self.redis_client.client.incr(key)
            logger.debug(f"Active jobs: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to increment active jobs: {e}")
            return 0

    def decrement_active_jobs(self) -> int:
        """
        Decrement and return active job count.

        Returns:
            Current active job count
        """
        try:
            key = "agent:active_jobs"
            count = self.redis_client.client.decr(key)
            if count < 0:
                self.redis_client.client.set(key, 0)
                count = 0
            logger.debug(f"Active jobs: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to decrement active jobs: {e}")
            return 0

    def get_active_jobs_count(self) -> int:
        """
        Get current active job count.

        Returns:
            Active job count
        """
        try:
            key = "agent:active_jobs"
            count = self.redis_client.client.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Failed to get active jobs count: {e}")
            return 0

    def save_processing_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Save processing metrics for monitoring.

        Args:
            metrics: Metrics data
        """
        try:
            key = f"agent:metrics:{datetime.utcnow().strftime('%Y%m%d%H')}"
            self.redis_client.client.rpush(key, json.dumps(metrics, default=str))
            self.redis_client.client.expire(key, 86400 * 7)  # Keep for 7 days
            logger.debug("Saved processing metrics")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
