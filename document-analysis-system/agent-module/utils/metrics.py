"""
Metrics Tracking Utility
Tracks processing metrics, LLM usage, and performance.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime
from threading import Lock

from settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects and persists agent metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.enabled = settings.enable_metrics
        self.metrics_file = Path(settings.metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._metrics: Dict[str, Any] = {
            "agent_info": {
                "name": settings.agent_name,
                "version": settings.agent_version,
                "started_at": datetime.utcnow().isoformat()
            },
            "requests": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "by_status": {}
            },
            "processing": {
                "total_documents": 0,
                "total_processing_time_seconds": 0.0,
                "average_processing_time_seconds": 0.0,
                "by_node": {}
            },
            "llm": {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost_estimate": 0.0,
                "by_model": {}
            },
            "errors": {
                "total": 0,
                "by_type": {}
            }
        }

        # Load existing metrics if available
        self._load_metrics()

        logger.info(
            "Metrics collector initialized",
            extra={"enabled": self.enabled, "metrics_file": str(self.metrics_file)}
        )

    def _load_metrics(self):
        """Load metrics from file if it exists."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self._metrics = json.load(f)
                logger.info("Loaded existing metrics from file")
            except Exception as e:
                logger.warning(f"Could not load metrics file: {e}")

    def _save_metrics(self):
        """Save metrics to file."""
        if not self.enabled:
            return

        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self._metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metrics: {e}")

    def record_request(self, status: str, processing_time: float):
        """
        Record a request.

        Args:
            status: Request status (successful, failed, etc.)
            processing_time: Processing time in seconds
        """
        if not self.enabled:
            return

        with self._lock:
            self._metrics["requests"]["total"] += 1

            if status == "successful":
                self._metrics["requests"]["successful"] += 1
            elif status == "failed":
                self._metrics["requests"]["failed"] += 1

            # Track by status
            status_key = status.lower()
            if status_key not in self._metrics["requests"]["by_status"]:
                self._metrics["requests"]["by_status"][status_key] = 0
            self._metrics["requests"]["by_status"][status_key] += 1

            # Update processing stats
            self._metrics["processing"]["total_documents"] += 1
            self._metrics["processing"]["total_processing_time_seconds"] += processing_time

            # Calculate average
            total_docs = self._metrics["processing"]["total_documents"]
            total_time = self._metrics["processing"]["total_processing_time_seconds"]
            self._metrics["processing"]["average_processing_time_seconds"] = (
                total_time / total_docs if total_docs > 0 else 0
            )

            self._save_metrics()

    def record_node_execution(self, node_name: str, execution_time: float):
        """
        Record node execution time.

        Args:
            node_name: Name of the node
            execution_time: Execution time in seconds
        """
        if not self.enabled:
            return

        with self._lock:
            if node_name not in self._metrics["processing"]["by_node"]:
                self._metrics["processing"]["by_node"][node_name] = {
                    "executions": 0,
                    "total_time": 0.0,
                    "average_time": 0.0
                }

            node_metrics = self._metrics["processing"]["by_node"][node_name]
            node_metrics["executions"] += 1
            node_metrics["total_time"] += execution_time
            node_metrics["average_time"] = (
                node_metrics["total_time"] / node_metrics["executions"]
            )

            self._save_metrics()

    def record_llm_call(
        self,
        model: str,
        tokens: int,
        cost_estimate: float = 0.0
    ):
        """
        Record LLM API call.

        Args:
            model: Model name
            tokens: Number of tokens used
            cost_estimate: Estimated cost in USD
        """
        if not self.enabled:
            return

        with self._lock:
            self._metrics["llm"]["total_calls"] += 1
            self._metrics["llm"]["total_tokens"] += tokens
            self._metrics["llm"]["total_cost_estimate"] += cost_estimate

            # Track by model
            if model not in self._metrics["llm"]["by_model"]:
                self._metrics["llm"]["by_model"][model] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_estimate": 0.0
                }

            model_metrics = self._metrics["llm"]["by_model"][model]
            model_metrics["calls"] += 1
            model_metrics["tokens"] += tokens
            model_metrics["cost_estimate"] += cost_estimate

            self._save_metrics()

    def record_error(self, error_type: str, error_message: str):
        """
        Record an error.

        Args:
            error_type: Type/class of error
            error_message: Error message
        """
        if not self.enabled:
            return

        with self._lock:
            self._metrics["errors"]["total"] += 1

            if error_type not in self._metrics["errors"]["by_type"]:
                self._metrics["errors"]["by_type"][error_type] = {
                    "count": 0,
                    "last_message": "",
                    "last_occurrence": ""
                }

            error_metrics = self._metrics["errors"]["by_type"][error_type]
            error_metrics["count"] += 1
            error_metrics["last_message"] = error_message
            error_metrics["last_occurrence"] = datetime.utcnow().isoformat()

            self._save_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return self._metrics.copy()

    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics = {
                "agent_info": {
                    "name": settings.agent_name,
                    "version": settings.agent_version,
                    "started_at": datetime.utcnow().isoformat()
                },
                "requests": {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "by_status": {}
                },
                "processing": {
                    "total_documents": 0,
                    "total_processing_time_seconds": 0.0,
                    "average_processing_time_seconds": 0.0,
                    "by_node": {}
                },
                "llm": {
                    "total_calls": 0,
                    "total_tokens": 0,
                    "total_cost_estimate": 0.0,
                    "by_model": {}
                },
                "errors": {
                    "total": 0,
                    "by_type": {}
                }
            }
            self._save_metrics()
            logger.info("Metrics reset")


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


@contextmanager
def track_node_execution(node_name: str):
    """
    Context manager to track node execution time.

    Usage:
        with track_node_execution("pdf_parser"):
            # node code here
            pass
    """
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        get_metrics_collector().record_node_execution(node_name, execution_time)


@contextmanager
def track_llm_call(model: str):
    """
    Context manager to track LLM call.

    Usage:
        with track_llm_call("gpt-4"):
            # LLM call here
            pass
    """
    # This is a placeholder - actual token counting happens in the LLM client
    yield


@contextmanager
def track_request():
    """
    Context manager to track full request processing.

    Usage:
        with track_request() as tracker:
            # process request
            tracker.success()  # or tracker.failure()
    """
    start_time = time.time()

    class RequestTracker:
        def __init__(self):
            self.status = "unknown"

        def success(self):
            self.status = "successful"

        def failure(self):
            self.status = "failed"

    tracker = RequestTracker()

    try:
        yield tracker
    finally:
        processing_time = time.time() - start_time
        get_metrics_collector().record_request(tracker.status, processing_time)
