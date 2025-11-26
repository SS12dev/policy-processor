"""
Redis client for state management and caching.
"""
import json
import redis
from typing import Any, Optional
from config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Redis client wrapper with convenience methods."""

    def __init__(self):
        """Initialize Redis connection."""
        self._client = None
        self._connect()

    def _connect(self):
        """Establish Redis connection."""
        try:
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_socket_connect_timeout,
            )
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store a JSON-serializable value in Redis.

        Args:
            key: Redis key
            value: Value to store (will be JSON-serialized)
            ttl: Time-to-live in seconds (default from settings)

        Returns:
            True if successful
        """
        try:
            serialized = json.dumps(value, default=str)
            ttl = ttl or settings.redis_default_ttl
            return self._client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def get_json(self, key: str) -> Optional[Any]:
        """
        Retrieve and deserialize a JSON value from Redis.

        Args:
            key: Redis key

        Returns:
            Deserialized value or None if not found
        """
        try:
            value = self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def set_status(self, job_id: str, status: dict, ttl: Optional[int] = None) -> bool:
        """
        Store job status.

        Args:
            job_id: Job identifier
            status: Status dictionary
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        key = f"job:status:{job_id}"
        return self.set_json(key, status, ttl)

    def get_status(self, job_id: str) -> Optional[dict]:
        """
        Retrieve job status.

        Args:
            job_id: Job identifier

        Returns:
            Status dictionary or None
        """
        key = f"job:status:{job_id}"
        return self.get_json(key)

    def set_result(self, job_id: str, result: dict, ttl: Optional[int] = None) -> bool:
        """
        Store processing result.

        Args:
            job_id: Job identifier
            result: Result dictionary
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        key = f"job:result:{job_id}"
        return self.set_json(key, result, ttl)

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Retrieve processing result.

        Args:
            job_id: Job identifier

        Returns:
            Result dictionary or None
        """
        key = f"job:result:{job_id}"
        return self.get_json(key)

    def cache_chunk(self, doc_hash: str, chunk_index: int, data: Any, ttl: Optional[int] = None) -> bool:
        """
        Cache a document chunk.

        Args:
            doc_hash: Document hash/identifier
            chunk_index: Chunk index
            data: Chunk data
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        key = f"cache:chunk:{doc_hash}:{chunk_index}"
        return self.set_json(key, data, ttl)

    def get_cached_chunk(self, doc_hash: str, chunk_index: int) -> Optional[Any]:
        """
        Retrieve cached chunk.

        Args:
            doc_hash: Document hash/identifier
            chunk_index: Chunk index

        Returns:
            Cached chunk data or None
        """
        key = f"cache:chunk:{doc_hash}:{chunk_index}"
        return self.get_json(key)

    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key: Redis key

        Returns:
            True if successful
        """
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    def increment_counter(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.

        Args:
            key: Counter key
            amount: Amount to increment

        Returns:
            New counter value
        """
        return self._client.incrby(key, amount)

    def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to a channel.

        Args:
            channel: Channel name
            message: Message to publish (will be JSON-serialized)

        Returns:
            Number of subscribers that received the message
        """
        try:
            serialized = json.dumps(message, default=str)
            return self._client.publish(channel, serialized)
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            return 0

    def close(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()
            logger.info("Redis connection closed")


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """Get the global Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client
