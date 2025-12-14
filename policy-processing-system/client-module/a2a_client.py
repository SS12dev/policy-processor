"""
A2A Client for Policy Processing System
Based on working document-analysis-system implementation.
"""

import asyncio
import base64
from typing import Optional, AsyncIterator, Dict, Any
import httpx

from a2a import types
from a2a.client import Client, ClientConfig, ClientFactory

from settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PolicyProcessingClient:
    """Async client for policy processing agent using A2A SDK."""

    def __init__(self):
        """Initialize the client."""
        self.agent_url = settings.agent_url
        self.timeout = settings.agent_timeout
        self.prefer_streaming = settings.agent_prefer_streaming

        self.http_client: Optional[httpx.AsyncClient] = None
        self.a2a_client: Optional[Client] = None
        self.agent_card: Optional[types.AgentCard] = None
        
        logger.info(f"[CLIENT] Initialized: {self.agent_url}")

    async def connect(self):
        """Connect to the agent and fetch its card."""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0)
        )

        agent_card_url = f"{self.agent_url}/.well-known/agent-card.json"
        try:
            response = await self.http_client.get(agent_card_url)
            response.raise_for_status()
            card_dict = response.json()
            self.agent_card = types.AgentCard(**card_dict)

            client_config = ClientConfig(
                streaming=self.prefer_streaming,
                polling=not self.prefer_streaming,
                httpx_client=self.http_client
            )

            factory = ClientFactory(config=client_config)
            self.a2a_client = factory.create(card=self.agent_card)

            logger.info(f"[CLIENT] Connected to agent: {self.agent_card.name} v{self.agent_card.version}")
            return True

        except Exception as e:
            logger.error(f"[CLIENT] Failed to connect: {str(e)}")
            raise ConnectionError(f"Failed to connect to agent: {str(e)}")

    async def close(self):
        """Close the connection."""
        if self.http_client:
            try:
                await self.http_client.aclose()
            except RuntimeError as e:
                # Suppress "asynchronous generator is already running" errors during cleanup
                if "asynchronous generator is already running" not in str(e):
                    raise

    async def check_health(self) -> Dict[str, Any]:
        """
        Check agent server health (non-blocking).

        Returns:
            Dictionary with health status information:
            - available: bool - Whether agent is available
            - status: str - "online", "degraded", "timeout", or "offline"
            - active_jobs: int - Number of active jobs (if available)
            - redis: str - Redis connection status (if available)
            - error: str - Error message (if failed)
        """
        try:
            if not self.http_client:
                self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))

            # Use /health/ready endpoint for detailed status
            response = await asyncio.wait_for(
                self.http_client.get(f"{self.agent_url}/health/ready"),
                timeout=5.0
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "available": True,
                    "status": "online",
                    "active_jobs": data.get("active_jobs", 0),
                    "redis": data.get("redis", "unknown"),
                    "timestamp": data.get("timestamp", "")
                }
            else:
                # Server responded but not ready (503)
                data = response.json()
                return {
                    "available": False,
                    "status": "degraded",
                    "active_jobs": data.get("active_jobs"),
                    "redis": data.get("redis", "disconnected"),
                    "details": data
                }

        except asyncio.TimeoutError:
            logger.warning("[CLIENT] Health check timeout")
            return {
                "available": False,
                "status": "timeout",
                "error": "Health check timed out after 5 seconds"
            }
        except Exception as e:
            logger.error(f"[CLIENT] Health check failed: {str(e)}")
            return {
                "available": False,
                "status": "offline",
                "error": str(e)
            }

    async def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        use_gpt4: bool = False,
        enable_streaming: bool = True,
        confidence_threshold: float = 0.7,
        context_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a policy document with streaming updates.

        Args:
            file_bytes: PDF file bytes
            filename: Original filename
            use_gpt4: Use GPT-4 for extraction
            enable_streaming: Enable streaming responses
            confidence_threshold: Minimum confidence threshold
            context_id: Optional conversation context ID

        Yields:
            Processing updates and final results
        """
        if not self.a2a_client:
            await self.connect()

        if not context_id:
            import uuid
            context_id = str(uuid.uuid4())

        pdf_base64 = base64.b64encode(file_bytes).decode()

        logger.info(f"[CLIENT] Processing: {filename} ({len(file_bytes)} bytes)")
        logger.info(f"[CLIENT] Context ID: {context_id}")

        try:
            # Build parameters JSON for the agent to extract
            import json
            parameters_json = json.dumps({
                "document_base64": pdf_base64,
                "use_gpt4": use_gpt4,
                "enable_streaming": enable_streaming,
                "confidence_threshold": confidence_threshold
            })
            
            message = types.Message(
                message_id=f"msg_{context_id}",
                role=types.Role.user,
                parts=[
                    types.Part(root=types.FilePart(
                        file=types.FileWithBytes(
                            bytes=pdf_base64,
                            name=filename,
                            mime_type="application/pdf"
                        )
                    )),
                    types.Part(root=types.TextPart(
                        text=parameters_json  # Send parameters as JSON string
                    ))
                ],
                context_id=context_id
            )

            logger.info("[CLIENT] Sending request to agent...")
            
            task_id = None
            final_results = None
            
            async for event in self.a2a_client.send_message(message):
                # A2A SDK returns events as (task, update) tuples
                if isinstance(event, tuple):
                    task, update = event
                    task_id = task.id
                    
                    if isinstance(update, types.TaskStatusUpdateEvent):
                        status = update.status.state.value
                        logger.info(f"[CLIENT] Status: {status}")
                        
                        status_text = ""
                        if update.status.message and update.status.message.parts:
                            for part in update.status.message.parts:
                                if isinstance(part.root, types.TextPart):
                                    status_text = part.root.text
                                    break
                        
                        yield {
                            "type": "status",
                            "status": status,
                            "message": status_text,
                            "task_id": task_id,
                            "final": update.final
                        }
                        
                    elif isinstance(update, types.TaskArtifactUpdateEvent):
                        logger.info(f"[CLIENT] Artifact: {update.artifact.name}")
                        
                        if update.artifact.parts:
                            for part in update.artifact.parts:
                                if isinstance(part.root, types.DataPart):
                                    final_results = part.root.data
                                    logger.info(f"[CLIENT] Results received")
                        
                        yield {
                            "type": "artifact",
                            "artifact_id": update.artifact.artifact_id,
                            "name": update.artifact.name,
                            "data": final_results,
                            "task_id": task_id,
                            "last_chunk": update.last_chunk
                        }
                        
                        if update.last_chunk and final_results:
                            yield {
                                "type": "complete",
                                "status": "completed",
                                "job_id": task_id,
                                "results": final_results,
                                "message": "Processing completed"
                            }

        except Exception as e:
            logger.error(f"[CLIENT] Error: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "status": "failed",
                "message": f"Processing failed: {str(e)}"
            }


# Global client instance
_client: Optional[PolicyProcessingClient] = None


def get_client() -> PolicyProcessingClient:
    """Get or create the global client instance."""
    global _client
    if _client is None:
        _client = PolicyProcessingClient()
    return _client