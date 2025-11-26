"""
A2A Client for Streamlit Application.
import httpx
import json
import uuid
import base64
from typing import Dict, Any, Optional
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


class A2AClient:
    """
    Simplified client for communicating with the A2A server.

    Uses JSON-RPC 2.0 protocol to send messages to the single unified endpoint.
    """

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 300.0):
        """
        Initialize A2A client.

        Args:
            base_url: Base URL of the A2A server
            timeout: Request timeout in seconds (default: 300s for long processing)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())
        logger.info(f"A2A Client initialized: {self.base_url}")

    async def check_health(self) -> bool:
        """
        Check if the A2A server is healthy and responding.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_agent_card(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the agent card from the A2A server.

        Returns:
            Agent card dictionary or None if failed
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/.well-known/agent-card",
                    timeout=10.0
                )
                response.raise_for_status()
                card = response.json()
                logger.info(f"Retrieved agent card: {card.get('name', 'Unknown')}")
                return card
        except Exception as e:
            logger.error(f"Failed to get agent card: {e}")
            return None

    async def process_document(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        use_gpt4: bool = False,
        enable_streaming: bool = True,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process a policy document.

        Args:
            pdf_bytes: PDF file bytes
            filename: Original filename
            use_gpt4: Whether to use GPT-4 for extraction
            enable_streaming: Enable streaming progress updates
            confidence_threshold: Minimum confidence threshold

        Returns:
            Processing result with job_id and status
        """
        try:
            logger.info(f"Processing document: {filename} ({len(pdf_bytes)} bytes)")

            # Encode PDF as base64
            document_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

            # Build parameters
            parameters = {
                "document_base64": document_base64,
                "use_gpt4": use_gpt4,
                "enable_streaming": enable_streaming,
                "confidence_threshold": confidence_threshold
            }

            # Send request
            result = await self._send_jsonrpc_request(
                skill_id="process_policy",
                message_text=f"Process policy document: {filename}",
                parameters=parameters
            )

            logger.info(f"Document processing response: {json.dumps(result, default=str)[:200]}")
            return result

        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            return {
                "status": "failed",
                "message": f"Error: {str(e)}"
            }

    async def get_results(self, job_id: str) -> Dict[str, Any]:
        """
        Get processing results for a job.

        Args:
            job_id: The job ID to retrieve results for

        Returns:
            Processing results
        """
        try:
            logger.info(f"Getting results for job {job_id}")

            parameters = {"job_id": job_id}

            result = await self._send_jsonrpc_request(
                skill_id="process_policy",
                message_text=f"Get results for job {job_id}",
                parameters=parameters
            )

            return result

        except Exception as e:
            logger.error(f"Error getting results: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }

    async def _send_jsonrpc_request(
        self,
        skill_id: str,
        message_text: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a JSON-RPC 2.0 request to the A2A server.

        Args:
            skill_id: Skill identifier
            message_text: Message text
            parameters: Parameters for the skill

        Returns:
            Response data

        Raises:
            Exception: If the request fails
        """
        try:
            # Generate IDs
            context_id = str(uuid.uuid4())
            task_id = str(uuid.uuid4())
            message_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())

            # Build A2A message payload
            message_payload = {
                "messageId": message_id,
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": message_text
                    }
                ],
                "taskId": task_id,
                "contextId": context_id
            }

            # Build metadata
            metadata = {
                "skill_id": skill_id,
                "parameters": parameters or {}
            }

            # Build JSON-RPC request
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "method": "sendMessage",
                "params": {
                    "message": message_payload,
                    "metadata": metadata
                },
                "id": request_id
            }

            logger.debug(f"Sending JSON-RPC request: {json.dumps(jsonrpc_request, default=str)[:500]}")

            # Send request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/jsonrpc",
                    json=jsonrpc_request,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()

                result = response.json()
                logger.debug(f"Response: {json.dumps(result, default=str)[:500]}")

                # Check for JSON-RPC error
                if "error" in result:
                    error = result["error"]
                    error_msg = error.get("message", "Unknown error")
                    logger.error(f"JSON-RPC error: {error_msg}")
                    raise Exception(f"A2A Error: {error_msg}")

                # Extract the result
                if "result" in result:
                    return self._parse_a2a_response(result["result"])
                else:
                    raise Exception("Invalid A2A response format")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise Exception(f"Failed to connect to A2A server: {str(e)}")
        except Exception as e:
            logger.error(f"Error sending JSON-RPC request: {e}")
            raise

    def _parse_a2a_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the A2A response and extract relevant data.

        Args:
            result: The result from the JSON-RPC response

        Returns:
            Parsed response data
        """
        try:
            # The result typically contains messages or task status
            if "messages" in result:
                messages = result["messages"]
                if messages and len(messages) > 0:
                    # Get the last message from the agent
                    last_message = messages[-1]
                    if "parts" in last_message:
                        # Extract text from parts
                        text_parts = [
                            part.get("text", "") for part in last_message["parts"]
                            if part.get("kind") == "text"
                        ]
                        response_text = "\n".join(text_parts)

                        # Try to parse as JSON if it looks like JSON
                        if response_text.strip().startswith("{"):
                            try:
                                return json.loads(response_text)
                            except json.JSONDecodeError:
                                pass

                        # Return as text response
                        return {"response": response_text, "raw": result}

            # If task_id is present, return it
            if "taskId" in result:
                return {"task_id": result["taskId"], "raw": result}

            # Return raw result if we can't parse it
            return result

        except Exception as e:
            logger.error(f"Error parsing A2A response: {e}")
            return result


# Synchronous wrapper for use in Streamlit
class A2AClientSync:
    """Synchronous wrapper for A2AClient for use in Streamlit."""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 300.0):
        """Initialize sync client."""
        self.client = A2AClient(base_url, timeout)

    def check_health(self) -> bool:
        """Check server health (sync)."""
        import asyncio
        return asyncio.run(self.client.check_health())

    def get_agent_card(self) -> Optional[Dict[str, Any]]:
        """Get agent card (sync)."""
        import asyncio
        return asyncio.run(self.client.get_agent_card())

    def process_document(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        use_gpt4: bool = False,
        enable_streaming: bool = True,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Process document (sync)."""
        import asyncio
        return asyncio.run(
            self.client.process_document(
                pdf_bytes, filename, use_gpt4, enable_streaming, confidence_threshold
            )
        )

    def get_results(self, job_id: str) -> Dict[str, Any]:
        """Get results (sync)."""
        import asyncio
        return asyncio.run(self.client.get_results(job_id))
