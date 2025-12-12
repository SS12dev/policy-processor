"""
A2A Client Module
Handles communication with the A2A agent.
"""

import asyncio
import hashlib
import base64
from typing import Optional, AsyncIterator, Dict, Any
import httpx

from a2a import types
from a2a.client import Client, ClientConfig, ClientFactory

from settings import settings
from database import get_database


class DocumentAnalysisClient:
    """Client for document analysis agent."""

    def __init__(self):
        """Initialize the client."""
        self.agent_url = settings.agent_url
        self.timeout = settings.agent_timeout_seconds
        self.prefer_streaming = settings.prefer_streaming
        self.database = get_database()

        self.http_client: Optional[httpx.AsyncClient] = None
        self.a2a_client: Optional[Client] = None
        self.agent_card: Optional[types.AgentCard] = None

    async def connect(self):
        """Connect to the agent and fetch its card."""
        # Create HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0)
        )

        # Fetch agent card
        agent_card_url = f"{self.agent_url}/.well-known/agent-card.json"
        try:
            response = await self.http_client.get(agent_card_url)
            response.raise_for_status()
            card_dict = response.json()
            self.agent_card = types.AgentCard(**card_dict)

            # Create A2A client via factory
            client_config = ClientConfig(
                streaming=self.prefer_streaming,
                polling=not self.prefer_streaming,  # Enable polling if not streaming
                httpx_client=self.http_client
            )

            factory = ClientFactory(config=client_config)
            self.a2a_client = factory.create(card=self.agent_card)

            return True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to agent: {str(e)}")

    async def close(self):
        """Close the connection."""
        if self.http_client:
            await self.http_client.aclose()

    async def analyze_document(
        self,
        file_bytes: bytes,
        filename: str,
        context_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Analyze a document.

        Args:
            file_bytes: PDF file bytes
            filename: Original filename
            context_id: Optional conversation context ID

        Yields:
            Analysis updates and final results
        """
        if not self.a2a_client:
            await self.connect()

        # Calculate checksum
        checksum = hashlib.sha256(file_bytes).hexdigest()

        # Store document
        document_id = self.database.store_document(
            filename=filename,
            file_data=file_bytes,
            file_size=len(file_bytes),
            checksum=checksum
        )

        # Create request
        if not context_id:
            context_id = f"ctx_{document_id}_{int(asyncio.get_event_loop().time())}"

        request_id = self.database.create_request(
            document_id=document_id,
            context_id=context_id,
            streaming_enabled=self.prefer_streaming
        )

        # Encode PDF as base64
        pdf_base64 = base64.b64encode(file_bytes).decode()

        # Create A2A message
        message = types.Message(
            message_id=f"msg_{request_id}",
            role=types.Role.user,
            parts=[
                types.Part(root=types.FilePart(
                    file=types.FileWithBytes(
                        bytes=pdf_base64,
                        name=filename,
                        mime_type="application/pdf"
                    )
                ))
            ]
        )

        start_time = asyncio.get_event_loop().time()
        final_response = None
        task_id = None

        try:
            # Send message and process events
            async for event in self.a2a_client.send_message(message):
                if isinstance(event, tuple):
                    task, update = event
                    task_id = task.id

                    # Update request with task ID
                    if not task_id:
                        self.database.update_request_status(
                            request_id, "processing", task_id
                        )

                    if isinstance(update, types.TaskStatusUpdateEvent):
                        # Extract text from Message object
                        message_text = "Processing..."
                        if update.status.message:
                            for part in update.status.message.parts:
                                if isinstance(part.root, types.TextPart):
                                    message_text = part.root.text
                                    break

                        yield {
                            "type": "status",
                            "status": update.status.state.value,
                            "message": message_text,
                            "final": update.final
                        }

                        if update.final:
                            if update.status.state == types.TaskState.completed:
                                self.database.update_request_status(request_id, "completed")
                            elif update.status.state == types.TaskState.failed:
                                self.database.update_request_status(request_id, "failed")

                    elif isinstance(update, types.TaskArtifactUpdateEvent):
                        artifact = update.artifact
                        for part in artifact.parts:
                            if isinstance(part.root, types.DataPart):
                                final_response = part.root.data
                                yield {
                                    "type": "artifact",
                                    "artifact_id": artifact.artifact_id,
                                    "name": artifact.name,
                                    "data": part.root.data
                                }

                elif isinstance(event, types.Message):
                    for part in event.parts:
                        if isinstance(part.root, types.TextPart):
                            yield {
                                "type": "message",
                                "text": part.root.text
                            }

            # Store response
            processing_time = asyncio.get_event_loop().time() - start_time

            if final_response:
                self.database.store_response(
                    request_id=request_id,
                    response_data=final_response,
                    processing_time=processing_time,
                    status="success"
                )

                yield {
                    "type": "complete",
                    "response": final_response,
                    "processing_time": processing_time
                }
            else:
                self.database.store_response(
                    request_id=request_id,
                    response_data={},
                    processing_time=processing_time,
                    status="error",
                    error_message="No response received"
                )

        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            error_msg = str(e)

            self.database.store_response(
                request_id=request_id,
                response_data={},
                processing_time=processing_time,
                status="error",
                error_message=error_msg
            )

            self.database.update_request_status(request_id, "failed")

            yield {
                "type": "error",
                "error": error_msg
            }


# Global client instance
_client: Optional[DocumentAnalysisClient] = None


def get_client() -> DocumentAnalysisClient:
    """Get or create the global client instance."""
    global _client
    if _client is None:
        _client = DocumentAnalysisClient()
    return _client
