"""
Document Processing Agent with File Handling
This example demonstrates:
- Receiving files via FileWithBytes (base64 encoded)
- Receiving files via FileWithUri (URL references)
- Processing different file types (text, PDF, images)
- Returning processed files as artifacts
- Handling large file attachments
"""

import asyncio
import base64
from typing import AsyncIterator
from pathlib import Path
from a2a import types
from a2a.server import (
    AgentExecutor,
    RequestContext,
    A2AFastAPIApplication
)


class DocumentProcessingAgent(AgentExecutor):
    """
    An agent that processes documents and files.
    Demonstrates handling both embedded files (base64) and referenced files (URIs).
    """

    async def execute(
        self,
        request_context: RequestContext
    ) -> AsyncIterator[types.TaskStatusUpdateEvent | types.TaskArtifactUpdateEvent | types.Message]:
        """
        Process messages that may contain file attachments.
        """
        user_message = request_context.user_message

        print(f"[Doc Agent] Processing message with {len(user_message.parts)} parts")

        # Initial status
        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(
                state=types.TaskState.WORKING,
                message="Analyzing uploaded files..."
            ),
            final=False
        )

        response_parts = []
        processed_files = []

        # Process each part of the message
        for i, part in enumerate(user_message.parts):
            if isinstance(part.root, types.TextPart):
                # Handle text
                text = part.root.text
                print(f"[Doc Agent] Text part: {text[:100]}")
                response_parts.append(f"Text received: {text}")

            elif isinstance(part.root, types.FilePart):
                # Handle file
                file_part = part.root
                file_info = await self._process_file(file_part, i)
                processed_files.append(file_info)
                response_parts.append(file_info["summary"])

                # Yield progress update
                yield types.TaskStatusUpdateEvent(
                    task_id=request_context.task_id,
                    context_id=request_context.context_id,
                    status=types.TaskStatus(
                        state=types.TaskState.WORKING,
                        message=f"Processed file {i+1}: {file_info['name']}"
                    ),
                    final=False
                )

            elif isinstance(part.root, types.DataPart):
                # Handle structured data
                data = part.root.data
                print(f"[Doc Agent] Data part with {len(data)} fields")
                response_parts.append(f"Data structure received with fields: {', '.join(data.keys())}")

        # Create response message
        response_text = "\n\n".join(response_parts)

        yield types.Message(
            message_id=f"msg_{request_context.task_id}_response",
            role=types.Role.AGENT,
            parts=[
                types.Part(root=types.TextPart(text=response_text))
            ]
        )

        # Create artifacts for processed files
        for file_info in processed_files:
            # Return processed file as an artifact
            artifact_parts = [
                types.Part(root=types.TextPart(
                    text=f"Processed version of: {file_info['name']}\n\n{file_info['analysis']}"
                ))
            ]

            # If we generated a new file, include it
            if file_info.get("output_file"):
                artifact_parts.append(
                    types.Part(root=types.FilePart(
                        file=file_info["output_file"]
                    ))
                )

            yield types.TaskArtifactUpdateEvent(
                task_id=request_context.task_id,
                context_id=request_context.context_id,
                artifact=types.Artifact(
                    artifact_id=f"artifact_{request_context.task_id}_{file_info['index']}",
                    name=f"processed_{file_info['name']}",
                    description=f"Analysis of {file_info['name']}",
                    parts=artifact_parts
                ),
                last_chunk=True
            )

        # Final status
        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(
                state=types.TaskState.COMPLETED,
                message=f"Processed {len(processed_files)} files successfully"
            ),
            final=True
        )

    async def _process_file(self, file_part: types.FilePart, index: int) -> dict:
        """
        Process a file part and return analysis.

        Args:
            file_part: The file part to process
            index: Index of this file in the message

        Returns:
            Dictionary with file information and analysis
        """
        file_info = {
            "index": index,
            "name": "unknown",
            "type": "unknown",
            "size": 0,
            "summary": "",
            "analysis": ""
        }

        # Handle FileWithBytes (embedded file)
        if isinstance(file_part.file, types.FileWithBytes):
            file_with_bytes = file_part.file

            # Decode base64 content
            try:
                file_bytes = base64.b64decode(file_with_bytes.bytes)
                file_info["size"] = len(file_bytes)
                file_info["name"] = file_with_bytes.name or "embedded_file"
                file_info["type"] = file_with_bytes.mime_type or "application/octet-stream"

                # Analyze content based on type
                if file_with_bytes.mime_type and "text" in file_with_bytes.mime_type:
                    # Text file analysis
                    text_content = file_bytes.decode('utf-8', errors='ignore')
                    word_count = len(text_content.split())
                    line_count = len(text_content.splitlines())

                    file_info["analysis"] = (
                        f"Text file analysis:\n"
                        f"- Lines: {line_count}\n"
                        f"- Words: {word_count}\n"
                        f"- Characters: {len(text_content)}\n"
                        f"- First 100 chars: {text_content[:100]}"
                    )

                elif file_with_bytes.mime_type and "image" in file_with_bytes.mime_type:
                    # Image file analysis
                    file_info["analysis"] = (
                        f"Image file detected:\n"
                        f"- Size: {len(file_bytes)} bytes\n"
                        f"- Type: {file_with_bytes.mime_type}\n"
                        f"- Note: Advanced image processing would go here"
                    )

                else:
                    # Generic file
                    file_info["analysis"] = (
                        f"Binary file:\n"
                        f"- Size: {len(file_bytes)} bytes\n"
                        f"- Type: {file_with_bytes.mime_type}"
                    )

                # Create output file (example: add metadata)
                output_content = (
                    f"=== Processed File Metadata ===\n"
                    f"Original Name: {file_info['name']}\n"
                    f"Size: {file_info['size']} bytes\n"
                    f"Type: {file_info['type']}\n"
                    f"=== Analysis ===\n"
                    f"{file_info['analysis']}\n"
                )

                file_info["output_file"] = types.FileWithBytes(
                    bytes=base64.b64encode(output_content.encode()).decode(),
                    name=f"analysis_{file_info['name']}.txt",
                    mime_type="text/plain"
                )

            except Exception as e:
                file_info["analysis"] = f"Error processing file: {str(e)}"

            file_info["summary"] = (
                f"File '{file_info['name']}' ({file_info['type']}): "
                f"{file_info['size']} bytes - Base64 encoded"
            )

        # Handle FileWithUri (referenced file)
        elif isinstance(file_part.file, types.FileWithUri):
            file_with_uri = file_part.file

            file_info["name"] = file_with_uri.name or "referenced_file"
            file_info["type"] = file_with_uri.mime_type or "unknown"

            file_info["analysis"] = (
                f"File available at URI:\n"
                f"- URI: {file_with_uri.uri}\n"
                f"- Type: {file_with_uri.mime_type}\n"
                f"- Note: To process this file, the agent would need to:\n"
                f"  1. Download from the URI\n"
                f"  2. Verify content type\n"
                f"  3. Process according to file type\n"
                f"  4. Handle any authentication/authorization\n"
            )

            file_info["summary"] = (
                f"File '{file_info['name']}' ({file_info['type']}): "
                f"Available at {file_with_uri.uri}"
            )

        return file_info


def create_document_processing_server():
    """
    Create a document processing A2A server.
    """
    agent_card = types.AgentCard(
        name="Document Processing Agent",
        description="An agent that processes and analyzes documents and files",
        url="http://localhost:8002",
        version="1.0.0",
        default_input_modes=["text", "file", "data"],
        default_output_modes=["text", "file", "artifact"],
        capabilities=types.AgentCapabilities(
            streaming=True,
            push_notifications=False
        ),
        skills=[
            types.AgentSkill(
                name="document_analysis",
                description="Analyze text documents, count words, extract metadata",
                input_modes=["text", "file"],
                output_modes=["text", "artifact"]
            ),
            types.AgentSkill(
                name="image_processing",
                description="Basic image analysis and metadata extraction",
                input_modes=["file"],
                output_modes=["text", "artifact"]
            ),
            types.AgentSkill(
                name="data_processing",
                description="Process structured data (JSON)",
                input_modes=["data"],
                output_modes=["text", "data"]
            )
        ]
    )

    agent = DocumentProcessingAgent()

    app = A2AFastAPIApplication(
        agent_card=agent_card,
        agent_executor=agent
    )

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_document_processing_server()

    print("="*80)
    print("Starting Document Processing A2A Agent Server")
    print("="*80)
    print("Server: http://localhost:8002")
    print("Agent card: http://localhost:8002/.well-known/agent-card.json")
    print("RPC endpoint: http://localhost:8002/rpc")
    print("="*80)
    print("\nSupported file types:")
    print("  - Text files (via FileWithBytes or FileWithUri)")
    print("  - Images (basic analysis)")
    print("  - Structured data (JSON via DataPart)")
    print("="*80)
    print("\nNote on file size limits:")
    print("  - FileWithBytes: Limited by HTTP request size (typically 10MB-100MB)")
    print("  - FileWithUri: No inherent size limit (file stays at source)")
    print("  - For large files, prefer FileWithUri over FileWithBytes")
    print("="*80)

    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
