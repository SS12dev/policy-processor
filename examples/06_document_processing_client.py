"""
Document Processing Client with File Upload
This example demonstrates:
- Sending files via FileWithBytes (base64 encoded)
- Sending files via FileWithUri (URL references)
- Sending multiple files in one message
- Handling file size limits
- Receiving processed files as artifacts
"""

import asyncio
import base64
from pathlib import Path
import httpx
from a2a import types
from a2a.client import Client, ClientConfig, JsonRpcHttpClientTransport


async def send_text_file():
    """
    Example: Upload a text file using FileWithBytes.
    Best for small files that can be embedded in the request.
    """
    print("="*80)
    print("Sending Text File to Document Processing Agent")
    print("="*80)

    # Create a sample text file content
    sample_text = """
    This is a sample document for processing.
    It contains multiple lines of text.
    The agent will analyze this content and provide statistics.

    Some additional content:
    - Bullet points
    - Multiple paragraphs
    - Various text elements
    """

    # Encode as base64
    file_bytes_base64 = base64.b64encode(sample_text.encode('utf-8')).decode('utf-8')

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        # Connect to document processing agent
        card_response = await http_client.get(
            "http://localhost:8002/.well-known/agent-card.json"
        )
        agent_card = types.AgentCard(**card_response.json())

        print(f"Connected to: {agent_card.name}")

        transport = JsonRpcHttpClientTransport(
            http_client=http_client,
            base_url="http://localhost:8002/rpc"
        )

        client = Client.create(
            card=agent_card,
            config=ClientConfig(prefer_streaming=True),
            transport=transport
        )

        # Create message with file attachment
        user_message = types.Message(
            message_id="doc_msg_001",
            role=types.Role.USER,
            parts=[
                # Text part with instructions
                types.Part(root=types.TextPart(
                    text="Please analyze this document and provide statistics."
                )),
                # File part with the document
                types.Part(root=types.FilePart(
                    file=types.FileWithBytes(
                        bytes=file_bytes_base64,
                        name="sample_document.txt",
                        mime_type="text/plain"
                    ),
                    metadata={
                        "source": "user_upload",
                        "category": "text_analysis"
                    }
                ))
            ]
        )

        print("\nSending text file for analysis...")
        print(f"File size: {len(sample_text)} characters\n")

        # Send and receive responses
        async for event in client.send_message(user_message):
            if isinstance(event, tuple):
                task, update = event
                if isinstance(update, types.TaskStatusUpdateEvent):
                    print(f"[Status] {update.status.message}")

                elif isinstance(update, types.TaskArtifactUpdateEvent):
                    print(f"\n[Artifact Received] {update.artifact.name}")
                    print(f"Description: {update.artifact.description}")

                    # Extract and display artifact content
                    for part in update.artifact.parts:
                        if isinstance(part.root, types.TextPart):
                            print(f"\nContent:\n{part.root.text}")
                        elif isinstance(part.root, types.FilePart):
                            file_part = part.root
                            if isinstance(file_part.file, types.FileWithBytes):
                                decoded = base64.b64decode(file_part.file.bytes).decode('utf-8')
                                print(f"\nProcessed File Content:\n{decoded}")

            elif isinstance(event, types.Message):
                print(f"\n[Response]")
                for part in event.parts:
                    if isinstance(part.root, types.TextPart):
                        print(part.root.text)

        print("\n\nText file processing complete!")


async def send_file_by_uri():
    """
    Example: Reference a file by URI instead of embedding it.
    Best for large files or files already hosted elsewhere.
    """
    print("\n\n" + "="*80)
    print("Sending File by URI Reference")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        card_response = await http_client.get(
            "http://localhost:8002/.well-known/agent-card.json"
        )
        agent_card = types.AgentCard(**card_response.json())

        transport = JsonRpcHttpClientTransport(
            http_client=http_client,
            base_url="http://localhost:8002/rpc"
        )

        client = Client.create(
            card=agent_card,
            config=ClientConfig(prefer_streaming=True),
            transport=transport
        )

        # Create message with file reference
        user_message = types.Message(
            message_id="doc_msg_002",
            role=types.Role.USER,
            parts=[
                types.Part(root=types.TextPart(
                    text="Please analyze the document at the provided URI."
                )),
                types.Part(root=types.FilePart(
                    file=types.FileWithUri(
                        uri="https://example.com/documents/report.pdf",
                        name="report.pdf",
                        mime_type="application/pdf"
                    ),
                    metadata={
                        "source": "external_storage",
                        "category": "pdf_document"
                    }
                ))
            ]
        )

        print("\nSending file URI for processing...")
        print("URI: https://example.com/documents/report.pdf\n")

        async for event in client.send_message(user_message):
            if isinstance(event, tuple):
                task, update = event
                if isinstance(update, types.TaskStatusUpdateEvent):
                    print(f"[Status] {update.status.message}")

            elif isinstance(event, types.Message):
                for part in event.parts:
                    if isinstance(part.root, types.TextPart):
                        print(f"\n[Response]\n{part.root.text}")

        print("\n\nURI file processing complete!")


async def send_multiple_files():
    """
    Example: Send multiple files in a single message.
    """
    print("\n\n" + "="*80)
    print("Sending Multiple Files")
    print("="*80)

    # Create multiple sample files
    file1_content = "First document content with some text."
    file2_content = "Second document with different content."
    file3_content = '{"name": "data.json", "type": "json", "records": 100}'

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        card_response = await http_client.get(
            "http://localhost:8002/.well-known/agent-card.json"
        )
        agent_card = types.AgentCard(**card_response.json())

        transport = JsonRpcHttpClientTransport(
            http_client=http_client,
            base_url="http://localhost:8002/rpc"
        )

        client = Client.create(
            card=agent_card,
            config=ClientConfig(prefer_streaming=True),
            transport=transport
        )

        # Create message with multiple parts
        parts = [
            types.Part(root=types.TextPart(
                text="Please analyze these multiple documents."
            )),
            # File 1
            types.Part(root=types.FilePart(
                file=types.FileWithBytes(
                    bytes=base64.b64encode(file1_content.encode()).decode(),
                    name="document1.txt",
                    mime_type="text/plain"
                )
            )),
            # File 2
            types.Part(root=types.FilePart(
                file=types.FileWithBytes(
                    bytes=base64.b64encode(file2_content.encode()).decode(),
                    name="document2.txt",
                    mime_type="text/plain"
                )
            )),
            # Structured data
            types.Part(root=types.DataPart(
                data={
                    "metadata": {
                        "batch_id": "batch_001",
                        "total_files": 2,
                        "timestamp": "2025-01-01T00:00:00Z"
                    }
                }
            ))
        ]

        user_message = types.Message(
            message_id="doc_msg_003",
            role=types.Role.USER,
            parts=parts
        )

        print("\nSending 2 text files + structured data...")
        print("Files: document1.txt, document2.txt\n")

        async for event in client.send_message(user_message):
            if isinstance(event, tuple):
                task, update = event
                if isinstance(update, types.TaskStatusUpdateEvent):
                    print(f"[Status] {update.status.message}")
                elif isinstance(update, types.TaskArtifactUpdateEvent):
                    print(f"[Artifact] {update.artifact.name}")

            elif isinstance(event, types.Message):
                for part in event.parts:
                    if isinstance(part.root, types.TextPart):
                        print(f"\n[Response]\n{part.root.text}")

        print("\n\nMultiple file processing complete!")


async def demonstrate_file_size_considerations():
    """
    Example: Demonstrate considerations for file size limits.
    """
    print("\n\n" + "="*80)
    print("File Size Considerations")
    print("="*80)

    print("""
File Size Guidelines for A2A Protocol:

1. FileWithBytes (Base64 Encoded):
   - Recommended: < 10 MB
   - Maximum (typical): 10-100 MB depending on server configuration
   - Base64 encoding increases size by ~33%
   - Suitable for: Small documents, images, config files

2. FileWithUri (URI Reference):
   - No inherent size limit in the protocol
   - Limited only by the hosting service
   - Agent must download from the URI
   - Suitable for: Large files, videos, datasets, archives

3. HTTP Configuration Limits:
   - FastAPI default: Unlimited (but limited by Uvicorn/Gunicorn)
   - Uvicorn default: 16MB (configurable)
   - Can be configured via: --limit-max-requests-size
   - Example: uvicorn app:app --limit-max-requests-size 104857600  # 100MB

4. Optimization Strategies:
   - Use FileWithUri for files > 10MB
   - Compress files before encoding (if using FileWithBytes)
   - Stream large responses using artifacts with chunking
   - Consider splitting large batches across multiple messages

5. Configuring Server Limits:
    """)

    # Example server configuration code
    print("""
    # In your server code:
    import uvicorn
    from a2a.server import A2AFastAPIApplication

    app = create_your_app()

    # Configure with larger limits
    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=8000,
        limit_max_requests=1000,
        timeout_keep_alive=5,
        # Size limits (in bytes)
        # Note: This is a uvicorn configuration
    )

    # For production with Gunicorn:
    # gunicorn app:app -k uvicorn.workers.UvicornWorker \\
    #   --limit-request-line 8190 \\
    #   --limit-request-fields 100 \\
    #   --limit-request-field_size 8190
    """)


if __name__ == "__main__":
    print("Make sure the document processing server is running on localhost:8002")
    print("You can start it with: python 05_document_processing_agent.py\n")

    # Run examples
    asyncio.run(send_text_file())
    asyncio.run(send_file_by_uri())
    asyncio.run(send_multiple_files())
    asyncio.run(demonstrate_file_size_considerations())
