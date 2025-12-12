# Configuration Guide: Message Size Limits and Server Configuration

## Overview

This guide covers all aspects of configuring A2A servers and clients, with special focus on message size limits, payload configurations, and performance tuning.

## Message Size Limits

### Protocol-Level Limits

The A2A protocol specification **does not define hard limits** for message sizes. However, practical limits exist based on:

1. **Transport layer** (HTTP, gRPC)
2. **Server configuration** (Uvicorn, Gunicorn)
3. **Memory constraints**
4. **Network bandwidth**

### Typical Size Limits by Component

| Component | Default Limit | Configurable | Recommended Max |
|-----------|---------------|--------------|-----------------|
| HTTP Request Body | Varies | Yes | 10-100 MB |
| Base64 Encoded File | ~1.33x file size | N/A | < 10 MB |
| FileWithBytes | Depends on HTTP | Yes | 10 MB |
| FileWithUri | No limit (reference) | N/A | Unlimited |
| JSON-RPC Message | Depends on HTTP | Yes | 10 MB |
| WebSocket Frame | 16 MB (typical) | Yes | 16 MB |

### Understanding Base64 Overhead

When using `FileWithBytes`, files are base64-encoded, which increases size:

```python
original_size = 9 MB
base64_size = original_size * 1.33 = ~12 MB
```

This means a 9MB file becomes ~12MB when base64-encoded, potentially exceeding a 10MB limit.

## Server Configuration

### 1. Uvicorn Configuration

Uvicorn is the recommended ASGI server for A2A FastAPI applications.

#### Basic Configuration

```python
import uvicorn
from a2a.server import A2AFastAPIApplication

app = create_your_a2a_app()

# Run with custom configuration
uvicorn.run(
    app.app,  # The FastAPI app
    host="0.0.0.0",
    port=8000,
    log_level="info",
    # Worker configuration
    workers=4,  # Number of worker processes
    # Connection limits
    limit_concurrency=1000,  # Max concurrent connections
    limit_max_requests=10000,  # Max requests before worker restart
    # Timeouts
    timeout_keep_alive=5,  # Keep-alive timeout
    timeout_graceful_shutdown=30,  # Graceful shutdown timeout
)
```

#### Handling Large Requests

Uvicorn doesn't have a direct `--limit-max-request-size` parameter. Size limits are typically handled at the application level:

```python
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 100 * 1024 * 1024):  # 100MB default
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request too large. Maximum size: {self.max_size} bytes"
            )
        return await call_next(request)

# Add to your FastAPI app
app = FastAPI()
app.add_middleware(RequestSizeLimitMiddleware, max_size=100 * 1024 * 1024)
```

### 2. Gunicorn Configuration

For production deployments, Gunicorn with Uvicorn workers is recommended:

```bash
gunicorn app:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  --access-logfile - \
  --error-logfile -
```

#### Gunicorn Request Size Limits

```bash
gunicorn app:app \
  -k uvicorn.workers.UvicornWorker \
  --limit-request-line 8190 \      # Max size of request line
  --limit-request-fields 100 \     # Max number of headers
  --limit-request-field_size 8190  # Max size of header
```

**Note**: These limits apply to HTTP headers, not body size. Body size must be handled by middleware.

### 3. Complete Server Setup Example

```python
# server_config.py
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from a2a.server import A2AFastAPIApplication
from a2a import types

# Configuration
class ServerConfig:
    MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100 MB
    MAX_RESPONSE_SIZE = 100 * 1024 * 1024  # 100 MB
    REQUEST_TIMEOUT = 120  # seconds
    MAX_CONNECTIONS = 1000
    WORKERS = 4

# Size limit middleware
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            if int(content_length) > ServerConfig.MAX_REQUEST_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request Too Large",
                        "max_size_mb": ServerConfig.MAX_REQUEST_SIZE / (1024 * 1024),
                        "received_size_mb": int(content_length) / (1024 * 1024)
                    }
                )
        return await call_next(request)

# Timeout middleware
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        import asyncio
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=ServerConfig.REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": "Request Timeout"}
            )

# Create A2A app
agent_card = types.AgentCard(
    name="Configured Agent",
    description="Agent with size limits and timeouts",
    url="http://localhost:8000",
    version="1.0.0",
    default_input_modes=["text", "file"],
    default_output_modes=["text", "file"],
    capabilities=types.AgentCapabilities(
        streaming=True,
        push_notifications=False
    ),
    skills=[...]
)

agent_executor = YourAgentExecutor()

a2a_app = A2AFastAPIApplication(
    agent_card=agent_card,
    agent_executor=agent_executor
)

# Add middlewares
app = a2a_app.app
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(TimeoutMiddleware)

# Add custom error handlers
@app.exception_handler(413)
async def request_too_large_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=413,
        content={
            "error": "Request Entity Too Large",
            "message": "Please use FileWithUri for files larger than 100MB",
            "max_size": "100 MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=ServerConfig.WORKERS,
        limit_concurrency=ServerConfig.MAX_CONNECTIONS,
        timeout_keep_alive=5,
        log_level="info"
    )
```

## Client Configuration

### 1. HTTP Client Configuration

```python
import httpx
from a2a.client import Client, ClientConfig, JsonRpcHttpClientTransport

# Configure HTTP client with custom limits and timeouts
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        timeout=120.0,      # Total timeout
        connect=10.0,       # Connection timeout
        read=60.0,          # Read timeout
        write=60.0,         # Write timeout
        pool=5.0            # Pool timeout
    ),
    limits=httpx.Limits(
        max_connections=100,      # Max total connections
        max_keepalive_connections=20,  # Max keep-alive connections
        keepalive_expiry=5.0     # Keep-alive expiry
    ),
    follow_redirects=True,
    max_redirects=5
)

# Create transport
transport = JsonRpcHttpClientTransport(
    http_client=http_client,
    base_url="http://localhost:8000/rpc"
)

# Configure A2A client
client_config = ClientConfig(
    prefer_streaming=True,
    poll_interval_seconds=1.0,
    max_poll_attempts=60
)

client = Client.create(
    card=agent_card,
    config=client_config,
    transport=transport
)
```

### 2. Retry Configuration

```python
import httpx
from httpx import AsyncClient

# HTTP client with retry
transport = httpx.AsyncHTTPTransport(
    retries=3,  # Number of retries
)

http_client = AsyncClient(
    transport=transport,
    timeout=60.0
)
```

### 3. Handling Large Responses

```python
async def handle_large_response(client: Client, message: types.Message):
    """Handle potentially large streaming responses."""
    chunk_count = 0
    total_size = 0

    async for event in client.send_message(message):
        if isinstance(event, types.Message):
            # Process message chunk
            for part in event.parts:
                if isinstance(part.root, types.TextPart):
                    text = part.root.text
                    total_size += len(text.encode('utf-8'))
                    chunk_count += 1

            # Log progress
            if chunk_count % 10 == 0:
                print(f"Received {chunk_count} chunks, {total_size / 1024:.2f} KB")

    print(f"Complete: {chunk_count} chunks, {total_size / (1024 * 1024):.2f} MB")
```

## Message Send Configuration

### MessageSendConfiguration

Configure individual message sends:

```python
from a2a import types

configuration = types.MessageSendConfiguration(
    blocking=True,  # Wait for complete response
    history_length=10,  # Include last 10 messages
    accepted_output_modes=["text", "file"],  # Specify desired formats
    push_notification_config=None  # Optional push notification config
)

async for event in client.send_message(
    message,
    configuration=configuration
):
    # Process events
    pass
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `blocking` | bool | None | Whether to wait for complete response |
| `history_length` | int | None | Number of previous messages to include |
| `accepted_output_modes` | list[str] | None | Desired output formats |
| `push_notification_config` | PushNotificationConfig | None | Push notification settings |

## Optimization Strategies

### 1. For Small Payloads (< 1 MB)

```python
# Use FileWithBytes - simple and efficient
file_part = types.FilePart(
    file=types.FileWithBytes(
        bytes=base64.b64encode(file_bytes).decode(),
        name="small_file.txt"
    )
)
```

### 2. For Medium Payloads (1-10 MB)

```python
# Consider compression before base64 encoding
import gzip
import base64

# Compress
compressed = gzip.compress(file_bytes)

# Then encode
encoded = base64.b64encode(compressed).decode()

file_part = types.FilePart(
    file=types.FileWithBytes(
        bytes=encoded,
        name="compressed_file.txt.gz",
        mime_type="application/gzip"
    ),
    metadata={"compressed": True}
)
```

### 3. For Large Payloads (> 10 MB)

```python
# Use FileWithUri instead
# 1. Upload file to storage (S3, GCS, etc.)
file_url = await upload_to_storage(file_bytes)

# 2. Send URI reference
file_part = types.FilePart(
    file=types.FileWithUri(
        uri=file_url,
        name="large_file.pdf",
        mime_type="application/pdf"
    ),
    metadata={"storage": "s3"}
)
```

### 4. Streaming Large Responses

```python
class StreamingAgent(AgentExecutor):
    async def execute(self, request_context):
        # For large outputs, stream incrementally
        buffer = ""
        buffer_size = 1000  # Characters

        async for chunk in generate_large_output():
            buffer += chunk

            # Send when buffer is full
            if len(buffer) >= buffer_size:
                yield types.Message(
                    message_id=f"msg_{uuid4()}",
                    role=types.Role.AGENT,
                    parts=[types.Part(root=types.TextPart(text=buffer))]
                )
                buffer = ""

        # Send remaining buffer
        if buffer:
            yield types.Message(
                message_id=f"msg_{uuid4()}",
                role=types.Role.AGENT,
                parts=[types.Part(root=types.TextPart(text=buffer))]
            )
```

### 5. Artifact Chunking

```python
async def send_large_artifact(
    task_id: str,
    context_id: str,
    large_content: str,
    chunk_size: int = 10000
):
    """Send large content as chunked artifacts."""
    chunks = [
        large_content[i:i+chunk_size]
        for i in range(0, len(large_content), chunk_size)
    ]

    for i, chunk in enumerate(chunks):
        is_last = (i == len(chunks) - 1)

        yield types.TaskArtifactUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            artifact=types.Artifact(
                artifact_id=f"artifact_{task_id}",
                name=f"output_part_{i+1}.txt",
                parts=[types.Part(root=types.TextPart(text=chunk))]
            ),
            append=True,  # Append to existing artifact
            last_chunk=is_last
        )
```

## Performance Tuning

### Connection Pooling

```python
# Efficient connection reuse
async with httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=50
    )
) as http_client:
    # Reuse client for multiple requests
    for message in messages:
        async for event in client.send_message(message):
            process(event)
```

### Concurrent Requests

```python
import asyncio

async def send_multiple_messages(client: Client, messages: list):
    """Send multiple messages concurrently."""
    tasks = [
        client.send_message(message)
        for message in messages
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Memory Management

```python
async def process_large_stream(client: Client, message: types.Message):
    """Process streaming response without accumulating in memory."""
    async for event in client.send_message(message):
        # Process immediately
        process_event(event)
        # Don't accumulate in memory
        # Event is garbage collected after processing
```

## Monitoring and Debugging

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("a2a")

# Log message sizes
def log_message_size(message: types.Message):
    import json
    message_dict = message.model_dump()
    message_json = json.dumps(message_dict)
    size_bytes = len(message_json.encode('utf-8'))
    logger.info(f"Message size: {size_bytes / 1024:.2f} KB")
```

### Metrics Collection

```python
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        import time
        start_time = time.time()

        # Get request size
        content_length = int(request.headers.get("content-length", 0))

        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log metrics
        logger.info(
            f"Request: {request.url.path} | "
            f"Size: {content_length / 1024:.2f} KB | "
            f"Duration: {duration:.2f}s"
        )

        return response
```

## Troubleshooting

### Common Issues

#### 1. "Request Too Large" (413 Error)

**Cause**: Request exceeds server size limit

**Solutions**:
- Use FileWithUri instead of FileWithBytes for large files
- Compress files before base64 encoding
- Increase server size limits (if appropriate)
- Split large requests into multiple smaller requests

#### 2. Request Timeout

**Cause**: Request takes too long to process

**Solutions**:
- Increase timeout configuration
- Use streaming for long-running operations
- Optimize agent processing logic
- Return early responses with status updates

#### 3. Memory Issues

**Cause**: Accumulating large responses in memory

**Solutions**:
- Process streaming events immediately
- Don't accumulate entire response before processing
- Use artifact chunking for large outputs
- Implement proper garbage collection

## Best Practices Summary

1. **File Size Guidelines**:
   - < 10 MB: FileWithBytes
   - \> 10 MB: FileWithUri

2. **Server Configuration**:
   - Set appropriate size limits via middleware
   - Configure reasonable timeouts
   - Use connection pooling

3. **Client Configuration**:
   - Reuse HTTP clients
   - Configure appropriate timeouts
   - Handle retries gracefully

4. **Optimization**:
   - Stream large responses
   - Chunk large artifacts
   - Compress when appropriate
   - Process events immediately

5. **Monitoring**:
   - Log message sizes
   - Track processing times
   - Monitor memory usage
   - Set up alerts for errors

## Related Guides
- [Message Structure Guide](01_message_structure.md)
- [File Handling Guide](05_file_handling.md)
- [Streaming Guide](04_streaming.md)
