# A2A SDK Comprehensive Guide

Complete exploratory research and implementation guide for Google's Agent-to-Agent (A2A) Protocol Python SDK.

## Table of Contents

1. [Introduction](#introduction)
2. [What is the A2A Protocol?](#what-is-the-a2a-protocol)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Architecture Overview](#architecture-overview)
7. [Examples Directory](#examples-directory)
8. [Guides](#guides)
9. [Message Size Limits and Configuration](#message-size-limits-and-configuration)
10. [File Attachments](#file-attachments)
11. [Streaming vs Non-Streaming](#streaming-vs-non-streaming)
12. [LangGraph Integration](#langgraph-integration)
13. [Resources](#resources)

## Introduction

This repository contains comprehensive research, documentation, and working examples for the **A2A (Agent-to-Agent) Protocol Python SDK** developed by Google. The A2A protocol enables standardized communication between AI agents, creating interoperable agent ecosystems that can collaborate to solve complex problems.

## What is the A2A Protocol?

The A2A protocol is an open standard that defines:

- **Standardized Message Format**: Common structure for agent communication
- **Task Management**: Lifecycle management for agent tasks
- **Streaming Support**: Real-time progressive responses
- **Authentication**: Security schemes for agent-to-agent communication
- **Agent Discovery**: Agent cards for capability advertisement

### Key Benefits

1. **Interoperability**: Agents from different frameworks can communicate
2. **Flexibility**: Support for various communication patterns (sync/async, streaming/polling)
3. **Scalability**: Built on modern async Python for high performance
4. **Protocol Agnostic**: Supports JSON-RPC, gRPC, and REST

## Key Features

### What the A2A SDK Provides

1. **Client-Side**:
   - Easy client creation for connecting to A2A agents
   - Automatic protocol negotiation (streaming vs non-streaming)
   - Connection management and error handling
   - Event consumers for processing agent responses
   - Task resubscription for connection recovery

2. **Server-Side**:
   - FastAPI and Starlette application integrations
   - Agent executor abstraction
   - Built-in task management and state handling
   - Event queue system for async processing
   - Automatic agent card serving

3. **Type System**:
   - Comprehensive Pydantic models for all protocol types
   - Message parts: TextPart, FilePart, DataPart
   - File handling: FileWithBytes (base64), FileWithUri (URL reference)
   - Artifact system for structured outputs

4. **Advanced Features**:
   - Multi-turn conversations with context management
   - Streaming responses with progressive updates
   - Push notifications configuration
   - Middleware/interceptor support
   - LangGraph integration patterns

## Installation

```bash
# Basic installation
pip install a2a-sdk

# With development dependencies
pip install a2a-sdk[dev]

# Additional dependencies for examples
pip install uvicorn fastapi langgraph langchain langchain-core
```

### Requirements

- Python >= 3.10
- httpx >= 0.28.1
- pydantic >= 2.11.3
- protobuf >= 5.29.5

## Quick Start

### Server Example

```python
from a2a import types
from a2a.server import AgentExecutor, A2AFastAPIApplication

class MyAgent(AgentExecutor):
    async def execute(self, request_context):
        # Extract user message
        user_text = request_context.user_message.parts[0].root.text

        # Yield status update
        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(state=types.TaskState.WORKING),
            final=False
        )

        # Yield response
        yield types.Message(
            message_id="msg_1",
            role=types.Role.AGENT,
            parts=[types.Part(root=types.TextPart(text=f"Echo: {user_text}"))]
        )

# Create server
app = A2AFastAPIApplication(
    agent_card=types.AgentCard(name="My Agent", ...),
    agent_executor=MyAgent()
)
```

### Client Example

```python
from a2a import types
from a2a.client import Client, ClientConfig, JsonRpcHttpClientTransport
import httpx

async with httpx.AsyncClient() as http_client:
    # Fetch agent card
    card = await fetch_agent_card(http_client, "http://localhost:8000")

    # Create client
    transport = JsonRpcHttpClientTransport(
        http_client=http_client,
        base_url="http://localhost:8000/rpc"
    )
    client = Client.create(
        card=card,
        config=ClientConfig(prefer_streaming=True),
        transport=transport
    )

    # Send message
    message = types.Message(
        message_id="msg_1",
        role=types.Role.USER,
        parts=[types.Part(root=types.TextPart(text="Hello!"))]
    )

    async for event in client.send_message(message):
        # Process events
        print(event)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        A2A Protocol Layer                   │
│  (Message Format, Task Management, Streaming, Security)     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                     ┌───────▼────────┐
│   A2A Client   │                     │   A2A Server   │
│                │                     │                │
│ - Transport    │                     │ - FastAPI App  │
│ - Streaming    │                     │ - Executor     │
│ - Consumers    │                     │ - Task Manager │
│ - Middleware   │                     │ - Event Queue  │
└────────────────┘                     └────────────────┘
        │                                       │
        │                                       │
┌───────▼────────┐                     ┌───────▼────────┐
│ Your Client    │                     │  Your Agent    │
│ Application    │◄───────────────────►│  (LangGraph,   │
│                │      HTTP/gRPC      │   CrewAI, etc) │
└────────────────┘                     └────────────────┘
```

## Examples Directory

All examples are located in the [`examples/`](examples/) directory:

| File | Description |
|------|-------------|
| [01_simple_agent_server.py](examples/01_simple_agent_server.py) | Basic non-streaming agent server |
| [02_streaming_agent_server.py](examples/02_streaming_agent_server.py) | Streaming agent with progressive updates |
| [03_non_streaming_client.py](examples/03_non_streaming_client.py) | Client for non-streaming interactions |
| [04_streaming_client.py](examples/04_streaming_client.py) | Client for streaming interactions |
| [05_document_processing_agent.py](examples/05_document_processing_agent.py) | Agent that handles file uploads |
| [06_document_processing_client.py](examples/06_document_processing_client.py) | Client that sends files to agents |
| [07_langgraph_agent_server.py](examples/07_langgraph_agent_server.py) | LangGraph integration example |

### Running the Examples

1. **Start a server**:
   ```bash
   python examples/01_simple_agent_server.py
   ```

2. **Run the corresponding client** (in another terminal):
   ```bash
   python examples/03_non_streaming_client.py
   ```

## Guides

Comprehensive guides are available in the [`guides/`](guides/) directory:

- [Message Structure Guide](guides/01_message_structure.md) - Understanding A2A messages and parts
- [Server Implementation Guide](guides/02_server_implementation.md) - Building A2A servers
- [Client Implementation Guide](guides/03_client_implementation.md) - Building A2A clients
- [Streaming Guide](guides/04_streaming.md) - Implementing streaming responses
- [File Handling Guide](guides/05_file_handling.md) - Working with file attachments
- [Configuration Guide](guides/06_configuration.md) - Message size limits and server config
- [LangGraph Integration Guide](guides/07_langgraph_integration.md) - Integrating with LangGraph

## Message Size Limits and Configuration

### Default Limits

The A2A SDK itself **does not impose hard limits** on message sizes, but practical limits exist:

#### FileWithBytes (Base64 Encoded)
- **Recommended**: < 10 MB
- **Typical Maximum**: 10-100 MB (depends on HTTP server configuration)
- **Note**: Base64 encoding increases size by ~33%

#### FileWithUri (URL Reference)
- **Protocol Limit**: None (file stays at source)
- **Best for**: Large files, videos, datasets
- **Agent responsibility**: Download and process the file from the URI

### Configuring Server Limits

#### Uvicorn Configuration

```python
import uvicorn

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    limit_max_requests=1000,
    timeout_keep_alive=5,
    # Note: Uvicorn doesn't have a direct max request size parameter
    # It's typically limited by the underlying HTTP implementation
)
```

#### Gunicorn Configuration

```bash
gunicorn app:app -k uvicorn.workers.UvicornWorker \
  --limit-request-line 8190 \
  --limit-request-fields 100 \
  --limit-request-field_size 8190 \
  --timeout 120
```

#### FastAPI/Starlette Configuration

```python
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

app = FastAPI()

# Add middleware to handle large requests
@app.middleware("http")
async def check_request_size(request: Request, call_next):
    # Check content length
    content_length = request.headers.get("content-length")
    if content_length:
        length = int(content_length)
        max_size = 100 * 1024 * 1024  # 100 MB
        if length > max_size:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request too large"}
            )
    return await call_next(request)
```

### Best Practices

1. **For files < 10MB**: Use FileWithBytes
2. **For files > 10MB**: Use FileWithUri
3. **Compress before encoding**: Reduces base64 size
4. **Use streaming**: For large responses
5. **Chunk artifacts**: Split large outputs into multiple artifacts

## File Attachments

The A2A protocol supports file attachments through the **FilePart** type:

### FileWithBytes (Embedded Files)

```python
import base64

# Read file
with open("document.txt", "rb") as f:
    file_bytes = f.read()

# Create FilePart
file_part = types.FilePart(
    file=types.FileWithBytes(
        bytes=base64.b64encode(file_bytes).decode(),
        name="document.txt",
        mime_type="text/plain"
    ),
    metadata={"category": "document"}
)

# Add to message
message = types.Message(
    message_id="msg_1",
    role=types.Role.USER,
    parts=[types.Part(root=file_part)]
)
```

### FileWithUri (Referenced Files)

```python
file_part = types.FilePart(
    file=types.FileWithUri(
        uri="https://example.com/large-file.pdf",
        name="large-file.pdf",
        mime_type="application/pdf"
    )
)
```

### When to Use Each

| FileWithBytes | FileWithUri |
|---------------|-------------|
| Small files (< 10MB) | Large files (> 10MB) |
| Files need to be embedded | Files hosted elsewhere |
| Transient files | Persistent files |
| High security needs | Lower latency |

## Streaming vs Non-Streaming

### Streaming

**Advantages**:
- Real-time updates
- Better user experience for long operations
- Progressive content generation
- Lower perceived latency

**Server Implementation**:
```python
class StreamingAgent(AgentExecutor):
    async def execute(self, request_context):
        # Yield multiple updates
        for i, chunk in enumerate(generate_chunks()):
            yield types.Message(
                message_id=f"msg_{i}",
                role=types.Role.AGENT,
                parts=[types.Part(root=types.TextPart(text=chunk))]
            )
```

**Client Usage**:
```python
client = Client.create(
    card=agent_card,
    config=ClientConfig(prefer_streaming=True),
    transport=transport
)

async for event in client.send_message(message):
    # Process events as they arrive
    handle_event(event)
```

### Non-Streaming (Polling)

**Advantages**:
- Simpler implementation
- Better for short operations
- Easier error handling
- Lower server resources

**Server Implementation**:
```python
class NonStreamingAgent(AgentExecutor):
    async def execute(self, request_context):
        # Process completely, then yield final result
        result = await process_request(request_context)

        yield types.Message(
            message_id="msg_1",
            role=types.Role.AGENT,
            parts=[types.Part(root=types.TextPart(text=result))]
        )

        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(state=types.TaskState.COMPLETED),
            final=True
        )
```

**Client Usage**:
```python
client = Client.create(
    card=agent_card,
    config=ClientConfig(
        prefer_streaming=False,
        poll_interval_seconds=1.0,
        max_poll_attempts=60
    ),
    transport=transport
)
```

### Agent Card Configuration

```python
agent_card = types.AgentCard(
    name="My Agent",
    capabilities=types.AgentCapabilities(
        streaming=True,  # Indicates streaming support
        push_notifications=False
    ),
    # ... other fields
)
```

## LangGraph Integration

LangGraph can be integrated with A2A to create sophisticated, stateful agents:

### Key Integration Points

1. **Context ID → Thread ID**: A2A's context_id maps to LangGraph's thread_id for conversation history
2. **Agent Executor**: Wraps LangGraph agent as an A2A-compliant executor
3. **Tool Usage**: LangGraph tools work seamlessly with A2A messages
4. **Memory**: LangGraph's checkpointer maintains conversation state

### Example Integration

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

class LangGraphAgent(AgentExecutor):
    def __init__(self):
        self.tools = [your_tools]
        self.memory = MemorySaver()
        self.graph = create_react_agent(
            model=your_llm,
            tools=self.tools,
            checkpointer=self.memory
        )

    async def execute(self, request_context):
        # Extract user input
        user_text = extract_text(request_context.user_message)

        # Configure with context ID
        config = {"configurable": {"thread_id": request_context.context_id}}

        # Invoke LangGraph
        result = await asyncio.to_thread(
            self.graph.invoke,
            {"messages": [("user", user_text)]},
            config
        )

        # Convert to A2A response
        yield convert_to_a2a_message(result)
```

See [examples/07_langgraph_agent_server.py](examples/07_langgraph_agent_server.py) for a complete implementation.

## Resources

### Official Documentation
- [A2A Protocol Specification](https://google.github.io/A2A/)
- [A2A Python SDK (PyPI)](https://pypi.org/project/a2a-sdk/)
- [A2A GitHub Repository](https://github.com/a2aproject/a2a-python)

### Related Projects
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### Community & Support
- [A2A Discussion Forum](https://github.com/a2aproject/a2a-python/discussions)
- [Issue Tracker](https://github.com/a2aproject/a2a-python/issues)

### Tutorials & Blogs
- [Building an A2A Currency Agent with LangGraph](https://a2aprotocol.ai/blog/a2a-langraph-tutorial-20250513)
- [Google A2A Python SDK Tutorial](https://a2aprotocol.ai/blog/google-a2a-python-sdk-tutorial)
- [Getting Started with Google A2A](https://medium.com/google-cloud/getting-started-with-google-a2a-a-hands-on-tutorial-for-the-agent2agent-protocol-3d3b5e055127)

## Contributing

This repository is a research and educational resource. Feel free to:
- Submit issues for clarifications
- Propose improvements to examples
- Share your own A2A implementations
- Add new guides or tutorials

## License

The examples and documentation in this repository are provided for educational purposes. The A2A SDK itself is licensed by its respective maintainers.

## Acknowledgments

- Google and the A2A Project team for developing the protocol and SDK
- LangChain team for LangGraph integration
- The open-source community for feedback and contributions

---

**Last Updated**: December 2025
**A2A SDK Version**: 0.3.20
**Python Version**: 3.11+
