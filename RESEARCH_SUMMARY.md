# A2A SDK Comprehensive Research Summary

## Executive Summary

This document summarizes the complete exploratory research of Google's Agent-to-Agent (A2A) Protocol Python SDK (version 0.3.20). The research covered all aspects of the SDK including server/client implementation, streaming/non-streaming modes, file handling, message size limits, LangGraph integration, and configuration options.

## Key Findings

### 1. Library Structure and Components

#### Core Modules
- **`a2a.client`**: Client-side components for interacting with A2A agents
- **`a2a.server`**: Server-side components for implementing A2A agents
- **`a2a.types`**: Comprehensive type definitions (94+ Pydantic models)
- **`a2a.utils`**: Utility functions
- **`a2a.extensions`**: Extension system for protocol enhancements
- **`a2a.grpc`**: gRPC support (in addition to JSON-RPC/HTTP)

#### Server Components
- `AgentExecutor`: Abstract base for implementing agent logic
- `A2AFastAPIApplication`: FastAPI integration
- `RequestContext`: Contains request information and user message
- `TaskManager`: Manages task lifecycle and state
- `EventQueue`: Handles event streaming
- Multiple storage backends (in-memory, database)

#### Client Components
- `Client`: Abstract base class for A2A clients
- `BaseClient`: Concrete implementation with transport abstraction
- `JsonRpcHttpClientTransport`: HTTP-based transport
- `ClientConfig`: Configuration for streaming, polling, etc.
- `ClientCallContext`: Per-request context and metadata
- Middleware/interceptor support for request/response processing

### 2. Message Structure and Types

#### Message Parts (3 Types)
1. **TextPart**: Plain text content
   - Fields: `text` (required), `metadata` (optional)
   - Use case: Natural language, instructions, responses

2. **FilePart**: File attachments
   - **FileWithBytes**: Base64-encoded file content
     - Best for: Small files (< 10MB)
     - Overhead: ~33% size increase from base64 encoding
   - **FileWithUri**: URL reference to file
     - Best for: Large files (> 10MB)
     - No size limit (file stays at source)

3. **DataPart**: Structured JSON data
   - Fields: `data` (dict, required), `metadata` (optional)
   - Use case: Configuration, API responses, structured data

#### Key Message Fields
- `message_id` (required): Unique identifier
- `role` (required): USER or AGENT
- `parts` (required): List of message parts
- `context_id` (optional): Conversation grouping
- `task_id` (optional): Associated task reference
- `metadata` (optional): Custom application data

### 3. Message Size Limits and Configuration

#### Protocol-Level Findings
- **No hard limits** defined in the A2A protocol specification
- Limits determined by transport layer and server configuration

#### Practical Limits Discovered

| Component | Typical Limit | Configurable | Recommendation |
|-----------|---------------|--------------|----------------|
| HTTP Request Body | 10-100 MB | Yes | 10 MB default |
| FileWithBytes | Depends on HTTP | Yes | < 10 MB |
| FileWithUri | No limit | N/A | > 10 MB |
| Base64 Encoding | +33% overhead | N/A | Consider compression |

#### Configuration Points

**Server-Side (Uvicorn)**:
```python
uvicorn.run(
    app,
    limit_concurrency=1000,
    limit_max_requests=10000,
    timeout_keep_alive=5
)
```

**Application-Level Middleware**:
```python
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 100 * 1024 * 1024):
        # Enforce 100MB limit
```

**Client-Side**:
```python
httpx.AsyncClient(
    timeout=httpx.Timeout(120.0, connect=10.0),
    limits=httpx.Limits(max_connections=100)
)
```

### 4. Streaming vs Non-Streaming

#### Streaming Capabilities

**Server Implementation**:
- Agent executor yields multiple events (messages, status updates, artifacts)
- Progressive content generation
- Real-time status updates
- Better UX for long operations

**Client Consumption**:
- Async iterator pattern
- Event consumers for reactive processing
- Resubscription support for connection recovery

**Configuration**:
```python
# Server declares capability
AgentCard(capabilities=AgentCapabilities(streaming=True))

# Client prefers streaming
ClientConfig(prefer_streaming=True)
```

#### Non-Streaming (Polling)

**Characteristics**:
- Simpler implementation
- Better for short operations
- Agent completes processing before responding
- Client polls for completion

**Configuration**:
```python
ClientConfig(
    prefer_streaming=False,
    poll_interval_seconds=1.0,
    max_poll_attempts=60
)
```

#### Streaming is Designed into the Agent Executor

Key finding: The A2A SDK is **designed with streaming as the primary pattern**. The `AgentExecutor.execute()` method returns an `AsyncIterator`, which naturally supports:

1. **Progressive Responses**: Agent can yield partial results
2. **Status Updates**: Intermediate progress notifications
3. **Artifact Streaming**: Large outputs can be chunked
4. **Non-Blocking**: Client receives updates as they're generated

**The SDK makes streaming easy while allowing non-streaming as a fallback**:
- If server supports streaming + client wants streaming → Streaming
- If server doesn't support streaming → Automatic polling fallback
- Client controls the preference via `ClientConfig.prefer_streaming`

### 5. File Attachments

#### FileWithBytes (Embedded)

**Implementation**:
```python
import base64

file_bytes = base64.b64encode(content).decode()
FilePart(file=FileWithBytes(
    bytes=file_bytes,
    name="file.txt",
    mime_type="text/plain"
))
```

**Characteristics**:
- File travels with the message
- Base64 encoding required
- Size limit: HTTP request limit
- Best for: Small, transient files

#### FileWithUri (Referenced)

**Implementation**:
```python
FilePart(file=FileWithUri(
    uri="https://storage.example.com/file.pdf",
    name="file.pdf",
    mime_type="application/pdf"
))
```

**Characteristics**:
- File stays at source
- No size limit in protocol
- Agent downloads when needed
- Best for: Large, persistent files

#### Best Practices Identified

1. **Size threshold**: Use FileWithUri for files > 10MB
2. **Compression**: Gzip before base64 for better efficiency
3. **Security**: FileWithUri requires access control consideration
4. **Latency**: FileWithBytes faster for small files

### 6. LangGraph Integration

#### Integration Pattern Discovered

LangGraph agents integrate with A2A through:

1. **Agent Executor Wrapper**: LangGraph agent wrapped in `AgentExecutor`
2. **Context Mapping**: `context_id` → `thread_id` for conversation history
3. **Memory Management**: LangGraph's `MemorySaver` checkpointer preserves state
4. **Tool Integration**: LangGraph tools work seamlessly with A2A messages

#### Implementation Architecture

```python
class LangGraphAgent(AgentExecutor):
    def __init__(self):
        self.memory = MemorySaver()
        self.graph = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=self.memory
        )

    async def execute(self, request_context):
        # Extract user input
        user_text = extract_text(request_context.user_message)

        # Map context_id to thread_id
        config = {
            "configurable": {"thread_id": request_context.context_id}
        }

        # Invoke LangGraph
        result = self.graph.invoke(
            {"messages": [("user", user_text)]},
            config
        )

        # Convert to A2A response
        yield convert_to_a2a(result)
```

#### Key Benefits

1. **Standardized Communication**: A2A handles protocol layer
2. **Agent Logic Separation**: LangGraph handles reasoning
3. **Conversation Continuity**: Context preservation across turns
4. **Tool Ecosystem**: Access to LangGraph's tool library
5. **Interoperability**: LangGraph agents can communicate with any A2A agent

### 7. Agent Card and Discovery

#### Agent Card Components

```python
AgentCard(
    name="Agent Name",
    description="What the agent does",
    url="http://agent-endpoint.com",
    version="1.0.0",

    # I/O Modes
    default_input_modes=["text", "file"],
    default_output_modes=["text", "artifact"],

    # Capabilities
    capabilities=AgentCapabilities(
        streaming=True,
        push_notifications=False
    ),

    # Skills
    skills=[
        AgentSkill(
            name="skill_name",
            description="Skill description",
            input_modes=["text"],
            output_modes=["text"]
        )
    ],

    # Security (optional)
    security_schemes={...},

    # Protocol
    protocol_version="0.3.0",
    preferred_transport="JSONRPC"
)
```

#### Discovery Mechanism

- Agent card served at `/.well-known/agent-card.json`
- Clients fetch card to discover capabilities
- Enables capability negotiation
- Supports authentication schemes declaration

### 8. Task Management and State

#### Task States

```python
class TaskState(str, Enum):
    WORKING = "working"           # In progress
    INPUT_REQUIRED = "input_required"  # Needs user input
    COMPLETED = "completed"       # Finished successfully
    FAILED = "failed"             # Error occurred
    CANCELLED = "cancelled"       # User cancelled
```

#### Task Events

1. **TaskStatusUpdateEvent**: Status changes
   - Fields: `task_id`, `context_id`, `status`, `final`
   - Use: Progress updates, completion notification

2. **TaskArtifactUpdateEvent**: Artifact generation
   - Fields: `task_id`, `artifact`, `append`, `last_chunk`
   - Use: Deliver generated files, reports, structured outputs

#### Multi-Turn Conversations

- Each conversation has a unique `context_id`
- Messages reference same `context_id` for continuity
- Tasks created within context
- Server manages conversation state
- LangGraph integration preserves history via checkpointer

### 9. Advanced Features

#### Event Consumers

```python
async def my_consumer(event, card):
    # Process events reactively
    if isinstance(event, types.Message):
        log_message(event)

client = Client.create(
    card=card,
    consumers=[my_consumer],
    ...
)
```

#### Middleware/Interceptors

```python
class AuthInterceptor(ClientCallInterceptor):
    async def intercept(self, method_name, request_payload, http_kwargs, agent_card, context):
        # Add auth headers
        http_kwargs["headers"]["Authorization"] = f"Bearer {token}"
        return request_payload, http_kwargs
```

#### Artifact Chunking

For large outputs:
```python
yield TaskArtifactUpdateEvent(
    artifact=Artifact(...),
    append=True,      # Append to existing
    last_chunk=False  # More chunks coming
)
```

#### Task Resubscription

For connection recovery:
```python
# Reconnect to ongoing task
async for event in client.resubscribe(
    TaskIdParams(task_id=task_id)
):
    process_event(event)
```

## What the A2A SDK Provides vs What You Implement

### SDK Provides

1. **Protocol Implementation**: Message format, serialization, validation
2. **Transport Layer**: HTTP/gRPC clients and servers
3. **Type System**: Comprehensive Pydantic models
4. **Server Framework**: FastAPI/Starlette integration
5. **Client Framework**: Connection management, streaming
6. **Task Management**: State tracking, event queuing
7. **Discovery**: Agent card serving and parsing

### You Implement

1. **Agent Logic**: The actual processing (`AgentExecutor.execute()`)
2. **Tools/Functions**: Capabilities your agent can use
3. **Response Generation**: Creating appropriate messages and artifacts
4. **Error Handling**: Application-specific error management
5. **Authentication**: Implementing security schemes (if needed)
6. **Storage**: Choosing persistence layer for tasks/state
7. **LLM Integration**: Connecting to AI models (OpenAI, Anthropic, etc.)

## Configuration Flexibility

### Server-Side Configuration

1. **Application Server**: Uvicorn, Gunicorn, Hypercorn
2. **Workers**: Single or multi-process
3. **Size Limits**: Via middleware
4. **Timeouts**: Request and keep-alive
5. **Concurrency**: Connection limits
6. **Storage**: In-memory or database
7. **Logging**: Custom logging configuration

### Client-Side Configuration

1. **Transport**: HTTP, gRPC
2. **Streaming**: Prefer streaming or polling
3. **Polling**: Interval and max attempts
4. **Timeouts**: Connect, read, write
5. **Retry**: Number of retries
6. **Connection Pooling**: Max connections, keep-alive
7. **Middleware**: Custom interceptors

### Per-Message Configuration

```python
MessageSendConfiguration(
    blocking=True,
    history_length=10,
    accepted_output_modes=["text", "file"]
)
```

## Testing and Validation

### What Was Tested

1. **Server Implementations**:
   - Simple echo agent (non-streaming)
   - Streaming agent with progressive updates
   - Document processing agent with file handling
   - LangGraph-integrated agent

2. **Client Implementations**:
   - Non-streaming client
   - Streaming client with consumers
   - File upload client (both methods)
   - Multi-message client

3. **Integration Scenarios**:
   - Single-turn conversations
   - Multi-turn conversations
   - File uploads (base64 and URI)
   - Streaming responses
   - Error handling
   - Reconnection/resubscription

### Validation Results

All example implementations:
- ✓ Successfully validate against A2A protocol
- ✓ Handle both streaming and non-streaming modes
- ✓ Support file attachments correctly
- ✓ Manage conversation state properly
- ✓ Integrate with LangGraph
- ✓ Follow best practices

## Performance Considerations

### Identified Bottlenecks

1. **Base64 Encoding**: 33% size overhead
2. **HTTP Request Parsing**: Limited by server config
3. **Memory Accumulation**: Streaming helps prevent
4. **Connection Overhead**: Connection pooling helps

### Optimization Strategies

1. **Compression**: Gzip before base64
2. **Chunking**: Split large responses
3. **Streaming**: Use for long operations
4. **Connection Reuse**: HTTP client pooling
5. **FileWithUri**: For large files
6. **Concurrent Processing**: Async patterns

## Industry Standards Compliance

### Followed Standards

1. **HTTP/REST**: Standard HTTP methods and status codes
2. **JSON-RPC 2.0**: For RPC communication
3. **Base64 Encoding**: RFC 4648
4. **MIME Types**: Standard media types
5. **Async Python**: Modern async/await patterns
6. **Pydantic**: Type validation and serialization
7. **OpenAPI/Swagger**: FastAPI integration

### Best Practices Applied

1. **Type Safety**: Comprehensive Pydantic models
2. **Error Handling**: Structured error responses
3. **Logging**: Structured logging support
4. **Documentation**: Docstrings and type hints
5. **Testing**: Unit and integration test support
6. **Security**: Authentication scheme support
7. **Scalability**: Async design, connection pooling

## Limitations and Considerations

### Known Limitations

1. **No Built-in Authentication**: Must implement security schemes
2. **No Built-in Rate Limiting**: Implement at application level
3. **File Size Limits**: Determined by transport/server, not protocol
4. **No Built-in Caching**: Implement if needed
5. **gRPC Support**: Requires additional dependencies

### Design Considerations

1. **Streaming First**: Architecture optimized for streaming
2. **Async Only**: No synchronous API
3. **Pydantic Dependency**: Strict type validation
4. **FastAPI Focus**: Primary server framework
5. **Python 3.10+**: Modern Python required

## Deployment Recommendations

### Development

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Production

```bash
gunicorn app:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "4", "--bind", "0.0.0.0:8000"]
```

## Conclusion

The A2A SDK is a **comprehensive, production-ready framework** for building interoperable AI agents. Key strengths:

1. **Well-Designed**: Clear separation of concerns
2. **Flexible**: Supports multiple patterns and use cases
3. **Performant**: Built on modern async Python
4. **Extensible**: Middleware, extensions, custom transports
5. **Type-Safe**: Comprehensive Pydantic models
6. **Standards-Compliant**: Follows web and Python best practices

The SDK successfully abstracts the complexity of agent-to-agent communication while providing the flexibility needed for sophisticated agent implementations, including seamless integration with frameworks like LangGraph.

## Appendix: File Structure

```
a2a exploration/
├── README.md                          # Main documentation
├── RESEARCH_SUMMARY.md                # This file
├── examples/                          # Working examples
│   ├── 01_simple_agent_server.py
│   ├── 02_streaming_agent_server.py
│   ├── 03_non_streaming_client.py
│   ├── 04_streaming_client.py
│   ├── 05_document_processing_agent.py
│   ├── 06_document_processing_client.py
│   └── 07_langgraph_agent_server.py
├── guides/                            # Detailed guides
│   ├── 01_message_structure.md
│   ├── 02_server_implementation.md
│   ├── 03_client_implementation.md
│   ├── 04_streaming.md
│   ├── 05_file_handling.md
│   ├── 06_configuration.md
│   └── 07_langgraph_integration.md
├── exploration scripts/               # Research scripts
│   ├── explore_a2a_structure.py
│   ├── deep_explore_a2a.py
│   ├── explore_types.py
│   └── explore_file_parts.py
└── .venv/                             # Virtual environment
```

## References

1. [A2A Protocol Specification](https://google.github.io/A2A/)
2. [A2A Python SDK (PyPI)](https://pypi.org/project/a2a-sdk/)
3. [A2A GitHub Repository](https://github.com/a2aproject/a2a-python)
4. [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
5. [Building an A2A Currency Agent with LangGraph](https://a2aprotocol.ai/blog/a2a-langraph-tutorial-20250513)
6. [Google A2A Python SDK Tutorial](https://a2aprotocol.ai/blog/google-a2a-python-sdk-tutorial)

---

**Research Date**: December 2025
**SDK Version**: 0.3.20
**Python Version**: 3.11+
