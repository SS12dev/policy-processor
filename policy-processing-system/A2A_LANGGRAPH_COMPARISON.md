# A2A & LangGraph Implementation Comparison

## Executive Summary

Successfully migrated `policy-processing-system` to match `document-analysis-system`'s proven A2A SDK and async patterns.

## Key Issues Fixed

### 1. **Event Loop Management** ✅
**Problem**: `RuntimeError: Event loop is closed`
- policy-processing-system tried to reuse/manage event loops manually
- Caused conflicts with Streamlit's event loop

**Solution**: 
- Use `asyncio.run()` inside button handlers to create fresh event loops
- Remove `nest_asyncio` dependency (not needed with proper pattern)
- Pattern: `asyncio.run(async_function())` in Streamlit callbacks

### 2. **Client Architecture** ✅
**Problem**: Synchronous wrapper with improper async bridging

**Solution**:
```python
# document-analysis-system pattern (CORRECT)
class PolicyProcessingClient:
    async def process_document(...) -> AsyncIterator[Dict]:
        # Proper async implementation
        
def get_client() -> PolicyProcessingClient:
    # Singleton factory
    
# In Streamlit app.py:
async def process_policy_document(...):
    client = get_client()
    await client.connect()
    async for event in client.process_document(...):
        # Process events
        
# Button click:
if st.button("Process"):
    asyncio.run(process_policy_document(...))  # Fresh loop
```

### 3. **A2A Protocol Implementation** ✅

#### Message Structure
```python
# CORRECT (document-analysis-system & fixed policy-processing-system)
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
            text=f"Process policy document: {filename}"
        ))
    ],
    context_id=context_id
)

metadata = types.TaskMetadata(
    skill_id="process_policy",
    parameters={...}
)

async for event in client.send_message(message=message, metadata=metadata):
    # Process TaskStatusUpdateEvent and TaskArtifactUpdateEvent
```

#### Event Processing
```python
# Both systems now use this pattern:
async for event in self.a2a_client.send_message(message, metadata):
    if isinstance(event, types.TaskStatusUpdateEvent):
        # Extract status from event.status.state.value
        # Extract message text from event.status.message.parts
        
    elif isinstance(event, types.TaskArtifactUpdateEvent):
        # Extract data from event.artifact.parts
        # Look for part.root.data (DataPart)
```

### 4. **LangGraph Integration** ✅

#### Agent Executor Pattern
Both systems properly implement `AgentExecutor`:

```python
# policy-processing-system/agent-module/agent.py
class PolicyProcessorAgent(AgentExecutor):
    def __init__(self, orchestrator: LangGraphOrchestrator, redis_storage):
        self.orchestrator = orchestrator
        self.storage = redis_storage
    
    @override
    async def execute(self, context: RequestContext, queue: EventQueue):
        # Extract parameters
        parameters = self._extract_parameters(context)
        
        # Send status updates via queue
        await queue.send_task_status_update(...)
        
        # Process with LangGraph
        result = await self.orchestrator.run(...)
        
        # Send artifact with results
        await queue.send_task_artifact_update(...)
```

#### LangGraph Orchestrator Pattern
```python
# core/langgraph_orchestrator.py (policy-processing-system)
class LangGraphOrchestrator:
    def _build_graph(self):
        graph = StateGraph(PolicyGraphState)
        
        # Add nodes (each is a processing step)
        graph.add_node("pdf_parser", pdf_parser_node)
        graph.add_node("policy_extractor", policy_extractor_node)
        graph.add_node("tree_generator", tree_generator_node)
        # ... 10 total nodes
        
        # Add edges (workflow routing)
        graph.add_edge(START, "pdf_parser")
        graph.add_conditional_edges("pdf_parser", should_continue)
        graph.add_edge("policy_extractor", "tree_generator")
        # ...
        graph.add_edge("formatter", END)
        
        return graph.compile()
    
    async def run(self, state: PolicyGraphState):
        result = await self.workflow.ainvoke(state)
        return result
```

## Architecture Comparison

### Client Architecture

| Aspect | document-analysis-system | policy-processing-system (FIXED) |
|--------|-------------------------|----------------------------------|
| Client Class | `DocumentAnalysisClient` | `PolicyProcessingClient` |
| Factory | `get_client()` singleton | `get_client()` singleton |
| Connection | `httpx.AsyncClient` | `httpx.AsyncClient` |
| A2A Client | `ClientFactory.create()` | `ClientFactory.create()` |
| Health Check | `async check_health()` | `async check_health()` |
| Processing | `async analyze_document()` yields events | `async process_document()` yields events |
| Streamlit Integration | `asyncio.run()` in button handler | `asyncio.run()` in button handler |

### Agent Architecture

| Aspect | document-analysis-system | policy-processing-system |
|--------|-------------------------|-------------------------|
| Agent Class | `DocumentAnalysisAgent` (implied) | `PolicyProcessorAgent` |
| Base Class | `AgentExecutor` | `AgentExecutor` |
| Orchestrator | LangGraph `StateGraph` | LangGraph `StateGraph` |
| State Storage | `MemorySaver` (in-memory) | `RedisAgentStorage` (distributed) |
| Node Count | ~6 nodes | 10 nodes |
| Workflow | Linear with retries | Complex with validation/refinement |
| Concurrency | Single instance | Redis-based locking for scaling |

### Server Architecture

| Aspect | document-analysis-system | policy-processing-system |
|--------|-------------------------|-------------------------|
| Server Creation | `A2AFastAPIApplication` | `A2AFastAPIApplication` |
| Agent Card | Auto-generated | Auto-generated with custom skills |
| Health Endpoint | `/health` | `/health` (custom) |
| RPC Endpoint | `/.well-known/agent-card.json` | `/.well-known/agent-card.json` |
| Skills | 1 skill (analyze) | 1 skill (process_policy) |

## File-by-File Changes

### client-module/a2a_client.py
**BEFORE** (broken):
```python
class A2AClientSync:
    def _run_async(self, coro):
        try:
            loop = asyncio.get_event_loop()  # ❌ Reuses closed loop
            if loop.is_running():
                nest_asyncio.apply()  # ❌ Hack
                return loop.run_until_complete(coro)
            return asyncio.run(coro)  # ❌ Can close loop
```

**AFTER** (fixed):
```python
class PolicyProcessingClient:
    async def process_document(...) -> AsyncIterator[Dict]:
        """Pure async implementation"""
        # No loop management - caller handles via asyncio.run()
        
def get_client() -> PolicyProcessingClient:
    """Singleton factory"""
```

### client-module/app.py
**BEFORE** (broken):
```python
a2a_client = A2AClientSync(...)

if st.button("Process"):
    result = a2a_client.process_document(...)  # ❌ Sync call with hidden async
```

**AFTER** (fixed):
```python
from a2a_client import get_client

async def process_policy_document(...):
    client = get_client()
    await client.connect()
    async for event in client.process_document(...):
        # Process streaming events

if st.button("Process"):
    asyncio.run(process_policy_document(...))  # ✅ Fresh event loop
```

### agent-module/agent.py
**BEFORE**: Custom JSON-RPC (removed)
**AFTER**: Proper A2A SDK integration
```python
class PolicyProcessorAgent(AgentExecutor):
    @override
    async def execute(self, context: RequestContext, queue: EventQueue):
        # Extract from A2A context
        parameters = self._extract_parameters(context)
        
        # Send A2A status updates
        await queue.send_task_status_update(
            status=types.TaskStatus(
                state=types.TaskState.processing,
                message=types.Message(...)
            )
        )
        
        # Send A2A artifact with results
        await queue.send_task_artifact_update(
            artifact=types.Artifact(
                parts=[types.Part(root=types.DataPart(data=results))]
            )
        )
```

### agent-module/server.py
**BEFORE**: Manual FastAPI setup
**AFTER**: A2A SDK server
```python
from a2a.server.apps.fastapi_app import A2AFastAPIApplication

def create_server():
    agent = get_agent()
    
    app = A2AFastAPIApplication(
        agent_executor=agent,
        agent_card=create_agent_card()
    ).build_app()
    
    # Add custom health endpoint
    @app.get("/health")
    async def health():
        return {"status": "healthy", ...}
    
    return app

app = create_server()  # Export for uvicorn
```

## Testing Verification

### Health Check Test
```bash
# Agent server
curl http://localhost:8001/health
# Response: {"status":"healthy","agent":"PolicyProcessorAgent","version":"1.0.0"}

# Agent card
curl http://localhost:8001/.well-known/agent-card.json
# Response: Full agent card with skills
```

### Client Connection Test
```python
# In Streamlit sidebar
async def check_agent_health():
    client = get_client()
    return await client.check_health()

is_healthy = asyncio.run(check_agent_health())
# Shows: "● Server Online" or "● Server Offline"
```

### Document Processing Test
```python
async def process_policy_document(pdf_bytes, filename, ...):
    client = get_client()
    await client.connect()
    
    async for event in client.process_document(pdf_bytes, filename, ...):
        if event["type"] == "status":
            # UI shows progress
        elif event["type"] == "artifact":
            # UI receives results
        elif event["type"] == "complete":
            # UI shows success
```

## Best Practices Learned

### 1. **Async Event Loop Management**
✅ **DO**: Use `asyncio.run()` in Streamlit button handlers
```python
if st.button("Process"):
    asyncio.run(async_function())  # Creates fresh loop
```

❌ **DON'T**: Try to manage loops manually
```python
loop = asyncio.get_event_loop()  # Can get closed loop
loop.run_until_complete(coro)  # Fails in Streamlit
```

### 2. **A2A Client Factory Pattern**
✅ **DO**: Use singleton factory
```python
_client = None

def get_client():
    global _client
    if _client is None:
        _client = PolicyProcessingClient()
    return _client
```

❌ **DON'T**: Create clients in `@st.cache_resource`
```python
@st.cache_resource  # ❌ Bad with async
def get_client():
    return AsyncClient()
```

### 3. **Event Processing**
✅ **DO**: Use `AsyncIterator` for streaming
```python
async def process(...) -> AsyncIterator[Dict]:
    async for event in self.a2a_client.send_message(...):
        yield {"type": "status", ...}
```

❌ **DON'T**: Return single result blocking
```python
def process(...) -> Dict:  # ❌ Blocks UI
    result = wait_for_completion()
    return result
```

### 4. **LangGraph State Management**
✅ **DO**: Use typed state with `TypedDict`
```python
class PolicyGraphState(TypedDict):
    pdf_bytes: str
    extracted_text: str
    # ... all state fields typed
```

❌ **DON'T**: Use plain dicts
```python
state = {}  # ❌ No type checking
```

## Performance Optimization

### Client-Side
- Connection pooling via `httpx.AsyncClient`
- Singleton client instance
- Streaming events for progress updates

### Agent-Side
- Redis-based distributed locking
- Stateless design for horizontal scaling
- LangGraph checkpointing for resume capability
- Concurrent request handling

## Deployment Notes

### Environment Variables
Both systems now use `.env` files:
```bash
# agent-module/.env
LLM_PROVIDER=proxy
LLM_BASE_URL=http://...
REDIS_HOST=localhost
REDIS_PORT=6379
SERVER_PORT=8001

# client-module/.env
AGENT_URL=http://localhost:8001
DATABASE_URL=sqlite:///data/policy_processor.db
```

### Docker Deployment
```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  agent:
    build: ./agent-module
    ports:
      - "8001:8001"
    environment:
      - REDIS_HOST=redis
    depends_on:
      - redis
  
  client:
    build: ./client-module
    ports:
      - "8501:8501"
    environment:
      - AGENT_URL=http://agent:8001
    depends_on:
      - agent
```

## Migration Checklist

- [x] Replace synchronous client with async client
- [x] Update app.py to use `asyncio.run()` pattern
- [x] Fix A2A message structure (FilePart, TextPart)
- [x] Fix A2A event processing (TaskStatusUpdateEvent, TaskArtifactUpdateEvent)
- [x] Implement proper agent executor
- [x] Use A2AFastAPIApplication for server
- [x] Add health endpoint
- [x] Test client-agent connection
- [x] Test document upload flow
- [x] Verify streaming events
- [x] Test database persistence

## Conclusion

The policy-processing-system now follows the same proven patterns as document-analysis-system:

1. **Async-first design** with proper event loop management
2. **A2A SDK integration** for standardized agent communication
3. **LangGraph orchestration** for complex workflows
4. **Streaming events** for real-time UI updates
5. **Stateless agents** with Redis for distributed deployment

Both systems are now production-ready and can be deployed with confidence.
