# Quick Reference: A2A Client Implementation

## The Critical Fix

### ❌ WRONG (Causes AttributeError)
```python
metadata = types.TaskMetadata(  # Does NOT exist in client SDK!
    skill_id="process_policy",
    parameters={...}
)
```

### ✅ CORRECT
```python
import json

# Put parameters in TextPart as JSON
params_json = json.dumps({
    "document_base64": pdf_base64,
    "use_gpt4": True,
    "enable_streaming": True,
    "confidence_threshold": 0.7
})

message = types.Message(
    parts=[
        types.Part(root=types.FilePart(file=...)),
        types.Part(root=types.TextPart(text=params_json))  # ✅ JSON here
    ]
)

async for event in client.send_message(message):  # No metadata param!
```

## Available A2A Client Types

### ✅ You CAN use:
- `types.Message`
- `types.Part`
- `types.FilePart`
- `types.TextPart`
- `types.DataPart`
- `types.Role` (user, assistant)
- `types.TaskStatusUpdateEvent` (received)
- `types.TaskArtifactUpdateEvent` (received)

### ❌ You CANNOT use (server-side only):
- `types.TaskMetadata`
- `types.TaskStatus`
- `types.Artifact` (only received in events)
- `types.AgentCard` (only fetched, not created)

## Streamlit Event Loop Rules

### ✅ DO
```python
# Sync at module level
def check_health():
    return httpx.get("http://localhost:8001/health").status_code == 200

# Async in button handler
if st.button("Process"):
    asyncio.run(process_document())  # Fresh loop each time
```

### ❌ DON'T
```python
# At module level
asyncio.run(check_health())  # ❌ Event loop closed!

# Trying to reuse loops
loop = asyncio.get_event_loop()  # ❌ May be closed
```

## Message Structure Pattern

```python
# CLIENT SENDS:
Message(
    parts=[
        Part(FilePart(file=pdf_bytes)),      # The document
        Part(TextPart(text=json_params))     # The parameters
    ]
)

# AGENT RECEIVES:
context.message.parts[0].root  # FilePart → PDF
context.message.parts[1].root.text  # TextPart → Parse as JSON

# AGENT SENDS BACK:
TaskStatusUpdateEvent(status=...)     # Progress
TaskArtifactUpdateEvent(artifact=...) # Results

# CLIENT RECEIVES:
event.status.state.value               # "processing", "completed"
event.artifact.parts[0].root.data      # Final results
```

## Complete Working Example

```python
# a2a_client.py
async def process_document(file_bytes, filename, use_gpt4=False):
    # 1. Encode file
    pdf_base64 = base64.b64encode(file_bytes).decode()
    
    # 2. Build parameters
    params = json.dumps({
        "document_base64": pdf_base64,
        "use_gpt4": use_gpt4
    })
    
    # 3. Create message
    message = types.Message(
        message_id=f"msg_{uuid.uuid4()}",
        role=types.Role.user,
        parts=[
            types.Part(root=types.FilePart(
                file=types.FileWithBytes(
                    bytes=pdf_base64,
                    name=filename,
                    mime_type="application/pdf"
                )
            )),
            types.Part(root=types.TextPart(text=params))
        ]
    )
    
    # 4. Send and process events
    async for event in self.a2a_client.send_message(message):
        if isinstance(event, types.TaskStatusUpdateEvent):
            yield {"type": "status", "status": event.status.state.value}
        elif isinstance(event, types.TaskArtifactUpdateEvent):
            data = event.artifact.parts[0].root.data
            yield {"type": "artifact", "data": data}
```

```python
# app.py
async def process_doc(file_bytes, filename):
    client = get_client()
    await client.connect()
    
    async for event in client.process_document(file_bytes, filename):
        if event["type"] == "status":
            st.text(f"Status: {event['status']}")
        elif event["type"] == "artifact":
            st.success("Complete!")
            return event["data"]

# In button handler
if st.button("Process"):
    result = asyncio.run(process_doc(pdf_bytes, filename))
```

## Troubleshooting

### Error: "TaskMetadata not found"
**Fix**: Remove TaskMetadata, use TextPart with JSON

### Error: "Event loop is closed"
**Fix**: Use sync HTTP at module level, asyncio.run() in buttons

### Error: "No results received"
**Fix**: Check agent logs, ensure JSON parsing works

### Agent doesn't see parameters
**Fix**: Verify JSON in TextPart, check agent's _extract_parameters()

## Testing Commands

```bash
# Test agent health
curl http://localhost:8001/health

# Test agent card
curl http://localhost:8001/.well-known/agent-card.json

# Start agent
cd agent-module && python server.py

# Start client
cd client-module && streamlit run app.py --server.port 8501
```

## Success Indicators

- ✅ No import errors
- ✅ Sidebar shows "Server Online"
- ✅ Can upload PDF
- ✅ Process button works
- ✅ Progress updates appear
- ✅ Results saved to database
- ✅ No event loop errors
- ✅ No TaskMetadata errors
