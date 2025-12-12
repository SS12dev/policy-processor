# Message Structure Guide

## Overview

The A2A protocol uses a structured message format based on Pydantic models. Understanding this structure is crucial for building A2A-compliant agents.

## Core Message Components

### Message

The `Message` type represents a single message in the conversation:

```python
from a2a import types

message = types.Message(
    message_id="unique_msg_id",  # Required: Unique identifier
    role=types.Role.USER,        # Required: USER or AGENT
    parts=[...],                  # Required: List of message parts
    context_id="ctx_123",        # Optional: Conversation context
    task_id="task_456",          # Optional: Associated task
    metadata={...},              # Optional: Additional data
    reference_task_ids=[...],    # Optional: Referenced tasks
    extensions=[...]             # Optional: Protocol extensions
)
```

### Message Roles

```python
class Role(str, Enum):
    USER = "user"   # Message from the user/client
    AGENT = "agent" # Message from the agent/server
```

## Message Parts

Messages consist of one or more "parts". Each part represents a different type of content:

### Part Types

The `Part` type is a union of three specific part types:

```python
types.Part(root=...)  # root can be TextPart, FilePart, or DataPart
```

### 1. TextPart - Plain Text Content

```python
text_part = types.TextPart(
    text="Hello, how can I help you?",
    metadata={"language": "en", "tone": "friendly"}
)

# Wrap in Part
part = types.Part(root=text_part)
```

**Fields**:
- `text` (str, required): The text content
- `metadata` (dict, optional): Additional metadata
- `kind` (literal "text", default): Part type identifier

**Use cases**:
- User queries
- Agent responses
- Instructions
- Conversational text

### 2. FilePart - File Attachments

FilePart can contain files in two ways:

#### FileWithBytes (Embedded Files)

```python
import base64

# Read and encode file
with open("document.txt", "rb") as f:
    file_bytes = base64.b64encode(f.read()).decode()

file_part = types.FilePart(
    file=types.FileWithBytes(
        bytes=file_bytes,              # Required: Base64 encoded content
        name="document.txt",            # Optional: File name
        mime_type="text/plain"          # Optional: MIME type
    ),
    metadata={"source": "upload"}
)

part = types.Part(root=file_part)
```

**FileWithBytes Fields**:
- `bytes` (str, required): Base64-encoded file content
- `name` (str, optional): Original filename
- `mime_type` (str, optional): MIME type (e.g., "text/plain", "image/jpeg")

**Best for**:
- Small files (< 10MB)
- Files that need to be embedded
- Transient files

#### FileWithUri (Referenced Files)

```python
file_part = types.FilePart(
    file=types.FileWithUri(
        uri="https://example.com/file.pdf",  # Required: File location
        name="report.pdf",                    # Optional: File name
        mime_type="application/pdf"           # Optional: MIME type
    ),
    metadata={"access": "public"}
)

part = types.Part(root=file_part)
```

**FileWithUri Fields**:
- `uri` (str, required): URL to the file
- `name` (str, optional): Filename
- `mime_type` (str, optional): MIME type

**Best for**:
- Large files (> 10MB)
- Files hosted elsewhere
- Persistent file references

### 3. DataPart - Structured Data

```python
data_part = types.DataPart(
    data={
        "type": "analytics",
        "metrics": {
            "views": 1000,
            "clicks": 50
        },
        "timestamp": "2025-01-01T00:00:00Z"
    },
    metadata={"format": "json"}
)

part = types.Part(root=data_part)
```

**Fields**:
- `data` (dict, required): Any JSON-serializable dictionary
- `metadata` (dict, optional): Additional metadata
- `kind` (literal "data", default): Part type identifier

**Use cases**:
- Structured configuration
- API responses
- Analytical data
- Form submissions

## Complete Message Examples

### Simple Text Message

```python
message = types.Message(
    message_id="msg_001",
    role=types.Role.USER,
    parts=[
        types.Part(root=types.TextPart(
            text="What is the weather today?"
        ))
    ]
)
```

### Multi-Part Message

```python
message = types.Message(
    message_id="msg_002",
    role=types.Role.USER,
    parts=[
        # Text instruction
        types.Part(root=types.TextPart(
            text="Please analyze this document:"
        )),
        # Attached file
        types.Part(root=types.FilePart(
            file=types.FileWithBytes(
                bytes=base64_encoded_content,
                name="report.pdf",
                mime_type="application/pdf"
            )
        )),
        # Metadata
        types.Part(root=types.DataPart(
            data={
                "analysis_type": "summary",
                "language": "english"
            }
        ))
    ]
)
```

## Task and Context

### Context ID

The `context_id` groups messages in a conversation:

```python
# First message in conversation
msg1 = types.Message(
    message_id="msg_001",
    role=types.Role.USER,
    context_id="conv_abc123",  # New context
    parts=[...]
)

# Follow-up message in same conversation
msg2 = types.Message(
    message_id="msg_002",
    role=types.Role.USER,
    context_id="conv_abc123",  # Same context!
    task_id="task_from_msg1",  # Reference previous task
    parts=[...]
)
```

### Task ID

The `task_id` links messages to specific agent tasks:

```python
# User message creates a task
user_msg = types.Message(
    message_id="msg_001",
    role=types.Role.USER,
    parts=[...]
)

# Agent response references the task
agent_msg = types.Message(
    message_id="msg_002",
    role=types.Role.AGENT,
    task_id="task_123",  # Created by the agent
    context_id="conv_abc123",
    parts=[...]
)
```

## Metadata

Metadata provides flexibility for custom data:

```python
message = types.Message(
    message_id="msg_001",
    role=types.Role.USER,
    parts=[...],
    metadata={
        # Custom application data
        "user_id": "user_789",
        "session_id": "session_xyz",
        "priority": "high",
        "tags": ["urgent", "customer_support"],
        # Anything JSON-serializable
        "custom_data": {
            "nested": "values"
        }
    }
)
```

## Message Validation

Messages are validated using Pydantic:

```python
from pydantic import ValidationError

try:
    message = types.Message(
        message_id="msg_001",
        role=types.Role.USER,
        parts=[...]  # This is validated
    )
except ValidationError as e:
    print(f"Invalid message: {e}")
```

### Common Validation Rules

1. **message_id**: Must be a non-empty string
2. **role**: Must be "user" or "agent"
3. **parts**: Must be a non-empty list
4. **Each part**: Must be one of TextPart, FilePart, or DataPart

## Working with Message Parts

### Extracting Text from Messages

```python
def extract_text(message: types.Message) -> str:
    """Extract all text from a message."""
    text_parts = []
    for part in message.parts:
        if isinstance(part.root, types.TextPart):
            text_parts.append(part.root.text)
    return " ".join(text_parts)
```

### Finding Files in Messages

```python
def find_files(message: types.Message) -> list[types.FilePart]:
    """Find all file parts in a message."""
    files = []
    for part in message.parts:
        if isinstance(part.root, types.FilePart):
            files.append(part.root)
    return files
```

### Extracting Data Parts

```python
def extract_data(message: types.Message) -> list[dict]:
    """Extract all data parts from a message."""
    data_parts = []
    for part in message.parts:
        if isinstance(part.root, types.DataPart):
            data_parts.append(part.root.data)
    return data_parts
```

## Best Practices

### 1. Use Clear Message IDs

```python
# Good: Descriptive and unique
message_id = f"msg_{uuid.uuid4()}"

# Also good: Sequential with context
message_id = f"msg_{context_id}_{sequence_number}"

# Avoid: Non-unique or unclear
message_id = "message1"  # Too generic
```

### 2. Choose the Right Part Type

- **Text**: For natural language content
- **File**: For documents, images, binaries
- **Data**: For structured configuration or API data

### 3. Keep Messages Focused

```python
# Good: Clear, single-purpose message
message = types.Message(
    message_id="msg_001",
    role=types.Role.USER,
    parts=[
        types.Part(root=types.TextPart(
            text="Summarize this document"
        )),
        types.Part(root=types.FilePart(file=...))
    ]
)

# Avoid: Mixing unrelated content
# (unless they're genuinely related)
```

### 4. Use Metadata Wisely

```python
# Good: Relevant, structured metadata
metadata = {
    "intent": "question",
    "priority": "normal",
    "language": "en"
}

# Avoid: Putting essential data in metadata
# (use proper parts instead)
metadata = {
    "main_content": "..."  # Should be a TextPart
}
```

### 5. Handle File Sizes Appropriately

```python
# For small files (< 10MB)
if file_size < 10 * 1024 * 1024:
    file = types.FileWithBytes(bytes=base64_encoded)
else:
    # For large files
    file = types.FileWithUri(uri=uploaded_url)
```

## Examples in Practice

See the following example files for complete implementations:
- [01_simple_agent_server.py](../examples/01_simple_agent_server.py)
- [05_document_processing_agent.py](../examples/05_document_processing_agent.py)
- [06_document_processing_client.py](../examples/06_document_processing_client.py)

## Related Guides
- [Server Implementation Guide](02_server_implementation.md)
- [Client Implementation Guide](03_client_implementation.md)
- [File Handling Guide](05_file_handling.md)
